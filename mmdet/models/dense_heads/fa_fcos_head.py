# False-alarm-aware FCOS head for tiny object detection.
#
# One head covers every configuration studied in this project:
#   * label assignment: FCOS center sampling (default) or receptive-field
#     ranking assignment (RFLA / NWD-style) when `rf_assign` is given;
#   * scene presence branch: an image-level "object present" classifier
#     pooled from the dense classification tower, used at test time to
#     calibrate box scores as s * p^gamma;
#   * classification loss normalizer: per-batch positives (mmdet default)
#     or an exponential moving average, which keeps the loss scale stable
#     when a batch contains only negative (object-free) images.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32

from mmdet.core import build_assigner, reduce_mean
from ..builder import HEADS
from .fcos_head import FCOSHead

INF = 1e8


def resnet50_fpn_trf():
    """Theoretical receptive field of P2-P7 in ResNet-50-FPN (as in RFLA)."""
    j = [2**i for i in range(8)]
    r1 = 1 + (7 - 1) * j[0]
    r2 = r1 + (3 - 1) * j[1]
    p2 = r2 + (3 - 1) * j[2] * 3
    r3 = p2 + (3 - 1) * j[2]
    p3 = r3 + (3 - 1) * j[3] * 3
    r4 = p3 + (3 - 1) * j[3]
    p4 = r4 + (3 - 1) * j[4] * 5
    r5 = p4 + (3 - 1) * j[4]
    p5 = r5 + (3 - 1) * j[5] * 2
    p6 = p5 + (3 - 1) * j[6]
    p7 = p6 + (3 - 1) * j[7]
    return [p2, p3, p4, p5, p6, p7]


@HEADS.register_module()
class FAFCOSHead(FCOSHead):
    """FCOS head with optional RF assignment, presence branch and EMA norm.

    Args:
        rf_assign (dict | None): dict(fpn_layer='p2', fraction=0.5). When set,
            points are replaced by receptive-field boxes and assigned with
            `train_cfg.assigner` (e.g. HieAssigner), as in RFLA.
        presence (dict | None): dict(topk=4, loss_weight=1.0, gamma=1.0,
            detach=False). gamma=0 disables test-time calibration while
            keeping the loss; detach=True stops presence gradients from
            reaching the shared classification tower, so the branch is a
            probe on detection features rather than an influence on them.
        cls_norm (str): 'batch' or 'ema'.
        ema_momentum (float): momentum of the positive-count EMA.
    """

    def __init__(self,
                 *args,
                 rf_assign=None,
                 presence=None,
                 cls_norm='batch',
                 ema_momentum=0.9,
                 **kwargs):
        self.rf_assign = rf_assign
        self.presence = presence
        assert cls_norm in ('batch', 'ema')
        self.cls_norm = cls_norm
        self.ema_momentum = ema_momentum
        super().__init__(*args, **kwargs)
        if rf_assign is not None:
            # train_cfg is None when the model is built for inference, where
            # the assigner is never used; requiring it there would make every
            # receptive-field configuration untestable.
            self.assigner = (build_assigner(self.train_cfg.assigner)
                             if self.train_cfg is not None else None)
            offset = 0 if rf_assign.get('fpn_layer', 'p2') == 'p2' else 1
            trfs = resnet50_fpn_trf()
            self.rf_sizes = [
                trfs[i + offset] * rf_assign.get('fraction', 0.5)
                for i in range(len(self.strides))
            ]
        self.register_buffer('num_pos_ema', torch.tensor(-1.0))

    def _init_layers(self):
        super()._init_layers()
        if self.presence is not None:
            # one channel per class: on a single-class benchmark this is the
            # scalar scene-presence logit, and on a multi-class one it is the
            # per-class question "does this image contain any object of class c",
            # which is what per-class negatives make answerable.
            self.conv_presence = nn.Conv2d(self.feat_channels,
                                           self.cls_out_channels, 1)

    def forward(self, feats):
        cls_scores, bbox_preds, centernesses, presence_maps = [], [], [], []
        for x, scale, stride in zip(feats, self.scales, self.strides):
            cls_feat = x
            for conv in self.cls_convs:
                cls_feat = conv(cls_feat)
            cls_score = self.conv_cls(cls_feat)

            reg_feat = x
            for conv in self.reg_convs:
                reg_feat = conv(reg_feat)
            bbox_pred = self.conv_reg(reg_feat)

            centerness = self.conv_centerness(
                reg_feat if self.centerness_on_reg else cls_feat)
            bbox_pred = scale(bbox_pred).float()
            if self.norm_on_bbox:
                bbox_pred = bbox_pred.clamp(min=0)
                if not self.training:
                    bbox_pred *= stride
            else:
                bbox_pred = bbox_pred.exp()

            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
            centernesses.append(centerness)
            if self.presence is None:
                presence_maps.append(None)
            else:
                feat = cls_feat
                if self.presence.get('detach', False):
                    feat = feat.detach()
                presence_maps.append(self.conv_presence(feat))
        return cls_scores, bbox_preds, centernesses, presence_maps

    def image_presence_logits(self, presence_maps):
        """Top-k mean pooling of dense presence logits, per class.

        Returns (num_images, num_classes). Pooling is per class, so a scene
        holding one class does not raise the evidence for another.
        """
        num_imgs = presence_maps[0].size(0)
        num_cls = presence_maps[0].size(1)
        flat = torch.cat(
            [m.reshape(num_imgs, num_cls, -1) for m in presence_maps], 2)
        k = min(self.presence.get('topk', 4), flat.size(2))
        return flat.topk(k, dim=2)[0].mean(dim=2)

    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'centernesses', 'presence_maps'))
    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             presence_maps,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)
        labels, bbox_targets = self.get_targets(all_level_points, gt_bboxes,
                                                gt_labels)

        num_imgs = cls_scores[0].size(0)
        flatten_cls_scores = torch.cat([
            s.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for s in cls_scores
        ])
        flatten_bbox_preds = torch.cat(
            [b.permute(0, 2, 3, 1).reshape(-1, 4) for b in bbox_preds])
        flatten_centerness = torch.cat(
            [c.permute(0, 2, 3, 1).reshape(-1) for c in centernesses])
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = reduce_mean(
            torch.tensor(
                len(pos_inds), dtype=torch.float,
                device=bbox_preds[0].device))
        loss_cls = self.loss_cls(
            flatten_cls_scores,
            flatten_labels,
            avg_factor=self._cls_avg_factor(num_pos))

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_centerness_targets = self.centerness_target(pos_bbox_targets)
        centerness_denorm = max(
            reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = self.bbox_coder.decode(
                pos_points, pos_bbox_preds)
            pos_decoded_target_preds = self.bbox_coder.decode(
                pos_points, pos_bbox_targets)
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=centerness_denorm)
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets,
                avg_factor=max(num_pos, 1.0))
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()

        losses = dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness)

        if self.presence is not None:
            img_logits = self.image_presence_logits(presence_maps)
            img_targets = img_logits.new_zeros(img_logits.shape)
            for i, labels in enumerate(gt_labels):
                if len(labels):
                    img_targets[i, labels.unique()] = 1.0
            losses['loss_presence'] = self.presence.get(
                'loss_weight', 1.0) * F.binary_cross_entropy_with_logits(
                    img_logits, img_targets)
        return losses

    def _cls_avg_factor(self, num_pos):
        if self.cls_norm == 'batch':
            return max(num_pos, 1.0)
        if self.training:
            value = num_pos.detach()
            if self.num_pos_ema < 0:
                self.num_pos_ema.fill_(float(value))
            else:
                self.num_pos_ema.mul_(self.ema_momentum).add_(
                    (1 - self.ema_momentum) * value)
        return max(float(self.num_pos_ema), 1.0)

    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'centernesses', 'presence_maps'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   centernesses,
                   presence_maps,
                   img_metas=None,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        gamma = 0.0 if self.presence is None else self.presence.get(
            'gamma', 1.0)
        # gamma may be one value per class, which is how a class whose branch
        # does not separate on validation data is left uncalibrated instead of
        # being suppressed by a branch that never fires for it.
        if isinstance(gamma, (list, tuple)):
            gamma = cls_scores[0].new_tensor(gamma)
            assert gamma.numel() == self.cls_out_channels, gamma.shape
            active = bool((gamma > 0).any())
        else:
            active = gamma > 0
        if active:
            # (num_images, num_classes), applied to the matching score channel
            log_p = F.logsigmoid(self.image_presence_logits(presence_maps))
            cls_scores = [
                self._scale_logits(s, gamma * log_p) for s in cls_scores
            ]
        return super().get_bboxes(
            cls_scores,
            bbox_preds,
            centernesses,
            img_metas=img_metas,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms)

    @staticmethod
    def _scale_logits(logits, log_factor):
        """Logits of sigmoid(logits) * exp(log_factor), per image and class.

        log_factor is (num_images, num_classes) and broadcasts over the spatial
        dimensions of a (num_images, num_classes, H, W) score map.
        """
        log_s = F.logsigmoid(logits) + log_factor[..., None, None]
        log_s = log_s.clamp(max=-1e-6)
        return log_s - torch.log1p(-torch.exp(log_s))

    def centerness_target(self, pos_bbox_targets):
        if self.rf_assign is None:
            return super().centerness_target(pos_bbox_targets)
        # RF assignment may pick points outside the gt box; clamp as RFLA does
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        if len(left_right) == 0:
            return left_right[..., 0]
        centerness_targets = (
            left_right.min(dim=-1)[0].clamp(min=0.01) /
            left_right.max(dim=-1)[0]) * (
                top_bottom.min(dim=-1)[0].clamp(min=0.01) /
                top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)

    def get_targets(self, points, gt_bboxes_list, gt_labels_list):
        if self.rf_assign is None:
            return super().get_targets(points, gt_bboxes_list, gt_labels_list)
        assert self.assigner is not None, (
            'receptive-field assignment needs train_cfg.assigner; the head was '
            'built for inference')

        num_levels = len(points)
        num_points = [p.size(0) for p in points]
        rfields = torch.cat([
            torch.cat([p - rf / 2, p + rf / 2], dim=1)
            for p, rf in zip(points, self.rf_sizes)
        ])
        concat_points = torch.cat(points, dim=0)

        labels_list, bbox_targets_list = [], []
        for gt_bboxes, gt_labels in zip(gt_bboxes_list, gt_labels_list):
            labels, targets = self._get_rf_target_single(
                gt_bboxes, gt_labels, concat_points, rfields)
            labels_list.append(labels.split(num_points, 0))
            bbox_targets_list.append(targets.split(num_points, 0))

        concat_lvl_labels, concat_lvl_bbox_targets = [], []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [targets[i] for targets in bbox_targets_list])
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
        return concat_lvl_labels, concat_lvl_bbox_targets

    def _get_rf_target_single(self, gt_bboxes, gt_labels, points, rfields):
        num_points, num_gts = points.size(0), gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_full((num_points, ), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4))

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1])
        areas = areas[None].repeat(num_points, 1)
        xs = points[:, 0:1].expand(num_points, num_gts)
        ys = points[:, 1:2].expand(num_points, num_gts)
        gts = gt_bboxes[None].expand(num_points, num_gts, 4)
        bbox_targets = torch.stack((xs - gts[..., 0], ys - gts[..., 1],
                                    gts[..., 2] - xs, gts[..., 3] - ys), -1)

        inds = self.assigner.assign(rfields, gt_bboxes).gt_inds - 1
        assigned = inds[:, None] == torch.arange(
            num_gts, device=inds.device)[None]
        areas[~assigned] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes
        bbox_targets = bbox_targets[range(num_points), min_area_inds]
        return labels, bbox_targets
