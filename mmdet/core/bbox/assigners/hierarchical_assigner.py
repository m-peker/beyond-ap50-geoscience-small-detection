# Hierarchical label assignment (HLA) from RFLA (Xu et al., ECCV 2022).
# Faithful port of github.com/Chasel-Tsui/mmdet-rfla (mmdet 2.13) to mmdet 2.24.1:
# identical ranking logic, unused experimental branches removed.
import torch

from ..builder import BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


@BBOX_ASSIGNERS.register_module()
class HieAssigner(BaseAssigner):
    """Two-stage top-k ranking assignment on a Gaussian similarity metric.

    Stage 1 assigns each gt its top-k1 priors. Stage 2 shrinks priors by
    `ratio` and assigns each gt its top-k2 priors among those left unassigned.

    Returns gt_inds: 0 = background, i+1 = positive for gt i.
    """

    def __init__(self,
                 ignore_iof_thr=-1,
                 gpu_assign_thr=-1,
                 iou_calculator=dict(type='BboxDistanceMetric'),
                 assign_metric='kl',
                 topk=(3, 1),
                 ratio=0.9):
        self.ignore_iof_thr = ignore_iof_thr
        self.gpu_assign_thr = gpu_assign_thr
        self.iou_calculator = build_iou_calculator(iou_calculator)
        self.assign_metric = assign_metric
        self.topk = topk
        self.ratio = ratio

    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        assign_on_cpu = (self.gpu_assign_thr > 0
                         and gt_bboxes.shape[0] > self.gpu_assign_thr)
        device = bboxes.device
        if assign_on_cpu:
            bboxes, gt_bboxes = bboxes.cpu(), gt_bboxes.cpu()
            if gt_labels is not None:
                gt_labels = gt_labels.cpu()

        overlaps = self.iou_calculator(
            gt_bboxes, bboxes, mode=self.assign_metric)
        overlaps2 = self.iou_calculator(
            gt_bboxes, self._rescale(bboxes, self.ratio),
            mode=self.assign_metric)

        k1, k2 = self.topk
        assigned_gt_inds = self._assign_wrt_ranking(overlaps, k1)
        result = self._reassign_wrt_ranking(assigned_gt_inds, overlaps2, k2,
                                            gt_labels)
        if assign_on_cpu:
            result.gt_inds = result.gt_inds.to(device)
            result.max_overlaps = result.max_overlaps.to(device)
            if result.labels is not None:
                result.labels = result.labels.to(device)
        return result

    @staticmethod
    def _rank(overlaps, k):
        num_gts, num_bboxes = overlaps.shape
        assigned = overlaps.new_full((num_bboxes, ), -1, dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            assigned[:] = 0
            return assigned, overlaps.new_zeros((num_bboxes, ))
        max_overlaps, _ = overlaps.max(dim=0)
        gt_max_overlaps, _ = overlaps.topk(k, dim=1, largest=True, sorted=True)
        assigned[(max_overlaps >= 0) & (max_overlaps < 0.8)] = 0
        for i in range(num_gts):
            for j in range(k):
                assigned[overlaps[i, :] == gt_max_overlaps[i, j]] = i + 1
        return assigned, max_overlaps

    def _assign_wrt_ranking(self, overlaps, k):
        return self._rank(overlaps, k)[0]

    def _reassign_wrt_ranking(self, first, overlaps, k, gt_labels=None):
        num_gts = overlaps.size(0)
        second, max_overlaps = self._rank(overlaps, k)
        if num_gts > 0 and overlaps.size(1) > 0:
            second = torch.where(first > 0, first, second)
        labels = None
        if gt_labels is not None:
            labels = second.new_full((second.numel(), ), -1)
            pos = second > 0
            labels[pos] = gt_labels[second[pos] - 1]
        return AssignResult(num_gts, second, max_overlaps, labels=labels)

    @staticmethod
    def _rescale(bboxes, ratio):
        cx = (bboxes[:, 2] + bboxes[:, 0]) / 2
        cy = (bboxes[:, 3] + bboxes[:, 1]) / 2
        w = (bboxes[:, 2] - bboxes[:, 0]) * ratio
        h = (bboxes[:, 3] - bboxes[:, 1]) * ratio
        return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)
