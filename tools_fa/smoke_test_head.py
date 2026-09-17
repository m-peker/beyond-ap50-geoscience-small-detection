"""CPU smoke test: every FAFCOSHead variant builds, trains one step and tests.

Checks loss finiteness on a batch mixing a positive and a negative image, and
that presence calibration never raises a box score.

Also rebuilds each variant the way tools/test.py does, with train_cfg stripped,
and runs inference on it.  Training and deployment are separate build paths and
a head can work in one and fail in the other, which is only discovered after a
run has already spent its GPU hours.
"""
import sys

import torch
from mmcv import Config

from mmdet.models import build_detector

VARIANTS = {
    'fcos': {},
    'rfla': dict(cfg='rfla_p2_all.py'),
    'nwd': dict(cfg='nwd_p2_all.py'),
    'presence_ema': dict(presence=dict(topk=4, loss_weight=1.0, gamma=1.0),
                         cls_norm='ema'),
}


def build(name, spec, for_inference=False):
    cfg = Config.fromfile('configs_fa/' + spec.get('cfg', 'fcos_p2_all.py'))
    cfg.model.backbone.init_cfg = None
    for key in ('presence', 'cls_norm'):
        if key in spec:
            cfg.model.bbox_head[key] = spec[key]
    if for_inference:  # exactly what tools/test.py does
        test_cfg = cfg.model.pop('test_cfg', None)
        cfg.model.train_cfg = None
        model = build_detector(cfg.model, test_cfg=test_cfg)
    else:
        model = build_detector(cfg.model)
    model.init_weights()
    return model


def main():
    torch.manual_seed(0)
    img = torch.randn(2, 3, 512, 512)
    metas = [
        dict(img_shape=(512, 512, 3), ori_shape=(512, 512, 3),
             pad_shape=(512, 512, 3), scale_factor=1.0, flip=False,
             batch_input_shape=(512, 512)) for _ in range(2)
    ]
    gt_bboxes = [torch.tensor([[100., 100., 112., 110.], [300., 40., 318., 52.]]),
                 torch.zeros((0, 4))]
    gt_labels = [torch.tensor([0, 0]), torch.zeros((0, ), dtype=torch.long)]

    ok = True
    for name, spec in VARIANTS.items():
        model = build(name, spec)
        model.train()
        losses = model.forward_train(img, metas, gt_bboxes, gt_labels)
        total = sum(v for v in losses.values())
        total.backward()
        finite = all(torch.isfinite(v).all() for v in losses.values())
        model.eval()
        with torch.no_grad():
            dets = model.simple_test(img, metas, rescale=False)
        n_det = [len(d[0]) for d in dets]
        print(f'{name:14s} finite={finite} '
              + ' '.join(f'{k}={float(v):.4f}' for k, v in losses.items())
              + f' dets={n_det}')
        ok &= bool(finite)

        # the deployment build path, which strips train_cfg
        infer = build(name, spec, for_inference=True)
        infer.eval()
        with torch.no_grad():
            infer_dets = infer.simple_test(img, metas, rescale=False)
        print(f'{"":14s} inference-only build ok, dets={[len(d[0]) for d in infer_dets]}')

        if name == 'presence_ema':
            head = model.bbox_head
            # the factor is per image AND per class: (num_images, num_classes)
            logits = torch.randn(2, 1, 8, 8)
            factor = torch.log(torch.tensor([[0.5], [0.01]]))
            scaled = head._scale_logits(logits, factor)
            ratio = scaled.sigmoid() / logits.sigmoid()
            print('  calibration ratio per image:',
                  [round(float(r.mean()), 4) for r in ratio])
            ok &= scaled.shape == logits.shape
            ok &= bool((scaled.sigmoid() <= logits.sigmoid() + 1e-6).all())

            # multi-class: each class must be scaled by its own factor only
            multi = torch.randn(2, 3, 8, 8)
            mfactor = torch.log(torch.tensor([[1.0, 0.5, 0.1],
                                              [0.1, 1.0, 0.5]]))
            mscaled = head._scale_logits(multi, mfactor)
            mratio = (mscaled.sigmoid() / multi.sigmoid()).mean(dim=(2, 3))
            print('  per-class ratio:',
                  [[round(float(v), 3) for v in row] for row in mratio])
            expect = torch.tensor([[1.0, 0.5, 0.1], [0.1, 1.0, 0.5]])
            ok &= bool((mratio - expect).abs().max() < 0.02)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
