"""Verify the HieAssigner port matches the official RFLA implementation.

Loads the original mmdet-rfla sources (mmdet 2.13) from <TOD_ROOT>/ext and compares
gt assignments on random tiny-object layouts with RF boxes from a P2-P6 grid.
"""
import os.path as osp
import sys
import types

import torch

from mmdet.core.bbox.assigners.assign_result import AssignResult
from mmdet.core.bbox.assigners.base_assigner import BaseAssigner
from mmdet.core.bbox.assigners.hierarchical_assigner import HieAssigner
from mmdet.models.dense_heads.fa_fcos_head import resnet50_fpn_trf
from paths import EXT

RFLA = osp.join(EXT, 'mmdet-rfla', 'mmdet') + '/'


def load_original():
    metric_ns = {'torch': torch, 'math': __import__('math')}
    src = open(RFLA + 'core/bbox/iou_calculators/metric_calculator.py').read()
    src = src.replace('from .builder import IOU_CALCULATORS', '')
    src = src.replace('@IOU_CALCULATORS.register_module()', '')
    exec(src, metric_ns)

    ns = {'torch': torch, 'random': __import__('random'),
          'AssignResult': AssignResult, 'BaseAssigner': BaseAssigner,
          'build_iou_calculator': lambda cfg: metric_ns['BboxDistanceMetric']()}
    src = open(RFLA + 'core/bbox/assigners/hierarchical_assigner.py').read()
    for line in ('from operator import gt', 'from turtle import back',
                 'from ..builder import BBOX_ASSIGNERS',
                 'from ..iou_calculators import build_iou_calculator',
                 'from .assign_result import AssignResult',
                 'from .base_assigner import BaseAssigner',
                 '@BBOX_ASSIGNERS.register_module()'):
        src = src.replace(line, '')
    exec(src, ns)
    return ns['HieAssigner'], resnet50_fpn_trf()


def rf_boxes(img=512, strides=(4, 8, 16, 32, 64), fraction=0.5):
    trfs = resnet50_fpn_trf()
    out = []
    for i, s in enumerate(strides):
        n = img // s
        ys, xs = torch.meshgrid(torch.arange(n), torch.arange(n))
        pts = torch.stack([xs.reshape(-1), ys.reshape(-1)], 1).float() * s + s // 2
        rf = trfs[i] * fraction
        out.append(torch.cat([pts - rf / 2, pts + rf / 2], 1))
    return torch.cat(out)


def main():
    OrigHie, _ = load_original()
    torch.manual_seed(0)
    boxes = rf_boxes()
    mism = 0
    for trial in range(50):
        n = torch.randint(1, 12, (1, )).item()
        xy = torch.rand(n, 2) * 480
        wh = 4 + torch.rand(n, 2) * 36
        gts = torch.cat([xy, xy + wh], 1)
        for metric in ('kl', 'wd'):
            orig = OrigHie(ignore_iof_thr=-1, gpu_assign_thr=256,
                           assign_metric=metric, topk=[3, 1], ratio=0.9)
            mine = HieAssigner(ignore_iof_thr=-1, gpu_assign_thr=256,
                               assign_metric=metric, topk=[3, 1], ratio=0.9)
            # the original rescales boxes in place; give each a copy
            a = orig.assign(boxes.clone(), gts.clone()).gt_inds
            b = mine.assign(boxes.clone(), gts.clone()).gt_inds
            mism += int((a != b).sum())
    print('total mismatched assignments over 50 layouts:', mism)
    sys.exit(0 if mism == 0 else 1)


if __name__ == '__main__':
    main()
