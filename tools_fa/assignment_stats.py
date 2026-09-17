"""Positive-sample statistics of each label assignment on LEVIR-Ship train.

For every ground-truth ship: number of positive locations, how many of them lie
outside the ship box, and which pyramid level they come from. Runs on CPU with
the exact target functions used in training.
"""
import json
import os.path as osp
from collections import defaultdict

import numpy as np
import torch
from mmcv import Config

from mmdet.models import build_head
from paths import WORK, ann

ANN = ann('train')
STRIDES = [4, 8, 16, 32, 64]


def build(cfg_file):
    cfg = Config.fromfile(cfg_file)
    head_cfg = cfg.model.bbox_head
    head_cfg.update(train_cfg=cfg.model.train_cfg, test_cfg=cfg.model.test_cfg)
    return build_head(head_cfg)


def main():
    ann = json.load(open(ANN))
    by_img = defaultdict(list)
    for a in ann['annotations']:
        x, y, w, h = a['bbox']
        by_img[a['image_id']].append([x, y, x + w, y + h])

    heads = {name: build(f'configs_fa/{name}_p2_all.py')
             for name in ('fcos', 'rfla', 'nwd')}
    sizes = [(512 // s, 512 // s) for s in STRIDES]
    points = heads['fcos'].prior_generator.grid_priors(
        sizes, dtype=torch.float32, device='cpu')
    level_of = torch.cat([torch.full((len(p), ), i) for i, p in enumerate(points)])
    flat_points = torch.cat(points)

    stats = {}
    for name, head in heads.items():
        per_gt = []
        for boxes in by_img.values():
            gt = torch.tensor(boxes, dtype=torch.float32)
            labels, _ = head.get_targets(points, [gt], [torch.zeros(len(gt), dtype=torch.long)])
            labels = torch.cat(labels)
            pos = torch.nonzero(labels == 0).reshape(-1)
            # recover the gt each positive regresses to (min-area rule)
            _, targets = head.get_targets(points, [gt], [torch.zeros(len(gt), dtype=torch.long)])
            for g in range(len(gt)):
                x1, y1, x2, y2 = gt[g]
                p = flat_points[pos]
                d = torch.stack([p[:, 0] - x1, p[:, 1] - y1, x2 - p[:, 0], y2 - p[:, 1]], 1)
                # positives whose regression target equals this gt
                tgt = torch.cat(targets)[pos]
                lvl_stride = torch.tensor(STRIDES, dtype=torch.float32)[level_of[pos]]
                if head.norm_on_bbox:
                    tgt = tgt * lvl_stride[:, None]
                mine = (tgt - d).abs().max(1)[0] < 1e-3
                n = int(mine.sum())
                outside = int((mine & (d.min(1)[0] <= 0)).sum())
                levels = level_of[pos][mine].tolist()
                size = float(((x2 - x1) * (y2 - y1)).sqrt())
                per_gt.append((size, n, outside, levels))
        sz = np.array([s for s, *_ in per_gt])
        n = np.array([k for _, k, _, _ in per_gt])
        out = np.array([o for _, _, o, _ in per_gt])
        lv = [l for *_, ls in per_gt for l in ls]
        bins = [(0, 16), (16, 24), (24, 1e9)]
        stats[name] = dict(
            mean_pos=float(n.mean()), zero_pos_frac=float((n == 0).mean()),
            outside_frac=float(out.sum() / max(n.sum(), 1)),
            level_hist={STRIDES[i]: lv.count(i) / max(len(lv), 1) for i in range(5)},
            by_size={f'{lo:g}-{hi:g}': float(n[(sz >= lo) & (sz < hi)].mean())
                     for lo, hi in bins})
        print(name, json.dumps(stats[name]))
    json.dump(stats, open(osp.join(WORK, 'assignment_stats.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
