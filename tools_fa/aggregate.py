"""Aggregate per-seed metrics and clutter-stratified false alarms.

For every run directory <work-root>/<config><tag>_s<seed>/ with metrics.json
and test_results.pkl, computes mean and std over seeds and, at the recall-0.8
operating point, the false-alarm image rate per clutter category.

Usage: python tools_fa/aggregate.py [--work-root ...] [--latex out.tex]
"""
import argparse
import glob
import json
import os.path as osp
import pickle
import re
from collections import defaultdict

import numpy as np

from fa_eval import evaluate, load_coco
from paths import WORK, ann

ANN = ann('test')
COLUMNS = ['AP', 'AP50', 'AP75', 'AP50_pos', 'dAP50_neg', 'R@FPPI0.01',
           'R@FPPI0.1', 'LAMR', 'FPPIneg@R0.8', 'FAimg@R0.8', 'FAimg@R0.9',
           'NegFPshare@R0.8', 'AP50_noidfix']


def clutter_fa(results, features, recall=0.8):
    """Fraction of negative tiles per category with a detection above the
    score threshold that reaches the given recall."""
    metrics, curves = evaluate(ANN, results)
    idx = np.where(curves['recall'] >= recall)[0]
    if not len(idx):
        return {}
    thr = curves['scores'][idx[0]]
    img_ids = load_coco(ANN).getImgIds()  # same order as mmdet results
    out = defaultdict(lambda: [0, 0])
    for img_id, res in zip(img_ids, results):
        f = features[str(img_id)]
        if not f['negative']:
            continue
        hit = len(res[0]) and (res[0][:, 4] >= thr).any()
        out[f['category']][0] += int(bool(hit))
        out[f['category']][1] += 1
    return {c: v[0] / v[1] for c, v in out.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work-root', default=WORK)
    parser.add_argument('--latex')
    parser.add_argument('--no-clutter', action='store_true')
    args = parser.parse_args()

    features = json.load(open(osp.join(args.work_root, 'scene_features.json')))
    runs = defaultdict(list)
    for m in sorted(glob.glob(osp.join(args.work_root, '*', 'metrics.json'))):
        run = osp.basename(osp.dirname(m))
        group = re.sub(r'_s\d+$', '', run)
        metrics = json.load(open(m))
        pkl = osp.join(osp.dirname(m), 'test_results.pkl')
        if not args.no_clutter and osp.exists(pkl):
            with open(pkl, 'rb') as f:
                for cat, v in clutter_fa(pickle.load(f), features).items():
                    metrics[f'FA_{cat}'] = v
        runs[group].append(metrics)

    cols = COLUMNS + ['FA_clear_sea', 'FA_texture', 'FA_cloud']
    summary = {}
    header = f"{'group':28s} n " + ' '.join(f'{c[:12]:>13s}' for c in cols)
    print(header)
    for group, ms in sorted(runs.items()):
        row = {}
        for c in cols:
            vals = [m[c] for m in ms if c in m and m[c] == m[c]]
            if vals:
                row[c] = (float(np.mean(vals)), float(np.std(vals)))
        summary[group] = dict(n=len(ms), **row)
        print(f'{group:28s} {len(ms)} ' + ' '.join(
            f'{row[c][0] * 100:6.1f}±{row[c][1] * 100:4.1f}' if c in row
            else f"{'-':>13s}" for c in cols))
    json.dump(summary, open(osp.join(args.work_root, 'summary.json'), 'w'),
              indent=1)

    if args.latex:
        with open(args.latex, 'w') as f:
            for group, row in sorted(summary.items()):
                cells = [group.replace('_', r'\_')]
                for c in cols:
                    cells.append(f'{row[c][0] * 100:.1f}$\\pm${row[c][1] * 100:.1f}'
                                 if c in row else '--')
                f.write(' & '.join(cells) + r' \\' + '\n')


if __name__ == '__main__':
    main()
