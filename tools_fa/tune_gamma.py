"""Choose the calibration strength gamma on the validation split.

Calibration multiplies every score in a tile by p^gamma, where p is the scene
presence probability.  Because that factor is constant within a tile it never
reorders detections, so a run evaluated with gamma=0 plus the per-tile
probabilities determines the result for every gamma exactly -- no re-inference
per value.

gamma is selected on the validation split, which nothing else in this study
uses, and applied once to test.  Selecting it on test would be tuning on the
number we report.

Usage:
  python tools_fa/tune_gamma.py RUN [--gammas 0 0.5 1 2 3] [--criterion LAMR]
"""
import argparse
import glob
import json
import os.path as osp
import pickle
import subprocess
import sys

import numpy as np

from fa_eval import evaluate, load_coco
from paths import WORK, ann, img
from presence_hist import presence_probabilities, run_dir

PY = sys.executable
ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
# lower is better for these, higher for the rest
LOWER_IS_BETTER = {'LAMR', 'FAimg@R0.8', 'FAimg@R0.9',
                   'FPPIneg@R0.8', 'FPPIneg@R0.9'}


def uncalibrated(work_root, run, split, config=None, ckpt=None, seed=0):
    """Detections on `split` with calibration off, computed once and cached.

    A post-hoc run's output directory holds results, not a checkpoint or a
    config; those come from the run it was fitted on, so both can be given.
    """
    wd = osp.join(work_root, f'{run}_s{seed}')
    if not osp.isdir(wd):
        wd = run_dir(work_root, run)
    pkl = osp.join(wd, f'{split}_results_g0.pkl')
    if osp.exists(pkl):
        return pickle.load(open(pkl, 'rb'))
    if config is None:
        hits = [f for f in sorted(glob.glob(osp.join(wd, '*.py')))
                if not osp.basename(f).startswith('posthoc_cfg')]
        if not hits:
            raise SystemExit(f'no config in {wd}; pass --config')
        config = hits[0]
    cfg = config
    cmd = [PY, 'tools/test.py', cfg, ckpt or osp.join(wd, 'latest.pth'),
           '--out', pkl,
           '--cfg-options', 'model.bbox_head.presence.gamma=0',
           f'data.test.ann_file={ann(split)}',
           f'data.test.img_prefix={img(split)}']
    if subprocess.run(cmd, cwd=ROOT).returncode != 0:
        raise SystemExit('inference failed for ' + run)
    return pickle.load(open(pkl, 'rb'))


def apply_gamma(results, probs, gamma):
    """s * p^gamma, per image. Ranking inside an image is unchanged."""
    out = []
    for res, p in zip(results, probs):
        factor = float(p) ** gamma
        boxes = res[0].copy()
        if len(boxes):
            boxes[:, 4] = boxes[:, 4] * factor
        out.append([boxes])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run')
    parser.add_argument('--gammas', type=float, nargs='+',
                        default=[0, 0.25, 0.5, 1, 1.5, 2, 3, 4])
    parser.add_argument('--criterion', default='LAMR')
    parser.add_argument('--work-root', default=WORK)
    parser.add_argument('--config', help='config whose head has the branch')
    parser.add_argument('--ckpt', help='checkpoint holding the fitted branch')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    rows = {}
    for split in ('val', 'test'):
        results = uncalibrated(args.work_root, args.run, split,
                               config=args.config, ckpt=args.ckpt,
                               seed=args.seed)
        probs, _ = presence_probabilities(args.work_root, args.run,
                                          split=split, config=args.config,
                                          ckpt=args.ckpt)
        assert len(probs) == len(results), (len(probs), len(results))
        rows[split] = {}
        for g in args.gammas:
            m, _ = evaluate(ann(split), apply_gamma(results, probs, g))
            rows[split][g] = m

    keys = ['AP50', 'LAMR', 'FAimg@R0.8', 'FAimg@R0.9', 'FPPIneg@R0.9',
            'R@FPPI0.1']
    for split in ('val', 'test'):
        print(f'\n{split}:')
        print(f"{'gamma':>7s}" + ''.join(f'{k:>14s}' for k in keys))
        for g in args.gammas:
            print(f'{g:7.2f}' + ''.join(f'{rows[split][g][k] * 100:14.2f}'
                                        for k in keys))

    c = args.criterion
    vals = {g: rows['val'][g][c] for g in args.gammas}
    best = (min if c in LOWER_IS_BETTER else max)(vals, key=vals.get)
    print(f'\nselected on val by {c}: gamma = {best}')
    print(f'  val  {c} = {vals[best] * 100:.2f}')
    print(f'  test {c} = {rows["test"][best][c] * 100:.2f}   '
          + '  '.join(f'{k}={rows["test"][best][k] * 100:.2f}' for k in keys))
    json.dump(dict(run=args.run, criterion=c, selected_gamma=best,
                   val={str(g): rows['val'][g] for g in args.gammas},
                   test={str(g): rows['test'][g] for g in args.gammas}),
              open(osp.join(args.work_root,
                                f'gamma_{args.run}_s{args.seed}.json'), 'w'),
              indent=1)


if __name__ == '__main__':
    main()
