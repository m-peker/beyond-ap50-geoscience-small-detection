"""Soft calibration against a hard scene gate, on the same presence branch.

A reviewer will ask why the scene evidence is applied as a multiplicative
factor rather than as a decision: discard every detection in a tile whose
presence probability falls below a threshold.  The two use identical evidence,
so the comparison isolates the form of the intervention.

Both are post-processing on saved detections, so the whole sweep costs one
inference pass for the probabilities and no retraining.

Usage:
  python tools_fa/gate_vs_calibrate.py RUN --ckpt CKPT [--config CFG]
"""
import argparse
import json
import os.path as osp
import pickle

import numpy as np

from fa_eval import evaluate
from paths import WORK, ann
from presence_hist import presence_probabilities, run_dir
from tune_gamma import apply_gamma, uncalibrated

GAMMAS = [0.5, 1, 2, 4]
GATES = [0.1, 0.3, 0.5, 0.7, 0.9]
KEYS = ['AP50', 'LAMR', 'FAimg@R0.8', 'FAimg@R0.9', 'FPPIneg@R0.9',
        'R@FPPI0.1', 'F1max', 'maxRecall']


def apply_gate(results, probs, tau):
    """Drop every detection in a tile the branch judges empty."""
    out = []
    for res, p in zip(results, probs):
        boxes = res[0]
        out.append([boxes if float(p) >= tau else boxes[:0]])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run')
    parser.add_argument('--ckpt', required=True,
                        help='checkpoint holding the fitted presence branch')
    parser.add_argument('--config', help='config whose head has the branch')
    parser.add_argument('--detections',
                        help='uncalibrated detections (a gamma=0 test_results.pkl); '
                             'computed if not given')
    parser.add_argument('--work-root', default=WORK)
    args = parser.parse_args()

    if args.detections:
        with open(args.detections, 'rb') as f:
            results = pickle.load(f)
    else:
        results = uncalibrated(args.work_root, args.run, 'test')
    probs, _ = presence_probabilities(args.work_root, args.run, split='test',
                                      ckpt=args.ckpt, config=args.config)
    def measure(dets):
        m, curves = evaluate(ann('test'), dets)
        # A gate deletes detections outright, so the recall it can ever reach is
        # capped; calibration only rescales and cannot cap anything.
        m['maxRecall'] = float(curves['recall'].max())
        return m

    rows = {'base': measure(results)}
    for g in GAMMAS:
        rows[f'soft gamma={g}'] = measure(apply_gamma(results, probs, g))
    for t in GATES:
        rows[f'gate tau={t}'] = measure(apply_gate(results, probs, t))

    print(f"{'variant':16s}" + ''.join(f'{k:>13s}' for k in KEYS))
    for name, m in rows.items():
        print(f'{name:16s}' + ''.join(f'{m[k] * 100:13.2f}' for k in KEYS))
    json.dump({k: {m: v[m] for m in KEYS} for k, v in rows.items()},
              open(osp.join(args.work_root, f'gate_vs_calib_{args.run}.json'),
                   'w'), indent=1)


if __name__ == '__main__':
    main()
