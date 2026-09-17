"""Is the false-alarm reduction broad, or does one easy scene carry it?

The test split is drawn from 20 source acquisitions that do not appear in
training, and they differ in cloud cover and sea state.  A method that only
helps on one of them would show the same aggregate number as one that helps
everywhere.  This breaks the object-free tiles down by source scene at the
global operating point --- the threshold the whole test set reaches a given
recall at, which is what a deployment would use --- with the calibration off
and on.

Scenes contributing too few object-free tiles to estimate a rate are pooled
into one row rather than reported as noise.

Usage:
  python tools_fa/per_scene.py BASE_RUN CALIBRATED_RUN [--recall 0.9] [--min-neg 20]
"""
import argparse
import json
import os.path as osp
import pickle
import re
from collections import defaultdict

import numpy as np

from fa_eval import evaluate, load_coco
from paths import WORK, ann

ANN = ann('test')


def scene_of(file_name):
    """Source acquisition: the tile name without its row/column suffix."""
    return re.sub(r'_\d+_\d+\.png$', '', file_name)


def fired_by_scene(results, features, recall):
    _, curves = evaluate(ANN, results)
    idx = np.where(curves['recall'] >= recall)[0]
    if not len(idx):
        return None
    thr = curves['scores'][idx[0]]
    out = defaultdict(lambda: [0, 0])
    for img_id, res in zip(load_coco(ANN).getImgIds(), results):
        f = features[str(img_id)]
        if not f['negative']:
            continue
        hit = bool(len(res[0]) and (res[0][:, 4] >= thr).any())
        cell = out[scene_of(f['file_name'])]
        cell[0] += int(hit)
        cell[1] += 1
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('base')
    parser.add_argument('calibrated')
    parser.add_argument('--recall', type=float, default=0.9)
    parser.add_argument('--min-neg', type=int, default=20)
    parser.add_argument('--work-root', default=WORK)
    args = parser.parse_args()

    features = json.load(open(osp.join(args.work_root, 'scene_features.json')))
    counts = {}
    for label, run in (('base', args.base), ('calib', args.calibrated)):
        with open(osp.join(args.work_root, run, 'test_results.pkl'), 'rb') as f:
            counts[label] = fired_by_scene(pickle.load(f), features,
                                           args.recall)
        if counts[label] is None:
            raise SystemExit(f'{run} never reaches recall {args.recall}')

    big = [s for s, v in counts['base'].items() if v[1] >= args.min_neg]
    big.sort(key=lambda s: -counts['base'][s][1])
    print(f"false-alarm rate on object-free tiles at recall {args.recall}, "
          f"by source scene")
    print(f"{'scene':46s}{'neg':>5s}{'base':>9s}{'calib':>9s}{'delta':>9s}")
    rows = []
    for s in big:
        b, c = counts['base'][s], counts['calib'][s]
        rb, rc = 100 * b[0] / b[1], 100 * c[0] / c[1]
        rows.append((s, b[1], rb, rc))
        print(f'{s[:46]:46s}{b[1]:5d}{rb:8.1f}%{rc:8.1f}%{rc - rb:+8.1f}')
    small = [s for s in counts['base'] if s not in big]
    if small:
        b = [sum(counts['base'][s][i] for s in small) for i in (0, 1)]
        c = [sum(counts['calib'][s][i] for s in small) for i in (0, 1)]
        rb, rc = 100 * b[0] / b[1], 100 * c[0] / c[1]
        print(f'{"(%d scenes with <%d negatives, pooled)" % (len(small), args.min_neg):46s}'
              f'{b[1]:5d}{rb:8.1f}%{rc:8.1f}%{rc - rb:+8.1f}')
    improved = sum(1 for _, _, rb, rc in rows if rc < rb)
    print(f'\nimproved in {improved} of {len(rows)} scenes with '
          f'>={args.min_neg} object-free tiles')
    json.dump([dict(scene=s, negatives=n, base=rb, calibrated=rc)
               for s, n, rb, rc in rows],
              open(osp.join(args.work_root, 'per_scene.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
