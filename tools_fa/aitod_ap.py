"""Per-class AP on AI-TOD-v2, under AI-TOD's own evaluation protocol.

AI-TOD scores at maxDets=1500 with its own area ranges, and its summarize()
returns [AP, unused, AP50, AP75, AP_verytiny, AP_tiny, ...] rather than COCO's
layout.  Evaluating it with COCO defaults understates AP50 by about four points.

Detections are passed as an (N, 7) array: the dict form costs tens of gigabytes
at this scale.

Usage:
  python tools_fa/aitod_ap.py RESULTS_PKL [RESULTS_PKL ...] [--out FILE]
"""
import argparse
import contextlib
import io
import json
import os.path as osp
import pickle

import numpy as np

# aitodpycocotools predates numpy 1.24, which removed these aliases.
for _name, _builtin in (('float', float), ('int', int), ('bool', bool)):
    if not hasattr(np, _name):
        setattr(np, _name, _builtin)

from aitodpycocotools.coco import COCO  # noqa: E402
from aitodpycocotools.cocoeval import COCOeval  # noqa: E402

ANN = ('/home/arge/tod/aitod_build/work/aitod/annotations/aitodv2_test.json')
# index into COCOeval.stats for AI-TOD's summary rows
SLOTS = {'AP': 0, 'AP50': 2, 'AP75': 3, 'AP_verytiny': 4, 'AP_tiny': 5}


def detections_array(results, img_ids, cat_ids, only_class=None):
    blocks = []
    for img_id, per_class in zip(img_ids, results):
        for k, arr in enumerate(per_class):
            if not len(arr) or (only_class is not None and k != only_class):
                continue
            b = np.empty((len(arr), 7))
            b[:, 0] = img_id
            b[:, 1:3] = arr[:, :2]
            b[:, 3] = arr[:, 2] - arr[:, 0]
            b[:, 4] = arr[:, 3] - arr[:, 1]
            b[:, 5] = arr[:, 4]
            b[:, 6] = cat_ids[k]
            blocks.append(b)
    return np.concatenate(blocks) if blocks else np.zeros((0, 7))


def summarise(coco, dets, cat_ids=None):
    if not len(dets):
        return {k: 0.0 for k in SLOTS}
    with contextlib.redirect_stdout(io.StringIO()):
        ev = COCOeval(coco, coco.loadRes(dets), 'bbox')
        if cat_ids is not None:
            ev.params.catIds = list(cat_ids)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return {k: float(ev.stats[i]) for k, i in SLOTS.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results', nargs='+')
    parser.add_argument('--ann', default=ANN)
    parser.add_argument('--out')
    args = parser.parse_args()

    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(args.ann)
    img_ids, cat_ids = coco.getImgIds(), coco.getCatIds()
    names = [c['name'] for c in coco.loadCats(cat_ids)]

    out = {}
    for path in args.results:
        with open(path, 'rb') as f:
            results = pickle.load(f)
        dets = detections_array(results, img_ids, cat_ids)
        row = {'overall': summarise(coco, dets)}
        for k, name in enumerate(names):
            row[name] = summarise(coco, dets, cat_ids=[cat_ids[k]])
        out[osp.basename(osp.dirname(path))] = row
        print(f"{osp.basename(osp.dirname(path)):34s} "
              f"AP {row['overall']['AP'] * 100:6.2f}  "
              f"AP50 {row['overall']['AP50'] * 100:6.2f}", flush=True)
        for name in names:
            print(f"    {name:16s} AP50 {row[name]['AP50'] * 100:6.2f}",
                  flush=True)
    if args.out:
        json.dump(out, open(args.out, 'w'), indent=1)
        print('wrote', args.out)


if __name__ == '__main__':
    main()
