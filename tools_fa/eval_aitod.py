"""Per-class false-alarm evaluation of a detector on AI-TOD-v2.

AI-TOD-v2 has 18 images with no object at all and thousands with no instance of
a given class, so the protocol applies per class: for class c the negatives are
the images holding no c.  Operating points are lower than on LEVIR-Ship because
the benchmark is harder -- a detector that never reaches 0.9 recall for a class
has no 0.9 operating point, and the metric is undefined rather than bad.

Usage:
  python tools_fa/eval_aitod.py BASE_PKL CALIB_PKL [--recalls 0.5 0.7]
"""
import argparse
import json
import os.path as osp
import pickle

from fa_eval import evaluate
from paths import WORK, aitod_ann

CLASSES = ('airplane', 'bridge', 'storage-tank', 'ship', 'swimming-pool',
           'vehicle', 'person', 'wind-mill')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('base')
    parser.add_argument('calibrated')
    parser.add_argument('--ann', default=aitod_ann('test'))
    parser.add_argument('--recalls', type=float, nargs='+', default=[0.5, 0.7])
    parser.add_argument('--out', default=osp.join(WORK, 'aitodv2_per_class.json'))
    args = parser.parse_args()

    runs = {}
    for name, path in (('base', args.base), ('calibrated', args.calibrated)):
        with open(path, 'rb') as f:
            runs[name] = pickle.load(f)

    out = {}
    r0, r1 = args.recalls[0], args.recalls[-1]
    keys = ['AP50', 'LAMR', 'maxRecall', f'FAimg@R{r0}', f'FAimg@R{r1}',
            f'FPPIneg@R{r0}', f'FPPIneg@R{r1}']
    print(f"{'class':14s}{'':6s}" + ''.join(f'{k[:12]:>13s}' for k in keys))
    for cls in CLASSES:
        out[cls] = {}
        for name in ('base', 'calibrated'):
            m, _ = evaluate(args.ann, runs[name], category=cls,
                            recalls=tuple(args.recalls))
            out[cls][name] = {k: m[k] for k in
                              keys + ['num_neg_images', 'num_pos_images']}
        b, c = out[cls]['base'], out[cls]['calibrated']
        print(f"{cls:14s}{'base':>6s}" + ''.join(f'{b[k] * 100:13.2f}' for k in keys))
        print(f"{'':14s}{'calib':>6s}" + ''.join(f'{c[k] * 100:13.2f}' for k in keys)
              + f"   (neg {b['num_neg_images']}, pos {b['num_pos_images']})",
              flush=True)
    json.dump(out, open(args.out, 'w'), indent=1)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
