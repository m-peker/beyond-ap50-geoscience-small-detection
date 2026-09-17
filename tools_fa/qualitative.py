"""Figure: what the false alarms on object-free tiles actually look like.

Picks the highest-scoring false alarms a run produces on negative test tiles
at a fixed recall operating point, and lays them out one clutter category per
row.  Each cell is a 96-pixel crop around the false alarm, so a 20-pixel
detection is visible in print, with its score in the corner.

With --compare BASE, only tiles on which BASE fires and the main run does not
are shown, which is the suppression the method claims.

Usage:
  python tools_fa/qualitative.py OUT.pdf RUN [--recall 0.9] [--compare RUN]
"""
import argparse
import glob
import json
import os.path as osp
import pickle

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from fa_eval import evaluate, load_coco  # noqa: E402
from paths import WORK, ann, img  # noqa: E402

ANN = ann('test')
IMG = img('test')
CATEGORIES = ['cloud', 'texture', 'clear_sea']
NICE = {'cloud': 'Cloud', 'texture': 'Texture', 'clear_sea': 'Clear sea'}
CROP = 96


def load_run(group):
    """Detections of the first seed of a run group, keyed by COCO image id."""
    pkls = sorted(glob.glob(osp.join(WORK, group + '_s*', 'test_results.pkl')))
    if not pkls:
        raise SystemExit('no test_results.pkl for ' + group)
    with open(pkls[0], 'rb') as f:
        results = pickle.load(f)
    return results, dict(zip(load_coco(ANN).getImgIds(), results))


def threshold_at(results, recall):
    _, curves = evaluate(ANN, results)
    idx = np.where(curves['recall'] >= recall)[0]
    if not len(idx):
        raise SystemExit('recall %.2f never reached' % recall)
    return float(curves['scores'][idx[0]])


def false_alarms(by_id, features, thr):
    """(score, image_id, box, category) for every detection on a negative tile.

    Every detection on an object-free tile is a false alarm by definition, so
    no matching against ground truth is needed here.
    """
    out = []
    for img_id, res in by_id.items():
        f = features[str(img_id)]
        if not f['negative']:
            continue
        for box in res[0]:
            if box[4] >= thr:
                out.append((float(box[4]), img_id, box[:4], f['category']))
    out.sort(key=lambda t: -t[0])
    return out


def draw(ax, feat, box, score, label):
    tile = cv2.cvtColor(cv2.imread(osp.join(IMG, feat['file_name'])),
                        cv2.COLOR_BGR2RGB)
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    h, w = tile.shape[:2]
    x0 = int(np.clip(cx - CROP / 2, 0, w - CROP))
    y0 = int(np.clip(cy - CROP / 2, 0, h - CROP))
    ax.imshow(tile[y0:y0 + CROP, x0:x0 + CROP])
    ax.add_patch(Rectangle((box[0] - x0, box[1] - y0), box[2] - box[0],
                           box[3] - box[1], fill=False, color='#eb3434', lw=1.0))
    ax.text(0.03, 0.97, label, transform=ax.transAxes, va='top', ha='left',
            fontsize=6, color='#ffffff',
            bbox=dict(fc='#000000', ec='none', alpha=0.55, pad=1.0))
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out')
    parser.add_argument('run')
    parser.add_argument('--recall', type=float, default=0.9)
    parser.add_argument('--compare', help='baseline run to show suppression of')
    parser.add_argument('--per-row', type=int, default=4)
    args = parser.parse_args()

    features = json.load(open(osp.join(WORK, 'scene_features.json')))
    results, by_id = load_run(args.run)
    thr = threshold_at(results, args.recall)
    fas = false_alarms(by_id, features, thr)

    if args.compare:
        base_results, base_by_id = load_run(args.compare)
        base_thr = threshold_at(base_results, args.recall)
        fas = false_alarms(base_by_id, features, base_thr)
        # keep only tiles the main run no longer fires on at its own threshold
        quiet = {i for i, r in by_id.items()
                 if not (len(r[0]) and (r[0][:, 4] >= thr).any())}
        fas = [f for f in fas if f[1] in quiet]

    rows = [c for c in CATEGORIES if any(f[3] == c for f in fas)]
    if not rows:
        raise SystemExit('no false alarms to show at recall %.2f' % args.recall)
    fig, axes = plt.subplots(
        len(rows), args.per_row,
        figsize=(1.78 * args.per_row, 1.78 * len(rows) + 0.2), squeeze=False)
    for r, cat in enumerate(rows):
        picks = [f for f in fas if f[3] == cat][:args.per_row]
        for c in range(args.per_row):
            ax = axes[r][c]
            if c < len(picks):
                score, img_id, box, _ = picks[c]
                draw(ax, features[str(img_id)], box, score, f'{score:.2f}')
            else:
                ax.axis('off')
        axes[r][0].set_ylabel(NICE.get(cat, cat), fontsize=8)
    fig.tight_layout(pad=0.3)
    fig.savefig(args.out)
    fig.savefig(osp.splitext(args.out)[0] + '.png', dpi=220)
    print('wrote', args.out, '| threshold %.4f at recall %.2f' % (thr, args.recall),
          '|', len(fas), 'false alarms')


if __name__ == '__main__':
    main()
