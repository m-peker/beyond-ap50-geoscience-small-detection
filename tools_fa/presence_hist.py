"""Figure: the image-level presence probability on positive vs object-free tiles.

The scene-evidence branch is only useful if the probability it assigns
separates tiles that contain a ship from tiles that do not.  This runs a
trained model over the test split, records sigma(z-hat) per tile, and plots
the two distributions together with the separation they achieve.

The probability is read from the head's own pooling, not re-derived, so the
figure shows exactly the quantity that calibrates the box scores.

Usage:
  python tools_fa/presence_hist.py OUT.pdf RUN [--work-root ...]
"""
import argparse
import glob
import json
import os.path as osp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import rankdata  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from mmcv import Config  # noqa: E402
from mmcv.runner import load_checkpoint  # noqa: E402

from mmdet.datasets import build_dataloader, build_dataset  # noqa: E402
from mmdet.models import build_detector  # noqa: E402
from paths import WORK, ann, img  # noqa: E402

POS_COLOR, NEG_COLOR = '#2a78d6', '#eb6834'


def run_dir(work_root, run):
    hits = sorted(glob.glob(osp.join(work_root, run + '_s*')))
    if not hits:
        raise SystemExit('no run directory matching ' + run)
    return hits[0]


def presence_probabilities(work_root, run, device='cuda', limit=None,
                           split='test', ckpt=None, config=None,
                           data_from_config=False):
    wd = run_dir(work_root, run)
    if config is None:
        cfg_files = [f for f in sorted(glob.glob(osp.join(wd, '*.py')))
                     if not osp.basename(f).startswith('posthoc_cfg')]
        if not cfg_files:
            raise SystemExit('no dumped config in ' + wd +
                             '; pass one explicitly')
        config = cfg_files[0]
    cfg = Config.fromfile(config)
    if not data_from_config:
        cfg.data.test.ann_file = ann(split)
        cfg.data.test.img_prefix = img(split)
    # backbones differ in which init key they accept; clear only what is there
    for key in ('init_cfg', 'pretrained'):
        if key in cfg.model.backbone:
            cfg.model.backbone[key] = None

    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=2,
                              dist=False, shuffle=False)
    model = build_detector(cfg.model)
    load_checkpoint(model, ckpt or osp.join(wd, 'latest.pth'),
                    map_location='cpu')
    head = model.bbox_head
    if getattr(head, 'presence', None) is None:
        raise SystemExit(run + ' has no presence branch')
    model.to(device).eval()

    probs = []
    with torch.no_grad():
        for i, data in enumerate(loader):
            if limit is not None and i >= limit:
                break
            feats = model.extract_feat(data['img'][0].to(device))
            *_, presence_maps = head(feats)
            logit = head.image_presence_logits(presence_maps)
            probs.append(torch.sigmoid(logit)[0].cpu().numpy())
    # (num_images, num_classes); a single-class benchmark gives one column
    return np.stack(probs), dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out')
    parser.add_argument('run')
    parser.add_argument('--work-root', default=WORK)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--limit', type=int,
                        help='only the first N tiles (smoke test)')
    parser.add_argument('--config', help='config whose head has the branch; '
                                         'needed for a post-hoc run, whose '
                                         'directory holds results not weights')
    parser.add_argument('--ckpt', help='checkpoint holding the fitted branch')
    parser.add_argument('--fit-run', help='run directory the branch was fitted '
                                          'on, if different from RUN')
    args = parser.parse_args()

    probs, dataset = presence_probabilities(
        args.work_root, args.fit_run or args.run, args.device, args.limit,
        ckpt=args.ckpt, config=args.config)
    if probs.ndim == 2 and probs.shape[1] == 1:
        probs = probs[:, 0]          # the figure is for a single-class run
    elif probs.ndim == 2:
        raise SystemExit('this figure is for single-class runs; '
                         'a multi-class model gives one column per class')
    features = json.load(open(osp.join(args.work_root, 'scene_features.json')))
    negative = np.array([features[str(i)]['negative']
                         for i in dataset.img_ids[:len(probs)]])

    pos, neg = probs[~negative], probs[negative]
    # Threshold-free separation: probability that a random positive tile scores
    # above a random negative one, counting a tie as half (the ROC area).
    # Average ranks are required because a saturated presence branch produces
    # exact ties, which ordinal ranks would score as losses for the positives.
    ranks = rankdata(np.concatenate([pos, neg]))
    auc = (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (
        len(pos) * len(neg))

    plt.rcParams.update({'font.size': 8, 'font.family': 'serif',
                         'axes.edgecolor': '#52514e', 'axes.linewidth': 0.6})
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    bins = np.linspace(0, 1, 41)
    ax.hist(pos, bins=bins, color=POS_COLOR, alpha=0.75,
            label=f'contains a ship (n={len(pos)})')
    ax.hist(neg, bins=bins, color=NEG_COLOR, alpha=0.75,
            label=f'object-free (n={len(neg)})')
    ax.set_xlabel(r'Scene presence probability $\sigma(\hat{z})$')
    ax.set_ylabel('Test tiles')
    ax.set_title(f'AUC = {auc:.3f}', fontsize=8, loc='left')
    ax.legend(frameon=False, fontsize=7)
    ax.grid(True, axis='y', color='#e4e3df', lw=0.5)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out)
    fig.savefig(osp.splitext(args.out)[0] + '.png', dpi=200)
    print(f'{args.run}: AUC={auc:.4f} '
          f'median positive={np.median(pos):.3f} negative={np.median(neg):.3f}')
    # Calibration multiplies every score in a tile by the same factor, so it can
    # only suppress false alarms where the branch judges the tile empty.  The
    # object-free tiles it confidently gets wrong are a floor on FAimg.
    floors = {f'neg_above_{t}': int((neg > t).sum()) for t in (0.5, 0.9, 0.99)}
    print('  object-free tiles the branch calls occupied: '
          + ', '.join(f'p>{t}: {floors[f"neg_above_{t}"]}'
                      f' ({100 * floors[f"neg_above_{t}"] / len(neg):.1f}%)'
                      for t in (0.5, 0.9, 0.99)))
    json.dump(dict(auc=float(auc),
                   median_positive=float(np.median(pos)),
                   median_negative=float(np.median(neg)),
                   num_positive=int(len(pos)), num_negative=int(len(neg)),
                   probabilities=[float(v) for v in probs], **floors),
              open(osp.join(args.work_root, f'presence_{args.run}.json'), 'w'),
              indent=1)


if __name__ == '__main__':
    main()
