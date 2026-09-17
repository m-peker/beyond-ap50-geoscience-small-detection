"""How false alarms on object-free tiles relate to local ship-like structure.

The categorical stratification (cloud / texture / clear sea) says which kind of
scene fires, but the category rule ignores blob density -- the count of
ship-sized bright blobs -- which is the descriptor most directly related to
what a detector can confuse with a ship.  This reports the continuous
relationship instead of moving the category boundaries after seeing results.

For each run it computes, at a fixed recall operating point, the fraction of
object-free tiles that fire within each blob-density quantile bin, and the
rank agreement between blob density and firing (an AUC: the probability that a
firing tile has more blobs than a quiet one, 0.5 meaning no relationship).

Usage:
  python tools_fa/clutter_analysis.py RUN [RUN ...] [--recall 0.8] [--fig OUT.pdf]
"""
import argparse
import glob
import json
import os.path as osp
import pickle

import numpy as np
from scipy.stats import rankdata

from fa_eval import evaluate, load_coco
from paths import WORK, ann

ANN = ann('test')
NBINS = 5


def fired_flags(work_dir, features, recall):
    """(fires, blob_density) over object-free tiles at the recall operating point."""
    with open(osp.join(work_dir, 'test_results.pkl'), 'rb') as f:
        results = pickle.load(f)
    _, curves = evaluate(ANN, results)
    idx = np.where(curves['recall'] >= recall)[0]
    if not len(idx):
        return None
    thr = curves['scores'][idx[0]]
    fires, blobs = [], []
    for img_id, res in zip(load_coco(ANN).getImgIds(), results):
        f = features[str(img_id)]
        if not f['negative']:
            continue
        fires.append(bool(len(res[0]) and (res[0][:, 4] >= thr).any()))
        blobs.append(f['blob_density'])
    return np.array(fires), np.array(blobs)


def rank_auc(values, flags):
    """P(a firing tile ranks above a quiet one) by `values`, ties counted half."""
    pos, neg = values[flags], values[~flags]
    if not len(pos) or not len(neg):
        return float('nan')
    r = rankdata(np.concatenate([pos, neg]))
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (
        len(pos) * len(neg))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('runs', nargs='+', help='run group names, no _s<seed>')
    parser.add_argument('--recall', type=float, default=0.8)
    parser.add_argument('--work-root', default=WORK)
    parser.add_argument('--fig')
    args = parser.parse_args()

    features = json.load(open(osp.join(args.work_root, 'scene_features.json')))
    # Bin edges come from the tile population, not from any run's results, so
    # every run is summarised on the same bins.
    all_blobs = np.array([f['blob_density'] for f in features.values()
                          if f['negative']])
    edges = np.unique(np.quantile(all_blobs, np.linspace(0, 1, NBINS + 1)))

    out = {}
    for run in args.runs:
        dirs = sorted(glob.glob(osp.join(args.work_root, run + '_s*')))
        dirs = [d for d in dirs if osp.exists(osp.join(d, 'test_results.pkl'))]
        if not dirs:
            print(run, '- no results yet')
            continue
        rates, aucs = [], []
        for d in dirs:
            got = fired_flags(d, features, args.recall)
            if got is None:
                continue
            fires, blobs = got
            b = np.clip(np.digitize(blobs, edges[1:-1]), 0, len(edges) - 2)
            rates.append([fires[b == k].mean() if (b == k).any() else np.nan
                          for k in range(len(edges) - 1)])
            aucs.append(rank_auc(blobs, fires))
        rates = np.array(rates)
        out[run] = dict(seeds=len(rates), auc_mean=float(np.mean(aucs)),
                        auc_std=float(np.std(aucs)),
                        bin_edges=[float(e) for e in edges],
                        fa_rate_mean=np.nanmean(rates, 0).tolist(),
                        fa_rate_std=np.nanstd(rates, 0).tolist())
        bins = ' '.join(f'{100 * v:5.1f}' for v in out[run]['fa_rate_mean'])
        print(f"{run:26s} n={len(rates)}  AUC={out[run]['auc_mean']:.3f}"
              f"±{out[run]['auc_std']:.3f}  FA%% by blob-density bin: {bins}")

    dest = osp.join(args.work_root, f'clutter_continuous_R{args.recall}.json')
    json.dump(dict(recall=args.recall, runs=out), open(dest, 'w'), indent=1)
    print('written', dest)

    if args.fig and out:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        colors = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
        styles = ['-', '--', '-.', ':']
        plt.rcParams.update({'font.size': 8, 'font.family': 'serif',
                             'axes.edgecolor': '#52514e', 'axes.linewidth': 0.6})
        fig, ax = plt.subplots(figsize=(3.5, 2.2))
        centres = np.arange(len(edges) - 1)
        for k, (run, d) in enumerate(out.items()):
            m = 100 * np.array(d['fa_rate_mean'])
            s = 100 * np.array(d['fa_rate_std'])
            ax.plot(centres, m, styles[k % 4], color=colors[k % 4], lw=1.5,
                    marker='o', ms=3, label=run.replace('_', ' '))
            if d['seeds'] > 1:
                ax.fill_between(centres, m - s, m + s, color=colors[k % 4],
                                alpha=0.15, lw=0)
        ax.set_xticks(centres)
        ax.set_xticklabels([f'{edges[i]:.1f}-{edges[i + 1]:.1f}'
                            for i in range(len(edges) - 1)], fontsize=6)
        ax.set_xlabel(r'Ship-sized bright blobs per $10^4$ px')
        ax.set_ylabel('Object-free tiles with\na false alarm (%)')
        ax.legend(frameon=False, fontsize=6)
        ax.grid(True, axis='y', color='#e4e3df', lw=0.5)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        fig.tight_layout()
        fig.savefig(args.fig)
        fig.savefig(osp.splitext(args.fig)[0] + '.png', dpi=200)
        print('written', args.fig)


if __name__ == '__main__':
    main()
