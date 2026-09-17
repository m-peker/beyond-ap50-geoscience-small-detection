"""Figure: false-alarm image rate on object-free tiles versus recall.

Each group is the mean over its seeds on a common recall grid; the band spans
min-max over seeds. Series are identified by direct labels and line style as
well as color, so the figure survives grayscale printing.

Usage: python tools_fa/plot_curves.py OUT.pdf GROUP[=Label] [GROUP[=Label] ...]
  GROUP is a run name without the _s<seed> suffix, e.g. fcos_p2_all.
"""
import glob
import os.path as osp
import pickle
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from fa_eval import evaluate  # noqa: E402
from paths import WORK, ann  # noqa: E402

ANN = ann('test')
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
STYLES = ['-', '--', '-.', ':']
GRID = np.linspace(0.5, 0.95, 91)


def curve_on_grid(pkl):
    with open(pkl, 'rb') as f:
        _, c = evaluate(ANN, pickle.load(f))
    out = np.full(len(GRID), np.nan)
    for i, r in enumerate(GRID):
        idx = np.where(c['recall'] >= r)[0]
        if len(idx):
            out[i] = c['fa_img'][idx[0]]
    return out


def main():
    out_file, specs = sys.argv[1], sys.argv[2:]
    plt.rcParams.update({'font.size': 8, 'font.family': 'serif',
                         'axes.edgecolor': '#52514e', 'axes.linewidth': 0.6})
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    for k, spec in enumerate(specs):
        group, _, label = spec.partition('=')
        pkls = sorted(glob.glob(osp.join(WORK, group + '_s*', 'test_results.pkl')))
        curves = np.stack([curve_on_grid(p) for p in pkls]) * 100
        mean = np.nanmean(curves, 0)
        color, style = COLORS[k % 4], STYLES[k % 4]
        ax.plot(GRID * 100, mean, style, color=color, lw=1.5,
                label=f'{label or group} ($n$={len(pkls)})')
        if len(pkls) > 1:
            # min-max over seeds: on this benchmark the band is the finding, not
            # an error bar, so it is drawn solidly enough to read
            with np.errstate(all='ignore'):
                lo, hi = np.nanmin(curves, 0), np.nanmax(curves, 0)
            ax.fill_between(GRID * 100, lo, hi, color=color, alpha=0.22, lw=0)
    ax.legend(frameon=False, fontsize=7, loc='upper left',
              handlelength=2.4, borderaxespad=0.2)
    ax.set_xlabel('Recall on all ships (%)')
    ax.set_ylabel('Object-free tiles with a\nfalse alarm (%)')
    ax.grid(True, color='#e4e3df', lw=0.5)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.set_xlim(50, 95)
    ax.set_ylim(0, None)
    fig.tight_layout()
    fig.savefig(out_file)
    fig.savefig(osp.splitext(out_file)[0] + '.png', dpi=200)


if __name__ == '__main__':
    main()
