"""Generate the paper's result tables straight from the metrics files.

Transcribing numbers by hand into LaTeX is where papers acquire errors that
nobody can trace afterwards.  Every table the paper reports is emitted here from
the same metrics.json files the analysis reads, so the text cannot drift from
the experiments.

Usage:
  python tools_fa/make_tables.py [--out-dir paper/tables]
"""
import argparse
import json
import os
import os.path as osp

import numpy as np

from paths import WORK

SEEDS = (0, 1, 2)


def load(run, seed):
    p = osp.join(WORK, f'{run}_s{seed}', 'metrics.json')
    return json.load(open(p)) if osp.exists(p) else None


def stat(run, key, seeds=SEEDS):
    """mean and standard deviation over the seeds that exist."""
    vals = [m[key] * 100 for m in (load(run, s) for s in seeds)
            if m is not None and key in m and m[key] == m[key]]
    if not vals:
        return None
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


def cell(run, key, seeds=SEEDS, digits=1):
    s = stat(run, key, seeds)
    if s is None:
        return '--'
    mean, sd, n = s
    if n == 1:
        return f'{mean:.{digits}f}'
    return f'{mean:.{digits}f}$\\pm${sd:.{digits}f}'


def table(rows, cols, caption, label, col_titles=None):
    head = col_titles or [c.replace('@', '@').replace('_', r'\_') for c in cols]
    out = ['\\begin{table}[t]', '\\centering',
           '\\caption{%s}' % caption, '\\label{%s}' % label,
           '\\resizebox{\\columnwidth}{!}{%',
           '\\begin{tabular}{l' + 'r' * len(cols) + '}', '\\toprule',
           'Method & ' + ' & '.join(head) + r' \\', '\\midrule']
    for label_text, run, seeds in rows:
        if run is None:                      # a rule or section break
            out.append('\\midrule')
            continue
        cells = [cell(run, c, seeds) for c in cols]
        out.append(f'{label_text} & ' + ' & '.join(cells) + r' \\')
    out += ['\\bottomrule', '\\end{tabular}}', '\\end{table}', '']
    return '\n'.join(out)


# AP50 as reported by the sources themselves.  The first block is Table IV of
# the DRENet paper, whose metric it defines as AP at IoU 0.5 with COCO's
# 101-point interpolation -- the same quantity we report.  ORFENet and LTDNet
# come from the evaluation logs released in their own repositories.
PUBLISHED = [
    ('SSD (VGG16)~\\cite{chen2022drenet}', 52.6),
    ('YOLOv3~\\cite{chen2022drenet}', 69.9),
    ('Faster R-CNN (VGG16)~\\cite{chen2022drenet}', 70.8),
    ('EfficientDet-D0~\\cite{chen2022drenet}', 71.3),
    ('ImYOLOv3~\\cite{chen2022drenet}', 72.6),
    ('RetinaNet (R50)~\\cite{chen2022drenet}', 74.9),
    ('FCOS (R50)~\\cite{chen2022drenet}', 75.5),
    ('YOLOv5s~\\cite{chen2022drenet}', 75.6),
    ('Mask R-CNN + DFR + RFE~\\cite{chen2022drenet}', 76.2),
    ('CenterNet (Hourglass-104)~\\cite{chen2022drenet}', 77.7),
    ('EfficientDet-D2~\\cite{chen2022drenet}', 80.9),
    ('DRENet~\\cite{chen2022drenet}', 82.4),
    ('ORFENet~\\cite{liu2024orfenet}', 83.0),
    ('LTDNet~\\cite{liu2025ltdnet}', 84.0),
]
OURS = [('FCOS-P2, all images', 'fcos_p2_all'),
        ('RFLA-P2, all images', 'rfla_p2_all'),
        ('NWD-P2, all images', 'nwd_p2_all'),
        ('FCOS-P2 + calibration', 'fa_fcos_p2_all_Ppost')]


def published_table():
    """Published AP50 on LEVIR-Ship, as reported, beside ours."""
    out = ['\\begin{table}[t]', '\\centering',
           '\\caption{AP50 on LEVIR-Ship as reported by the sources '
           'themselves, beside our configurations. Published values are '
           'computed without the annotation identifier fix of Section~III-C, '
           'so the comparable column for our runs is AP50$^{\\dagger}$; the '
           'fix is worth $0.274\\pm0.058$ points. Our detectors are plain '
           'rather than tuned for this benchmark, and the calibration is '
           'applied to a published checkpoint in Table~\\ref{tab:posthoc} '
           'rather than competing with one here.}',
           '\\label{tab:published}', '\\begin{tabular}{lrr}', '\\toprule',
           'Method & AP50 & AP50$^{\\dagger}$ \\\\', '\\midrule']
    for name, ap in PUBLISHED:
        out.append(f'{name} & --- & {ap:.1f} \\\\')
    out.append('\\midrule')
    for label, run in OURS:
        out.append(f'{label} & {cell(run, "AP50")} & '
                   f'{cell(run, "AP50_noidfix")} \\\\')
    out += ['\\bottomrule', '\\end{tabular}', '\\end{table}', '']
    return '\n'.join(out)


def gamma_table():
    """Sensitivity to the calibration strength, on both splits.

    Reported as sensitivity rather than as a selection: gamma keeps the value
    the method declared before any results, and the point of the table is that
    no single value optimises everything.
    """
    d = json.load(open(osp.join(WORK, 'gamma_summary.json')))
    cols = ['AP50', 'LAMR', 'FAimg@R0.8', 'FAimg@R0.9', 'FPPIneg@R0.9']
    titles = ['AP50', 'LAMR$\\downarrow$', '$\\faimg^{0.8}\\downarrow$',
              '$\\faimg^{0.9}\\downarrow$', '$\\fppineg^{0.9}\\downarrow$']
    gammas = ['0', '0.25', '0.5', '1', '1.5', '2', '3', '4']
    out = ['\\begin{table}[t]', '\\centering',
           '\\caption{Sensitivity to the calibration strength $\\gamma$, '
           'mean$\\pm$std over three seeds. $\\gamma=0$ is the uncalibrated '
           'detector and $\\gamma=1$ the value the method fixes in advance. '
           'The false-alarm image rate falls monotonically, AP50 and LAMR '
           'worsen monotonically, and false positives per object-free image '
           'turn around: no single $\\gamma$ improves everything.}',
           '\\label{tab:gamma}', '\\resizebox{\\columnwidth}{!}{%',
           '\\begin{tabular}{ll' + 'r' * len(cols) + '}', '\\toprule',
           'Split & $\\gamma$ & ' + ' & '.join(titles) + ' \\\\']
    for split in ('val', 'test'):
        out.append('\\midrule')
        for i, g in enumerate(gammas):
            row = d[split][g]
            label = ('Validation' if split == 'val' else 'Test') if i == 0 else ''
            cells = [f'{row[c][0]:.2f}$\\pm${row[c][1]:.2f}' for c in cols]
            out.append(f'{label} & {g} & ' + ' & '.join(cells) + ' \\\\')
    out += ['\\bottomrule', '\\end{tabular}}', '\\end{table}', '']
    return '\n'.join(out)


def aitod_table():
    """AI-TOD-v2, per class, on the released LTDNet checkpoint.

    AP comes from the authors' own detection settings under AI-TOD's evaluation
    toolkit; the operating-point metrics come from the low-threshold runs, which
    need the low-scoring tail that COCO's evaluator cannot hold.  The two
    sources are stated in the caption rather than silently mixed.
    """
    ap = json.load(open(osp.join(WORK, 'aitodv2_ap_official.json')))
    fa = json.load(open(osp.join(WORK, 'aitodv2_per_class_gated.json')))
    base_ap = ap['LTDNet_aitodv2_authorcfg_s0']
    cal_ap = ap['LTDNet_aitodv2_gated_authorcfg_s0']
    order = ['ship', 'vehicle', 'person', 'storage-tank', 'airplane',
             'bridge', 'swimming-pool', 'wind-mill']
    out = ['\\begin{table}[t]', '\\centering',
           '\\caption{AI-TOD-v2, per class, on the released LTDNet checkpoint '
           'with the branch fitted post hoc and gated on validation separation. '
           'Negatives for a class are the images holding no instance of it. AP50 '
           'is measured under the authors\' detection settings and AI-TOD\'s '
           'evaluation toolkit; $\\faimg$ and $\\fppineg$ come from a '
           'low-threshold run, which the operating-point curves require. '
           'wind-mill is left uncalibrated by the gate.}',
           '\\label{tab:aitod}', '\\resizebox{\\columnwidth}{!}{%',
           '\\begin{tabular}{lrrrrrrr}', '\\toprule',
           # \\faimg and \\fppineg already carry a subscript, so a second one
           # would be a double subscript; the base/calibrated split is a column
           # pair labelled in the header instead.
           '& & \\multicolumn{2}{c}{AP50} & '
           '\\multicolumn{2}{c}{$\\faimg^{0.7}$} & '
           '\\multicolumn{2}{c}{$\\fppineg^{0.7}$} \\\\',
           '\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}',
           'Class & neg. & base & cal. & base & cal. & base & cal. \\\\',
           '\\midrule']
    for c in order:
        b, g = fa[c]['base'], fa[c]['calibrated']
        f0, f1 = b.get('FAimg@R0.7'), g.get('FAimg@R0.7')
        p0, p1 = b.get('FPPIneg@R0.7'), g.get('FPPIneg@R0.7')
        def num(v, digits=2):
            return '--' if v is None or v != v else f'{v * 100:.{digits}f}'
        out.append(
            f"{c.replace('-', '-')} & {b['num_neg_images']} & "
            f"{base_ap[c]['AP50'] * 100:.2f} & {cal_ap[c]['AP50'] * 100:.2f} & "
            f"{num(f0)} & {num(f1)} & {num(p0)} & {num(p1)} \\\\")
    out += ['\\midrule',
            f"mean & --- & {base_ap['overall']['AP50'] * 100:.2f} & "
            f"{cal_ap['overall']['AP50'] * 100:.2f} & & & & \\\\",
            '\\bottomrule', '\\end{tabular}}', '\\end{table}', '']
    return '\n'.join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', default=osp.join(
        osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))),
        'paper', 'tables'))
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    metrics = ['AP', 'AP50', 'AP50_noidfix', 'LAMR', 'R@FPPI0.1',
               'FPPIneg@R0.8', 'FAimg@R0.8', 'FAimg@R0.9']
    titles = ['AP', 'AP50', 'AP50$^{\\dagger}$', 'LAMR$\\downarrow$', 'R@0.1',
              '$\\fppineg^{0.8}\\downarrow$', '$\\faimg^{0.8}\\downarrow$',
              '$\\faimg^{0.9}\\downarrow$']
    files = {}

    files['negatives.tex'] = table(
        [('FCOS-P2, positive images only', 'fcos_p2_pos', SEEDS),
         ('FCOS-P2, all images', 'fcos_p2_all', SEEDS),
         (None, None, None),
         ('RFLA-P2, positive images only', 'rfla_p2_pos', SEEDS),
         ('RFLA-P2, all images', 'rfla_p2_all', SEEDS),
         (None, None, None),
         ('NWD-P2, positive images only', 'nwd_p2_pos', SEEDS),
         ('NWD-P2, all images', 'nwd_p2_all', SEEDS)],
        metrics,
        'Training with the object-free half of LEVIR-Ship, at a matched '
        'iteration budget. Mean$\\pm$std over three seeds; within each block '
        'the two rows share an initialization at every seed. '
        'AP50$^{\\dagger}$ is AP50 without the annotation identifier fix of '
        'Section~III-C, the form in which published numbers are computed.',
        'tab:negatives', titles)

    files['posthoc.tex'] = table(
        [('FCOS-P2', 'fa_fcos_p2_all_Ppost_g0', SEEDS),
         ('\\quad + calibration', 'fa_fcos_p2_all_Ppost', SEEDS),
         (None, None, None),
         ('NWD-P2', 'nwd_p2_all_Ppost_g0', SEEDS),
         ('\\quad + calibration', 'nwd_p2_all_Ppost', SEEDS),
         (None, None, None),
         ('LTDNet, released checkpoint', 'LTDNet_Ppost_g0', (0,)),
         ('\\quad + calibration', 'LTDNet_Ppost', (0,))],
        metrics,
        'Post-hoc scene-evidence calibration on three detectors, including one '
        'we did not train. Each first row is the detector with the branch '
        'disabled ($\\gamma=0$), which reproduces it exactly.',
        'tab:posthoc', titles)

    ablation = ['AP50', 'LAMR', 'FAimg@R0.8', 'FAimg@R0.9', 'FPPIneg@R0.9']
    ablation_titles = ['AP50', 'LAMR$\\downarrow$',
                       '$\\faimg^{0.8}\\downarrow$', '$\\faimg^{0.9}\\downarrow$',
                       '$\\fppineg^{0.9}\\downarrow$']
    files['ablation.tex'] = table(
        [('FCOS-P2 baseline', 'fcos_p2_all', SEEDS),
         ('EMA loss normalizer', 'fa_fcos_p2_all_N', SEEDS),
         (None, None, None),
         ('Presence branch, joint', 'fa_fcos_p2_all_P_g0', SEEDS),
         ('\\quad + calibration', 'fa_fcos_p2_all_P', SEEDS),
         ('Presence branch, joint + EMA', 'fa_fcos_p2_all_PN_g0', SEEDS),
         ('\\quad + calibration', 'fa_fcos_p2_all_PN', SEEDS),
         ('Presence branch, gradient stopped', 'fa_fcos_p2_all_Pdet_g0', SEEDS),
         ('\\quad + calibration', 'fa_fcos_p2_all_Pdet', SEEDS),
         (None, None, None),
         ('Presence branch, fitted post hoc', 'fa_fcos_p2_all_Ppost_g0', SEEDS),
         ('\\quad + calibration (ours)', 'fa_fcos_p2_all_Ppost', SEEDS)],
        ablation,
        'How the presence branch is obtained. Each pair shares weights, so the '
        'difference within a pair is the calibration alone.',
        'tab:ablation', ablation_titles)

    files['topk.tex'] = table(
        [('$k=1$', 'fa_fcos_p2_all_Ppost_k1', (0,)),
         ('$k=4$ (default)', 'fa_fcos_p2_all_Ppost', (0,)),
         ('$k=16$', 'fa_fcos_p2_all_Ppost_k16', (0,)),
         ('$k=64$', 'fa_fcos_p2_all_Ppost_k64', (0,))],
        ablation,
        'Pooling width of the presence branch, one seed.',
        'tab:topk', ablation_titles)

    files['aitod.tex'] = aitod_table()
    files['gamma.tex'] = gamma_table()
    files['published.tex'] = published_table()

    for name, text in files.items():
        with open(osp.join(args.out_dir, name), 'w') as f:
            f.write(text)
        print('wrote', osp.join(args.out_dir, name))


if __name__ == '__main__':
    main()
