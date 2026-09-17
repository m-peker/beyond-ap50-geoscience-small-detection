"""Check the numeric claims in the paper against the measurements.

Every number the text states about our experiments should be derivable from the
metrics files; this lists them beside what the data says so a mismatch is
visible rather than trusted.
"""
import json
import os.path as osp

import numpy as np

W = '/home/arge/tod/work'


def stat(run, key, seeds=(0, 1, 2)):
    v = [json.load(open(f'{W}/{run}_s{s}/metrics.json'))[key] * 100
         for s in seeds]
    return np.mean(v), np.std(v)


def paired(a, b, key, seeds=(0, 1, 2)):
    d = [(json.load(open(f'{W}/{b}_s{s}/metrics.json'))[key]
          - json.load(open(f'{W}/{a}_s{s}/metrics.json'))[key]) * 100
         for s in seeds]
    return np.mean(d), np.std(d)


claims = [
    ('abstract: 20-26 pt FAimg drop, FCOS',
     paired('fa_fcos_p2_all_Ppost_g0', 'fa_fcos_p2_all_Ppost', 'FAimg@R0.9')),
    ('abstract: same, NWD',
     paired('nwd_p2_all_Ppost_g0', 'nwd_p2_all_Ppost', 'FAimg@R0.9')),
    ('abstract: same, RFLA',
     paired('rfla_p2_all_Ppost_g0', 'rfla_p2_all_Ppost', 'FAimg@R0.9')),
    ('abstract: AP50 cost FCOS',
     paired('fa_fcos_p2_all_Ppost_g0', 'fa_fcos_p2_all_Ppost', 'AP50')),
    ('abstract: spread 3.8 -> 0.4 (baseline std)',
     stat('fcos_p2_all', 'FAimg@R0.9')),
    ('abstract: spread (calibrated std)',
     stat('fa_fcos_p2_all_Ppost', 'FAimg@R0.9')),
    ('V-B: positive-only AP50 mean/std', stat('fcos_p2_pos', 'AP50')),
    ('V-C: negatives, FCOS AP50 gain',
     paired('fcos_p2_pos', 'fcos_p2_all', 'AP50')),
    ('V-C: negatives, RFLA AP50 gain',
     paired('rfla_p2_pos', 'rfla_p2_all', 'AP50')),
    ('V-C: negatives, NWD AP50 gain',
     paired('nwd_p2_pos', 'nwd_p2_all', 'AP50')),
    ('V-E: +P AP50 cost', paired('fcos_p2_all', 'fa_fcos_p2_all_P', 'AP50')),
    ('V-E: +PN AP50 cost', paired('fcos_p2_all', 'fa_fcos_p2_all_PN', 'AP50')),
    ('V-E: +Pdet AP50 cost',
     paired('fcos_p2_all', 'fa_fcos_p2_all_Pdet', 'AP50')),
    ('V-E: normalizer AP50', paired('fcos_p2_all', 'fa_fcos_p2_all_N', 'AP50')),
    ('V-E: normalizer FAimg@R0.9',
     paired('fcos_p2_all', 'fa_fcos_p2_all_N', 'FAimg@R0.9')),
    ('V-E: +P FAimg@R0.9 gain',
     paired('fa_fcos_p2_all_P_g0', 'fa_fcos_p2_all_P', 'FAimg@R0.9')),
]
for label, (m, s) in claims:
    print(f'{label:46s} {m:+8.2f} ± {s:5.2f}')

print()
for s in (0, 1, 2):
    m = json.load(open(f'{W}/fcos_p2_pos_s{s}/metrics.json'))
    print(f'  positive-only seed {s}: AP50 {m["AP50"] * 100:.2f}  '
          f'FAimg@R0.9 {m["FAimg@R0.9"] * 100:.2f}')
