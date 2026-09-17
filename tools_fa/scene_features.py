"""Rule-based clutter descriptors for every test tile (reproducible, no labels).

cloud_frac   fraction of bright, low-saturation pixels (gray > 170, S < 0.25)
blob_density ship-sized bright blobs per 10^4 px: connected components of
             (gray - Gaussian background) > 12 with area in [4, 400] px
edge_density fraction of Canny edge pixels (waves, sun glint, small clouds)
category     cloud (cloud_frac >= 0.05) > texture: waves, glint, scattered small clouds
             (edge_density >= 0.02)
             > clear sea
"""
import json
import os.path as osp
import sys

import cv2
import numpy as np

from paths import WORK, ann, img

ANN = ann('test')
IMG = img('test')


def describe(path):
    bgr = cv2.imread(path)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[..., 1].astype(np.float32) / 255
    cloud = (gray > 170) & (sat < 0.25)

    background = cv2.GaussianBlur(gray, (0, 0), 8)
    residual = ((gray - background) > 12).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(residual, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    blobs = int(((areas >= 4) & (areas <= 400)).sum())

    edges = cv2.Canny(cv2.GaussianBlur(bgr, (3, 3), 0), 40, 120) > 0
    f = dict(cloud_frac=float(cloud.mean()),
             blob_density=blobs / (gray.size / 1e4),
             edge_density=float(edges.mean()),
             mean_gray=float(gray.mean()))
    if f['cloud_frac'] >= 0.05:
        f['category'] = 'cloud'
    elif f['edge_density'] >= 0.02:
        f['category'] = 'texture'
    else:
        f['category'] = 'clear_sea'
    return f


def main(out):
    ann = json.load(open(ANN))
    pos = {a['image_id'] for a in ann['annotations']}
    feats = {}
    for img in ann['images']:
        f = describe(IMG + img['file_name'])
        f['negative'] = img['id'] not in pos
        f['file_name'] = img['file_name']
        feats[img['id']] = f
    json.dump(feats, open(out, 'w'), indent=1)
    for neg in (True, False):
        cats = [f['category'] for f in feats.values() if f['negative'] == neg]
        print('negative' if neg else 'positive',
              {c: cats.count(c) for c in sorted(set(cats))})


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else osp.join(WORK, 'scene_features.json'))
