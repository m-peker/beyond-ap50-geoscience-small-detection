"""Sanity checks for fa_eval on the real LEVIR-Ship test annotations."""
import sys

import numpy as np
from pycocotools.coco import COCO

from fa_eval import evaluate
from paths import ann

ANN = ann('test')


def perfect_results(coco, fp_score=None):
    results, neg_seen = [], False
    for img_id in coco.getImgIds():
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
        boxes = [[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2],
                  a['bbox'][1] + a['bbox'][3], 0.9] for a in anns]
        if not anns and fp_score is not None and not neg_seen:
            boxes.append([10, 10, 20, 20, fp_score])
            neg_seen = True
        results.append([np.array(boxes, dtype=np.float32).reshape(-1, 5)])
    return results


def main():
    coco = COCO(ANN)
    n_neg = sum(1 for i in coco.getImgIds() if not coco.getAnnIds(imgIds=[i]))
    ok = True

    m0, _ = evaluate(ANN, perfect_results(coco), fix_zero_ann_id=False)
    print('perfect, raw ids: AP50 =', round(m0['AP50'], 4))
    m, _ = evaluate(ANN, perfect_results(coco))
    print('perfect   ', {k: round(v, 4) for k, v in m.items()})
    ok &= abs(m['AP50'] - 1) < 1e-6 and m['LAMR'] < 1e-6
    ok &= m['FAimg@R0.9'] == 0 and m['dAP50_neg'] == 0

    m, _ = evaluate(ANN, perfect_results(coco, fp_score=0.1))
    print('low FP    ', {k: round(v, 4) for k, v in m.items()})
    ok &= abs(m['AP50'] - 1) < 1e-6 and m['FAimg@R0.9'] == 0

    m, _ = evaluate(ANN, perfect_results(coco, fp_score=0.99))
    print('high FP   ', {k: round(v, 4) for k, v in m.items()})
    ok &= abs(m['FAimg@R0.8'] - 1 / n_neg) < 1e-9
    ok &= abs(m['FPPIneg@R0.8'] - 1 / n_neg) < 1e-9
    ok &= m['NegFPshare@R0.8'] == 1.0 and m['dAP50_neg'] > 0
    print('PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
