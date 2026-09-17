"""False-alarm-aware evaluation for single-class tiny object detection.

Reports standard COCO AP together with operating-point metrics that expose
false alarms on object-free (negative) images, which AP50 largely hides:

  AP, AP50, AP75            COCO protocol on all test images
  AP50_pos                  AP50 on positive images only (no negatives)
  dAP50_neg                 AP50_pos - AP50: cost of false alarms on negatives
  R@FPPI{0.01,0.1}          recall at a false-positives-per-image budget
  LAMR                      log-average miss rate over FPPI in [1e-2, 1]
  FPPIneg@R{0.8,0.9}        FPs per negative image at a fixed recall
  FAimg@R{0.8,0.9}          fraction of negative images with >=1 false alarm
  NegFPshare@R0.8           share of all FPs that fall on negative images
  F1max                     best F1 over score thresholds

Usage:
  python tools_fa/fa_eval.py ANN_JSON RESULTS_PKL [--out metrics.json]
"""
import argparse
import contextlib
import io
import json
import pickle

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def load_detections(results, img_ids, class_index=0):
    """mmdet results (list over images of per-class arrays) -> per image.

    class_index picks the score array of one class; on a single-class benchmark
    that is the only one there is.
    """
    dets = {}
    for img_id, res in zip(img_ids, results):
        arr = res[class_index] if len(res) > class_index else np.zeros((0, 5))
        dets[img_id] = arr[np.argsort(-arr[:, 4])] if len(arr) else arr
    return dets


def box_iou(a, b):
    """a: (n,4) xyxy, b: (m,4) xyxy -> (n,m)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).prod(-1)
    return inter / (area_a[:, None] + area_b[None] - inter + 1e-9)


def match(dets, gts, iou_thr):
    """Greedy score-ordered matching. Returns scores, tp flags, image ids."""
    scores, tps, owners = [], [], []
    for img_id, d in dets.items():
        g = gts.get(img_id, np.zeros((0, 4)))
        ious = box_iou(d[:, :4], g)
        taken = np.zeros(len(g), bool)
        for i in range(len(d)):
            tp = False
            if len(g):
                cand = np.where(~taken, ious[i], -1)
                j = int(cand.argmax())
                if cand[j] >= iou_thr:
                    taken[j] = True
                    tp = True
            scores.append(d[i, 4])
            tps.append(tp)
            owners.append(img_id)
    return np.array(scores), np.array(tps, bool), np.array(owners)


def operating_curves(dets, gts, neg_ids, iou_thr=0.5):
    num_gt = sum(len(g) for g in gts.values())
    num_img = len(dets)
    scores, tps, owners = match(dets, gts, iou_thr)
    order = np.argsort(-scores, kind='stable')
    scores, tps, owners = scores[order], tps[order], owners[order]
    is_neg = np.isin(owners, list(neg_ids))

    tp_cum = np.cumsum(tps)
    fp_cum = np.cumsum(~tps)
    fp_neg_cum = np.cumsum((~tps) & is_neg)
    recall = tp_cum / max(num_gt, 1)
    fppi = fp_cum / num_img
    fppi_neg = fp_neg_cum / max(len(neg_ids), 1)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)

    # images with at least one false alarm, as the threshold is lowered
    first_hit = {}
    for k in np.where(is_neg & ~tps)[0]:
        first_hit.setdefault(owners[k], k)
    fa_img = np.zeros(len(scores))
    if first_hit:
        idx = np.sort(np.array(list(first_hit.values())))
        fa_img = np.searchsorted(idx, np.arange(len(scores)), side='right')
        fa_img = fa_img / max(len(neg_ids), 1)
    return dict(scores=scores, recall=recall, fppi=fppi, fppi_neg=fppi_neg,
                fa_img=fa_img, precision=precision, fp_cum=fp_cum,
                fp_neg_cum=fp_neg_cum)


def at_recall(c, key, r):
    idx = np.where(c['recall'] >= r)[0]
    return float(c[key][idx[0]]) if len(idx) else float('nan')


def recall_at_fppi(c, budget):
    idx = np.where(c['fppi'] <= budget)[0]
    return float(c['recall'][idx[-1]]) if len(idx) else 0.0


def lamr(c, lo=1e-2, hi=1.0, n=9):
    refs = np.logspace(np.log10(lo), np.log10(hi), n)
    miss = []
    for ref in refs:
        miss.append(1 - recall_at_fppi(c, ref))
    miss = np.clip(np.array(miss), 1e-10, None)
    return float(np.exp(np.log(miss).mean()))


def coco_ap(coco, results_json, img_ids, cat_ids=None):
    if not results_json:
        return dict(AP=0.0, AP50=0.0, AP75=0.0)
    with contextlib.redirect_stdout(io.StringIO()):
        dt = coco.loadRes(results_json)
        ev = COCOeval(coco, dt, 'bbox')
        ev.params.imgIds = list(img_ids)
        if cat_ids is not None:
            ev.params.catIds = list(cat_ids)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return dict(AP=float(ev.stats[0]), AP50=float(ev.stats[1]),
                AP75=float(ev.stats[2]))


def load_coco(ann_file, fix_zero_ann_id=True):
    """pycocotools treats annotation id 0 as 'unmatched' inside COCOeval, so a
    correct detection of that object is scored as a false positive. The
    official LEVIR-Ship COCO files contain one such id per split."""
    with open(ann_file) as f:
        dataset = json.load(f)
    if fix_zero_ann_id and any(a['id'] == 0 for a in dataset['annotations']):
        for a in dataset['annotations']:
            a['id'] += 1
    coco = COCO()
    coco.dataset = dataset
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
    return coco


def evaluate(ann_file, results, iou_thr=0.5, fix_zero_ann_id=True,
             category=None, recalls=(0.8, 0.9)):
    """False-alarm-aware metrics, for one class.

    On a multi-class benchmark `category` names the class of interest and
    everything is read relative to it: its detections, its ground truth, and as
    negatives the images holding no instance of it.  Object-free scenes are
    per class there -- AI-TOD-v2, for instance, has 18 images free of every
    class but 7690 free of ships.
    """
    coco = load_coco(ann_file, fix_zero_ann_id)
    img_ids = coco.getImgIds()
    cat_ids = coco.getCatIds()
    if category is None:
        cat_id, class_index = cat_ids[0], 0
    else:
        names = [c['name'] for c in coco.loadCats(cat_ids)]
        if category not in names:
            raise SystemExit(f'no category {category!r}; have {names}')
        class_index = names.index(category)
        cat_id = cat_ids[class_index]
    gts = {}
    for img_id in img_ids:
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id], catIds=[cat_id],
                                            iscrowd=False))
        gts[img_id] = np.array(
            [[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2],
              a['bbox'][1] + a['bbox'][3]] for a in anns]).reshape(-1, 4)
    neg_ids = {i for i in img_ids if len(gts[i]) == 0}
    pos_ids = [i for i in img_ids if i not in neg_ids]

    dets = load_detections(results, img_ids, class_index)
    json_res = [
        dict(image_id=img_id, category_id=cat_id,
             bbox=[float(b[0]), float(b[1]), float(b[2] - b[0]),
                   float(b[3] - b[1])], score=float(b[4]))
        for img_id, d in dets.items() for b in d
    ]
    m = coco_ap(coco, json_res, img_ids, cat_ids=[cat_id])
    m['AP50_pos'] = coco_ap(coco, [r for r in json_res
                                   if r['image_id'] not in neg_ids],
                            pos_ids, cat_ids=[cat_id])['AP50']
    m['dAP50_neg'] = m['AP50_pos'] - m['AP50']

    c = operating_curves(dets, gts, neg_ids, iou_thr)
    m['R@FPPI0.01'] = recall_at_fppi(c, 0.01)
    m['R@FPPI0.1'] = recall_at_fppi(c, 0.1)
    m['LAMR'] = lamr(c)
    # operating points have to be reachable: on a harder benchmark a detector
    # may never attain 0.9 recall, and the metric is then undefined rather than
    # bad.  Section V reports which points each benchmark supports.
    for r in recalls:
        m[f'FPPIneg@R{r}'] = at_recall(c, 'fppi_neg', r)
        m[f'FAimg@R{r}'] = at_recall(c, 'fa_img', r)
    m['maxRecall'] = float(c['recall'].max()) if len(c['recall']) else 0.0
    idx = np.where(c['recall'] >= recalls[0])[0]
    if len(idx):
        k = idx[0]
        m[f'NegFPshare@R{recalls[0]}'] = float(
            c['fp_neg_cum'][k] / max(c['fp_cum'][k], 1))
    else:
        m[f'NegFPshare@R{recalls[0]}'] = float('nan')
    f1 = 2 * c['precision'] * c['recall'] / np.maximum(
        c['precision'] + c['recall'], 1e-9)
    m['F1max'] = float(f1.max()) if len(f1) else 0.0
    m['num_neg_images'] = len(neg_ids)
    m['num_pos_images'] = len(pos_ids)
    return m, c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ann_file')
    parser.add_argument('results')
    parser.add_argument('--iou-thr', type=float, default=0.5)
    parser.add_argument('--out')
    parser.add_argument('--no-id-fix', action='store_true')
    parser.add_argument('--recalls', type=float, nargs='+',
                        default=[0.8, 0.9],
                        help='operating points; they must be reachable')
    parser.add_argument('--category',
                        help='class of interest on a multi-class benchmark; '
                             'negatives are then the images without it')
    args = parser.parse_args()
    with open(args.results, 'rb') as f:
        results = pickle.load(f)
    metrics, _ = evaluate(args.ann_file, results, args.iou_thr,
                          fix_zero_ann_id=not args.no_id_fix,
                          category=args.category,
                          recalls=tuple(args.recalls))
    if not args.no_id_fix:
        metrics['AP50_noidfix'] = evaluate(
            args.ann_file, results, args.iou_thr, fix_zero_ann_id=False,
            category=args.category,
            recalls=tuple(args.recalls))[0]['AP50']
    for k, v in metrics.items():
        print(f'{k:18s} {v:.4f}' if isinstance(v, float) else f'{k:18s} {v}')
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(metrics, f, indent=2)


if __name__ == '__main__':
    main()
