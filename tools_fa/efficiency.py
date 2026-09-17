"""Parameters, FLOPs and inference speed for the configs in the study.

FLOPs and parameters are architecture properties and need no trained weights,
so they run anywhere.  Throughput needs the GPU and is measured on real test
images through the normal inference path, after warm-up, reporting the median
over the timed images (the median ignores the occasional scheduler hiccup).

Usage:
  python tools_fa/efficiency.py [--fps] [--out efficiency.json]
"""
import argparse
import json
import os.path as osp
import time

import torch
from mmcv import Config
from mmcv.cnn import get_model_complexity_info
from mmcv.parallel import DataContainer

from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from paths import WORK, ann, img

CONFIGS = [
    ('configs_fa/fcos_p2_all.py', 'FCOS-P2'),
    ('configs_fa/rfla_p2_all.py', 'RFLA-P2'),
    ('configs_fa/nwd_p2_all.py', 'NWD-P2'),
    ('configs_fa/fa_fcos_p2_all_PN.py', 'Ours (P+N)'),
    ('configs_ltdnet/ltdnet/LTDNet_LEVIR_Ship.py', 'LTDNet'),
]
SHAPE = (3, 512, 512)


def complexity(cfg_path):
    cfg = Config.fromfile(cfg_path)
    if 'pretrained' in cfg.model.get('backbone', {}):
        cfg.model.backbone.pretrained = None
    if 'init_cfg' in cfg.model.get('backbone', {}):
        cfg.model.backbone.init_cfg = None
    model = build_detector(cfg.model)
    model.eval()
    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    flops, params = get_model_complexity_info(
        model, SHAPE, print_per_layer_stat=False, as_strings=False)
    return flops, params


def throughput(cfg_path, num_images=100, warmup=20):
    cfg = Config.fromfile(cfg_path)
    cfg.data.test.ann_file = ann('test')
    cfg.data.test.img_prefix = img('test')
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=2, dist=False,
        shuffle=False)
    if 'pretrained' in cfg.model.get('backbone', {}):
        cfg.model.backbone.pretrained = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg')).cuda()
    model.eval()

    def unwrap(value):
        """Undo the collate the model's own scatter would normally undo.

        The test loader yields img as a list of tensors and img_metas as a list
        of DataContainers; timing a forward pass means unwrapping them here.
        """
        out = []
        for item in value:
            if isinstance(item, DataContainer):
                item = item.data[0]
            out.append(item.cuda() if isinstance(item, torch.Tensor) else item)
        return out

    times = []
    with torch.no_grad():
        for i, data in enumerate(loader):
            if i >= warmup + num_images:
                break
            data = dict(img=unwrap(data['img']),
                        img_metas=unwrap(data['img_metas']))
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(return_loss=False, rescale=True, **data)
            torch.cuda.synchronize()
            if i >= warmup:
                times.append(time.perf_counter() - t0)
    times.sort()
    return 1.0 / times[len(times) // 2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fps', action='store_true',
                        help='also measure throughput (needs a free GPU)')
    parser.add_argument('--out', default=osp.join(WORK, 'efficiency.json'))
    args = parser.parse_args()

    rows = {}
    for cfg_path, label in CONFIGS:
        flops, params = complexity(cfg_path)
        rows[label] = dict(config=cfg_path,
                           params_M=round(params / 1e6, 3),
                           gflops=round(flops / 1e9, 2))
        if args.fps:
            rows[label]['fps'] = round(throughput(cfg_path), 1)
        print(label, rows[label])
    with open(args.out, 'w') as f:
        json.dump(rows, f, indent=1)
    print('written', args.out)


if __name__ == '__main__':
    main()
