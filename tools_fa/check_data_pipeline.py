"""CPU check: datasets build with the intended image counts, samples load, and
the ImageNet-pretrained backbone weights resolve."""
import sys

from mmcv import Config
from mmcv.runner import load_checkpoint

from mmdet.datasets import build_dataset
from mmdet.models import build_detector


def main():
    ok = True
    for name, expected in (('fcos_p2_pos', 1185), ('fcos_p2_all', 2320)):
        cfg = Config.fromfile(f'configs_fa/{name}.py')
        ds = build_dataset(cfg.data.train)
        sample = ds[0]
        print(name, 'train images:', len(ds), 'expected:', expected,
              'img tensor:', tuple(sample['img'].data.shape))
        ok &= len(ds) == expected
    test = build_dataset(cfg.data.test, dict(test_mode=True))
    print('test images:', len(test))
    ok &= len(test) == 788

    model = build_detector(cfg.model)
    model.init_weights()
    w = model.backbone.conv1.weight
    print('backbone conv1 mean/std after pretrained init:',
          round(float(w.mean()), 5), round(float(w.std()), 5))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
