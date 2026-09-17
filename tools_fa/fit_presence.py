"""Fit the scene-presence branch on a frozen, already-trained detector.

Every joint-training variant we measured paid for the presence branch with
detection quality, by between one and six AP50 points depending on the base
detector, while the calibration the branch enables was worth 25 to 35 points of
false-alarm image rate under exact weight matching.  Fitting the branch after
the fact keeps the gain and removes the cost: the detector is the original
checkpoint, bit for bit, and the method applies to detectors someone else
trained.

The branch is a 1x1 convolution on the classification tower, 257 parameters,
trained with image-level labels through a top-k pooled logit -- the same head
and the same pooling used in joint training, so inference is unchanged.

Usage:
  python tools_fa/fit_presence.py RUN [--iters 2000] [--lr 0.01] [--seed 0]
"""
import argparse
import glob
import os
import os.path as osp
import sys
import time

import torch
import torch.nn.functional as F
from mmcv import Config
from mmcv.runner import load_checkpoint

from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from paths import WORK, ann, img


def run_config(work_dir):
    """The config a run was trained with.

    Skips the inference configs this tool generates next to it: one of those
    names its own source as _base_, and picking it up recurses forever.
    """
    hits = [f for f in sorted(glob.glob(osp.join(work_dir, '*.py')))
            if not osp.basename(f).startswith('posthoc_cfg')]
    if not hits:
        raise SystemExit('no training config in ' + work_dir)
    return hits[0]


def build_frozen(work_dir, presence_cfg, device='cuda', config=None,
                 ckpt_path=None):
    """The run's detector with a presence branch attached and nothing else trainable.

    `config` overrides the config dumped in the run directory, which is how a
    checkpoint trained elsewhere -- a released model, say -- is loaded: point it
    at a config whose head is FAFCOSHead, a subclass of FCOSHead that adds only
    the presence convolution, so every detection weight still matches by name.
    """
    cfg = Config.fromfile(config or run_config(work_dir))
    # the checkpoint supplies every detection weight, so skip pretrained loading;
    # backbones differ in which key they accept, so only clear the ones present
    for key in ('init_cfg', 'pretrained'):
        if key in cfg.model.backbone:
            cfg.model.backbone[key] = None
    cfg.model.bbox_head.presence = presence_cfg
    model = build_detector(cfg.model)
    # the checkpoint predates the branch, so its weights are simply absent
    load_checkpoint(model, ckpt_path or osp.join(work_dir, 'latest.pth'),
                    map_location='cpu', strict=False)
    for name, p in model.named_parameters():
        p.requires_grad_('conv_presence' in name)
    trainable = [p for n, p in model.named_parameters() if p.requires_grad]
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert names and all('conv_presence' in n for n in names), names
    print(f'trainable: {names} '
          f'({sum(p.numel() for p in trainable)} parameters)')
    return cfg, model.to(device), trainable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run')
    parser.add_argument('--iters', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--topk', type=int, default=4)
    parser.add_argument('--work-root', default=WORK)
    parser.add_argument('--out', help='checkpoint to write (default: RUN/presence.pth)')
    parser.add_argument('--config', help='config to build from, overriding the '
                                         'one dumped in the run directory')
    parser.add_argument('--ckpt', help='checkpoint to load (default: RUN/latest.pth)')
    parser.add_argument('--data-from-config', action='store_true',
                        help="fit on the config's own training set instead of "
                             'the default benchmark, for a second dataset')
    parser.add_argument('--evaluate', action='store_true',
                        help='after fitting, run the test split with calibration '
                             'on and off and write metrics for both')
    parser.add_argument('--eval-config',
                        help='config for --evaluate (default: --config, else the '
                             'one dumped in the run directory)')
    parser.add_argument('--eval-name',
                        help='work-dir prefix for --evaluate results '
                             '(default: <run>_Ppost)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    wd = osp.join(args.work_root, f'{args.run}_s{args.seed}')
    if not osp.isdir(wd):  # runs named without a matching seed suffix
        wd = sorted(glob.glob(osp.join(args.work_root, args.run + '_s*')))[0]
    presence = dict(topk=args.topk, loss_weight=1.0, gamma=1.0)
    cfg, model, trainable = build_frozen(wd, presence, config=args.config,
                                        ckpt_path=args.ckpt)

    # object-free images are the whole point, so they must not be filtered out
    train = cfg.data.train.copy()
    train['filter_empty_gt'] = False
    if not args.data_from_config:
        train['ann_file'] = ann('train')
        train['img_prefix'] = img('train')
    dataset = build_dataset(train)
    loader = build_dataloader(dataset, samples_per_gpu=args.batch,
                              workers_per_gpu=4, dist=False, shuffle=True,
                              seed=args.seed)

    model.eval()  # frozen BN and no dropout; only the branch learns
    head = model.bbox_head
    opt = torch.optim.Adam(trainable, lr=args.lr)
    step, t0, losses = 0, time.time(), []
    while step < args.iters:
        for data in loader:
            if step >= args.iters:
                break
            imgs = data['img'].data[0].cuda()
            # one target per class: 1 where the image holds an instance of it
            labels = data['gt_labels'].data[0]
            target = torch.zeros(len(labels), head.cls_out_channels,
                                 device=imgs.device)
            for i, lab in enumerate(labels):
                if len(lab):
                    target[i, lab.unique()] = 1.0
            with torch.no_grad():
                feats = model.extract_feat(imgs)
            *_, presence_maps = head(feats)
            logit = head.image_presence_logits(presence_maps)
            loss = F.binary_cross_entropy_with_logits(logit, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
            step += 1
            if step % 200 == 0:
                print(f'  iter {step:5d}/{args.iters}  '
                      f'loss {sum(losses[-200:]) / len(losses[-200:]):.4f}  '
                      f'{time.time() - t0:.0f}s', flush=True)

    out = args.out or osp.join(wd, 'presence.pth')
    ckpt = torch.load(args.ckpt or osp.join(wd, 'latest.pth'), map_location='cpu')
    sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    for k in ('bbox_head.conv_presence.weight', 'bbox_head.conv_presence.bias'):
        sd[k] = model.state_dict()[k].cpu()
    torch.save(dict(state_dict=sd, meta=ckpt.get('meta', {})), out)
    print(f'wrote {out}  (final loss {sum(losses[-200:]) / 200:.4f}, '
          f'{time.time() - t0:.0f}s)')
    if args.evaluate:
        evaluate_both(args, out)


def evaluate_both(args, ckpt):
    """Test the fitted model with calibration on and off.

    The gamma=0 pass is the check that matters: it must reproduce the original
    run's metrics exactly, since the detector weights were never touched.
    """
    import subprocess
    root = osp.dirname(osp.dirname(osp.abspath(__file__)))
    src = args.eval_config or args.config or run_config(osp.dirname(ckpt))
    base = args.eval_name or (args.run + '_Ppost')
    # The run's own config has presence=None, and --cfg-options cannot add keys
    # under a None, so derive a config that switches the branch on.
    cfg = osp.join(osp.dirname(ckpt), f'posthoc_cfg_{base}.py')
    with open(cfg, 'w') as f:
        f.write("# Generated by tools_fa/fit_presence.py: the run's own config\n"
                "# with the post-hoc presence branch enabled for inference.\n"
                f"_base_ = '{osp.abspath(src)}'\n"
                "model = dict(bbox_head=dict(presence=dict(\n"
                f"    _delete_=True, topk={args.topk}, loss_weight=1.0, gamma=1.0)))\n")
    for gamma in (1, 0):
        wd = osp.join(args.work_root,
                      f'{base}{"" if gamma else "_g0"}_s{args.seed}')
        os.makedirs(wd, exist_ok=True)
        pkl = osp.join(wd, 'test_results.pkl')
        opts = ['model.bbox_head.presence.gamma=%d' % gamma]
        if not args.data_from_config:
            opts += ['data.test.ann_file=' + ann('test'),
                     'data.test.img_prefix=' + img('test')]
        subprocess.run([sys.executable, 'tools/test.py', cfg, ckpt,
                        '--out', pkl, '--cfg-options'] + opts,
                       cwd=root, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run([sys.executable, 'tools_fa/fa_eval.py', ann('test'),
                        pkl, '--out', osp.join(wd, 'metrics.json')],
                       cwd=root, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f'  gamma={gamma} -> {wd}')


if __name__ == '__main__':
    main()
