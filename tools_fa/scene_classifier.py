"""A separate scene classifier, as the cascade baseline for the presence branch.

The natural objection to a 257-parameter head on shared features is that a
dedicated network would decide scene presence better.  This trains one -- an
ImageNet-pretrained ResNet-18 with a binary head, on the same tiles and labels
the branch sees -- and applies its probability through the same calibration, so
the only thing that differs is where the evidence comes from.

Its cost is the point of comparison: a second full forward pass over every tile
against one 1x1 convolution on features the detector already computed.

Usage:
  python tools_fa/scene_classifier.py [--epochs 20] [--out CKPT]
  python tools_fa/scene_classifier.py --predict CKPT --split test
"""
import argparse
import json
import os.path as osp
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import cv2
from pycocotools.coco import COCO
from torchvision.models import resnet18

from paths import WORK, ann, img

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class Tiles(Dataset):
    """LEVIR-Ship tiles with an image-level 'contains a ship' label."""

    def __init__(self, split, train=False):
        coco = COCO(ann(split))
        self.ids = coco.getImgIds()
        self.files = [coco.loadImgs([i])[0]['file_name'] for i in self.ids]
        self.labels = [float(len(coco.getAnnIds(imgIds=[i])) > 0)
                       for i in self.ids]
        self.dir, self.train = img(split), train

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        im = cv2.imread(osp.join(self.dir, self.files[i]))[:, :, ::-1]
        im = im.astype(np.float32) / 255
        if self.train and np.random.rand() < 0.5:
            im = im[:, ::-1]
        im = (im - MEAN) / STD
        return torch.from_numpy(np.ascontiguousarray(im.transpose(2, 0, 1))), \
            torch.tensor(self.labels[i])


def build():
    model = resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(512, 1)
    return model


def predict(ckpt, split, device='cuda'):
    model = build().to(device).eval()
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    loader = DataLoader(Tiles(split), batch_size=16, num_workers=4)
    out = []
    with torch.no_grad():
        for x, _ in loader:
            out.append(torch.sigmoid(model(x.to(device))[:, 0]).cpu().numpy())
    return np.concatenate(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default=osp.join(WORK, 'scene_classifier.pth'))
    parser.add_argument('--predict', help='checkpoint to run instead of training')
    parser.add_argument('--split', default='test')
    args = parser.parse_args()

    if args.predict:
        probs = predict(args.predict, args.split)
        dest = osp.join(WORK, f'scene_classifier_{args.split}.json')
        json.dump([float(p) for p in probs], open(dest, 'w'))
        print('wrote', dest, len(probs), 'probabilities')
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = build().cuda()
    train = DataLoader(Tiles('train', train=True), batch_size=args.batch,
                       shuffle=True, num_workers=4, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for x, y in train:
            loss = F.binary_cross_entropy_with_logits(
                model(x.cuda())[:, 0], y.cuda())
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
        sched.step()
        print(f'  epoch {epoch + 1:2d}/{args.epochs} loss '
              f'{sum(losses) / len(losses):.4f}  {time.time() - t0:.0f}s',
              flush=True)
    torch.save(model.state_dict(), args.out)
    n = sum(p.numel() for p in model.parameters())
    print(f'wrote {args.out}  ({n / 1e6:.2f} M parameters, '
          f'{time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
