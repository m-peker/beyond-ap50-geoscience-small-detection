# The released LTDNet AI-TOD-v2 model with a per-class scene-presence branch
# fitted post hoc.  FAFCOSHead subclasses FCOSHead, which LTDNet uses, so every
# published detection weight loads by name and the detector is unchanged.
import os  # noqa: E402  (config is exec'd, not imported)

_base_ = '../configs_ltdnet/ltdnet/LTDNet_AI_TODv2.py'

aitod_root = os.environ.get(
    'AITOD_ROOT', os.path.expanduser('~/tod/aitod_build/work/aitod')) + '/'
ann_root = aitod_root + 'annotations/'

model = dict(
    backbone=dict(pretrained=None),
    bbox_head=dict(
        type='FAFCOSHead',
        presence=dict(topk=4, loss_weight=1.0, gamma=1.0)),
    # keep the released model's detection budget (max_per_img=3000): AI-TOD-v2
    # images hold 27 objects on average and the 300 we use on LEVIR-Ship is
    # binding here.  Only the score threshold is lowered, so the operating-point
    # curves extend below the 0.05 the released config stops at.
    test_cfg=dict(score_thr=0.001, max_per_img=3000))

data = dict(
    train=dict(ann_file=ann_root + 'aitodv2_train.json',
               img_prefix=aitod_root + 'images/train/'),
    val=dict(ann_file=ann_root + 'aitodv2_test.json',
             img_prefix=aitod_root + 'images/test/'),
    test=dict(ann_file=ann_root + 'aitodv2_test.json',
              img_prefix=aitod_root + 'images/test/'))
