# The AI-TOD-v2 post-hoc setup read on the validation split, which nothing else
# in this study uses.  The presence branch is judged here, on data it was not
# fitted on and that the reported test numbers never see.
import os  # noqa: E402  (config is exec'd, not imported)

_base_ = './ltdnet_aitodv2_Ppost.py'

aitod_root = os.environ.get(
    'AITOD_ROOT', os.path.expanduser('~/tod/aitod_build/work/aitod')) + '/'
data = dict(test=dict(ann_file=aitod_root + 'annotations/aitodv2_val.json',
                      img_prefix=aitod_root + 'images/val/'))
