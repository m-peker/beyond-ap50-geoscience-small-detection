# LTDNet's released model with a scene-presence branch fitted post hoc.
# FAFCOSHead subclasses FCOSHead and adds only conv_presence, so every released
# detection weight loads by name and the detector is unchanged.
_base_ = './../configs_ltdnet/ltdnet/LTDNet_LEVIR_Ship.py'
model = dict(bbox_head=dict(
    type='FAFCOSHead',
    presence=dict(topk=4, loss_weight=1.0, gamma=1.0)))
