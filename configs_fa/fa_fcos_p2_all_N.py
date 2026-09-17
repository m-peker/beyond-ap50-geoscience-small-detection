_base_ = './fcos_p2_all.py'
model = dict(bbox_head=dict(cls_norm='ema'))
