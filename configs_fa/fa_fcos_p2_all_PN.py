_base_ = './fcos_p2_all.py'
model = dict(bbox_head=dict(presence=dict(_delete_=True, topk=4, loss_weight=1.0, gamma=1.0), cls_norm='ema'))
