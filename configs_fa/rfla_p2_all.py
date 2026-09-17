_base_ = './_base_/fcos_p2_levir.py'
data = dict(train=dict(filter_empty_gt=False))
model = dict(
    bbox_head=dict(rf_assign=dict(_delete_=True, fpn_layer='p2', fraction=0.5)),
    train_cfg=dict(
        assigner=dict(
            _delete_=True,
            type='HieAssigner',
            ignore_iof_thr=-1,
            gpu_assign_thr=256,
            iou_calculator=dict(type='BboxDistanceMetric'),
            assign_metric='kl',
            topk=[3, 1],
            ratio=0.9)))
