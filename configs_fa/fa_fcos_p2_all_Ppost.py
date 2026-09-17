# Baseline detector with a scene-presence branch fitted afterwards on frozen
# features (tools_fa/fit_presence.py).  The detector weights are the baseline's,
# unchanged; only the 257-parameter branch is new.
_base_ = './fcos_p2_all.py'
model = dict(bbox_head=dict(presence=dict(
    _delete_=True, topk=4, loss_weight=1.0, gamma=1.0)))
