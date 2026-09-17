"""Machine-dependent roots for data, work dirs and external repos.

Every path the tools use is derived from a single root, overridable with the
TOD_ROOT environment variable so the project moves between machines without
code edits.  Layout under the root:  data/levir/, work/, ext/, torch_home/.
"""
import os
import os.path as osp

ROOT = os.environ.get('TOD_ROOT', '/home/arge/tod')
DATA = osp.join(ROOT, 'data', 'levir')
WORK = osp.join(ROOT, 'work')
EXT = osp.join(ROOT, 'ext')


def ann(split):
    return osp.join(DATA, 'annotations', 'instances_%s2017.json' % split)


def img(split):
    return osp.join(DATA, 'images', '%s2017' % split) + '/'
