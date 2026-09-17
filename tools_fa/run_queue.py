"""Sequential experiment queue: train -> test -> false-alarm evaluation.

The queue file (JSON list of {"config": ..., "seed": ..., "tag": optional,
"cfg_options": optional list}) is re-read after every job, so jobs can be
appended while the queue runs. Finished jobs (metrics.json present) are
skipped. GPU work waits while any process whose command line matches
--wait-for is alive (used to avoid competing with other users' jobs).
"""
import argparse
import json
import os
import os.path as osp
import subprocess
import sys
import time

from paths import WORK, ann

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
PY = sys.executable


def busy(pattern):
    """True while another python process matching ``pattern`` is running.

    Used to keep the queue off the GPU while someone else's job holds it.
    """
    if not pattern:
        return False
    try:
        out = subprocess.run(['pgrep', '-f', pattern],
                             capture_output=True, text=True, timeout=60)
        mine = {str(os.getpid()), str(os.getppid())}
        pids = [p for p in out.stdout.split() if p not in mine]
        return len(pids) > 0
    except Exception:
        return False


def job_name(job):
    base = osp.splitext(osp.basename(job['config']))[0]
    return f"{base}{job.get('tag', '')}_s{job['seed']}"


def run(cmd, log):
    with open(log, 'a') as f:
        f.write('\n$ ' + ' '.join(cmd) + '\n')
        f.flush()
        return subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode


def do_job(job, work_root, ann_file):
    name = job_name(job)
    wd = osp.join(work_root, name)
    os.makedirs(wd, exist_ok=True)
    log = osp.join(wd, 'queue.log')
    opts = job.get('cfg_options', [])
    ckpt = osp.join(wd, 'latest.pth')
    if 'ckpt_from' in job:  # test-only variant of an already trained run
        ckpt = osp.join(work_root, job['ckpt_from'], 'latest.pth')
        if not osp.exists(ckpt):
            return name, 'waiting for ' + job['ckpt_from']
    t0 = time.time()

    if not osp.exists(ckpt):
        cmd = [PY, 'tools/train.py', job['config'], '--work-dir', wd,
               '--seed', str(job['seed'])]
        if opts:
            cmd += ['--cfg-options'] + opts
        if run(cmd, log) != 0 or not osp.exists(ckpt):
            return name, 'train failed'
    t_train = time.time() - t0

    pkl = osp.join(wd, 'test_results.pkl')
    if not osp.exists(pkl):
        cmd = [PY, 'tools/test.py', job['config'], ckpt, '--out', pkl]
        test_opts = job.get('test_cfg_options', opts)
        if test_opts:
            cmd += ['--cfg-options'] + test_opts
        if run(cmd, log) != 0:
            return name, 'test failed'

    cmd = [PY, 'tools_fa/fa_eval.py', ann_file, pkl,
           '--out', osp.join(wd, 'metrics.json')]
    if run(cmd, log) != 0:
        return name, 'eval failed'
    with open(osp.join(wd, 'timing.json'), 'w') as f:
        json.dump(dict(train_seconds=t_train,
                       total_seconds=time.time() - t0), f)
    return name, 'done'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('queue')
    parser.add_argument('--work-root', default=WORK)
    parser.add_argument('--ann-file',
                        default=ann('test'))
    parser.add_argument('--wait-for', default='nn_queue.py')
    args = parser.parse_args()

    failed = set()
    # Jobs whose 'ckpt_from' run has not finished yet.  Skipped for now and
    # retried after the next completion, which may be the one they wait for.
    deferred = set()
    status_file = osp.join(args.work_root, 'queue_status.txt')
    os.makedirs(args.work_root, exist_ok=True)

    def next_job():
        with open(args.queue) as f:
            jobs = json.load(f)
        for j in jobs:
            name = job_name(j)
            if name in failed or name in deferred:
                continue
            if osp.exists(osp.join(args.work_root, name, 'metrics.json')):
                continue
            return j
        return None

    while True:
        while busy(args.wait_for):
            with open(status_file, 'a') as f:
                f.write(time.strftime('%H:%M:%S') + ' waiting for GPU\n')
            time.sleep(300)
        job = next_job()
        if job is None:
            if deferred:  # nothing runnable left, so the waits can never clear
                failed |= deferred
                deferred = set()
            break
        name, status = do_job(job, args.work_root, args.ann_file)
        if status.startswith('waiting for'):
            deferred.add(name)
        elif status != 'done':
            failed.add(name)
        else:
            deferred = set()  # this run may be what a deferred job waited for
        with open(status_file, 'a') as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {name} {status}\n")
    print('queue empty; failed:', sorted(failed))


if __name__ == '__main__':
    main()
