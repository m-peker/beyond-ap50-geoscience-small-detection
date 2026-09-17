# Beyond AP50

False-alarm-aware evaluation and post-hoc scene-evidence calibration for tiny
object detection in remote sensing imagery.

This is the reference implementation for *Beyond AP50: Diagnosing and
Suppressing False Alarms in Tiny Ship Detection from Medium-Resolution Optical
Satellite Imagery*.

---

## What this is

Tiny ship detection is benchmarked almost exclusively with AP50, and recent
methods on LEVIR-Ship sit within about two points of one another. AP50 is close
to blind to the error that governs operational cost: false alarms on sea with
no ship in it.

Three detectors differing **only in random seed** make the point:

| seed | AP50 | object-free tiles that raise an alarm at 90% recall |
|---|---|---|
| 0 | 78.99 | 40.9% |
| 1 | 78.31 | 94.7% |
| 2 | 80.47 | 100.0% |

A results table would read those as one method measured three times.

This repository provides three things.

**An evaluation protocol** (`tools_fa/fa_eval.py`) that complements COCO AP with
false positives per object-free image, the fraction of object-free images that
raise an alarm at a fixed recall, and the log-average miss rate — reported at
operating points rather than pooled over all detections.

**A calibration that needs no retraining** (`tools_fa/fit_presence.py`). A
1×1 convolution on the detector's own classification tower predicts whether the
image contains any object; every box score is rescaled by that probability. It
is fitted on a **frozen** detector, so the detector is unchanged by construction
— disabling the calibration reproduces its original metrics exactly — and the
method applies to checkpoints you did not train.

| base detector | AP50 | object-free tiles with an alarm @ 90% recall |
|---|---|---|
| FCOS-P2 | 81.6 ± 0.2 → 81.1 ± 0.2 | 37.8 ± 3.8 → **13.5 ± 0.4** |
| NWD-P2 | 82.0 ± 1.8 → 81.3 ± 1.9 | 33.8 ± 3.2 → **7.8 ± 0.8** |
| RFLA-P2 | 82.2 ± 0.8 → 81.5 ± 0.9 | 29.1 ± 4.1 → **9.1 ± 0.3** |
| LTDNet (released checkpoint) | 84.2 → 84.0 | 40.9 → **15.0** |

Mean ± std over three seeds. The branch is 257 parameters, fits in about seven
minutes on a frozen detector, and leaves throughput unchanged.

**An audit of the pipeline** that turned up two faults worth knowing about:
MMDetection's `filter_empty_gt` default removes the object-free half of the
LEVIR-Ship training set unless a config says otherwise, and one annotation
identifier per split is read by the reference COCO evaluator as a non-match,
understating every published AP50 on this benchmark by 0.274 ± 0.058 points.

## Installation

The code is an MMDetection 2.24.1 tree, extended from the
[LTDNet](https://github.com/dyl96/LTDNet) release. An RTX 40-series GPU needs
CUDA 11.8; older stacks will not run on `sm_89`.

```bash
conda create -y -p ./env python=3.8
./env/bin/python -m pip install torch==2.0.0+cu118 torchvision==0.15.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
./env/bin/python -m pip install mmcv-full==1.7.2 \
    -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html
./env/bin/python -m pip install opencv-python==4.8.1.78 "cython<3" matplotlib \
    scipy terminaltables pycocotools six tqdm timm==0.9.16 yapf==0.32.0 gdown
./env/bin/python -m pip install --no-deps -e .
```

Two pins matter. **`yapf==0.32.0`**: later versions break MMCV's config dump and
kill training mid-run. **`timm==0.9.16`**: LTDNet's RepViT backbone needs the
`_registry` API that 0.6.x does not have.

For AI-TOD-v2 you also need `aitodpycocotools`:

```bash
git clone https://github.com/jwwangchn/cocoapi-aitod
./env/bin/python -m pip install cocoapi-aitod/aitodpycocotools
```

## Data

All paths derive from one environment variable, `TOD_ROOT` (default
`/home/arge/tod`), laid out as `data/levir/`, `work/`, `ext/`, `torch_home/`.

LEVIR-Ship, official DRENet partition:

```bash
gdown 18M6-lnHl9V9Jf06wIpe3cX-aZE67DqHM -O ann.zip      # annotations
gdown 1ItolDrLdSN0R-AnKbD90ngSWMtKOCUaY -O images.zip   # 861 MB
unzip -q ann.zip -d $TOD_ROOT/data/levir/
unzip -q images.zip -d $TOD_ROOT/data/levir/
```

This should give 2320 / 788 / 788 images in train / val / test.

AI-TOD-v2 must be synthesised from the xView training set, which needs an
account at [challenge.xviewdataset.org](https://challenge.xviewdataset.org);
the [AI-TOD toolkit](https://github.com/jwwangchn/AI-TOD) does the rest. Note
that xView's `train_images.zip` is written in streaming mode with zip64 offsets
that Info-ZIP and Python both misread past 12 GiB — use the `.tgz` if the site
offers one.

## Reproducing the results

Everything runs through a queue that survives interruption: finished jobs are
skipped, so it can be stopped and restarted.

```bash
export TOD_ROOT=/path/to/tod TORCH_HOME=$TOD_ROOT/torch_home
python tools_fa/run_queue.py tools_fa/queue_main.json \
    --work-root $TOD_ROOT/work --wait-for ""
```

46 jobs, about 41 minutes each on one RTX 4060. `--wait-for PATTERN` holds the
queue while another process matching `PATTERN` is using the GPU.

Then, for the calibration on a trained detector:

```bash
python tools_fa/fit_presence.py fcos_p2_all --seed 0 --evaluate
```

This fits the branch on frozen features and evaluates with the calibration both
on and off. **The `gamma=0` row must reproduce the base detector exactly**; it
is the check that the detector was not touched.

Aggregation and figures:

```bash
python tools_fa/aggregate.py                  # mean ± std over seeds
python tools_fa/make_tables.py                # the paper's tables, from metrics.json
python tools_fa/plot_curves.py out.pdf fcos_p2_pos=... fcos_p2_all=...
python tools_fa/check_claims.py               # every number in the paper, recomputed
```

## What is in `tools_fa/`

| file | what it does |
|---|---|
| `fa_eval.py` | the protocol: AP plus operating-point false-alarm metrics, per class |
| `fit_presence.py` | fits the presence branch on a frozen detector, optionally evaluates |
| `run_queue.py` | the experiment queue, resumable, GPU-aware |
| `make_tables.py` | generates the paper's LaTeX tables from the metrics files |
| `check_claims.py` | recomputes every numeric claim in the paper |
| `tune_gamma.py` | calibration-strength sweep, analytic over γ |
| `gate_vs_calibrate.py` | soft calibration against a hard scene gate |
| `scene_classifier.py` | the cascade baseline: a separate ResNet-18 scene classifier |
| `eval_aitod.py`, `aitod_ap.py` | per-class evaluation on AI-TOD-v2, under its own protocol |
| `per_scene.py` | false alarms broken down by source acquisition |
| `clutter_analysis.py`, `scene_features.py` | clutter descriptors and the continuous analysis |
| `presence_hist.py`, `qualitative.py` | the figures |
| `efficiency.py` | parameters, FLOPs, throughput |
| `smoke_test_head.py`, `check_rfla_port.py`, `test_fa_eval.py`, `check_data_pipeline.py` | the checks that run before any GPU time is spent |

`configs_fa/` holds the detector configurations; `mmdet/models/dense_heads/fa_fcos_head.py`
is the head, and `mmdet/core/bbox/assigners/hierarchical_assigner.py` the RFLA
port, verified against the official implementation on random layouts.

## Provenance and licence

The MMDetection tree is version 2.24.1 as distributed with the
[LTDNet](https://github.com/dyl96/LTDNet) release, which is Apache-2.0 like
[MMDetection](https://github.com/open-mmlab/mmdetection) itself; that licence
is in `LICENSE` and applies to this repository. LEVIR-Ship is released by the
authors of [DRENet](https://github.com/WindVChen/LEVIR-Ship); AI-TOD-v2 by the
authors of its [toolkit](https://github.com/jwwangchn/AI-TOD). Please cite those
works if you use the datasets.

## Citation

```bibtex
@article{peker2026beyondap50,
  author  = {Peker, Musa},
  title   = {Beyond {AP50}: Diagnosing and Suppressing False Alarms in Tiny
             Ship Detection from Medium-Resolution Optical Satellite Imagery},
  journal = {under review},
  year    = {2026}
}
```
