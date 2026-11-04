## Installation

Create and activate a Conda environment:

```bash
conda create -n dcf python=3.11 -y
conda activate dcf
```

Install the dependencies from the repository root:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
```

The supplied requirements pin PyTorch **2.2.0** and Torchvision **0.17.0** with
CUDA **11.8**.

## Data & Models

### Datasets

Download the datasets needed for your experiments. Each benchmark has a dataset
configuration and a matching DCF adapter configuration.

| Benchmark | Dataset | Configuration key |
| :--- | :--- | :--- |
| ImageNet → ImageNet-C | [Zenodo](https://zenodo.org/record/2235448) · [Project](https://github.com/hendrycks/robustness) | `imagenet` |
| ImageNet → ImageNet-C-Bar | [Download](https://dl.fbaipublicfiles.com/inc_bar/imagenet_c_bar.tar.gz) | `imagenet_c_bar` |
| ImageNet → ImageNet-3DCC | [Project & downloads](https://github.com/EPFL-VILAB/3DCommonCorruptions#3dcc-data) | `imagenet_3dcc` |
| ImageNet → ImageNet-R | [Download](https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar) · [Project](https://github.com/hendrycks/imagenet-r) | `imagenet_r` |
| ImageNet → ImageNet-Sketch | [Project & downloads](https://github.com/HaohanWang/ImageNet-Sketch) | `imagenet_k` |

Set `DATA_DIR` in `conf.py`, or override it from the command line with
`DATA_DIR "/path/to/datasets"`.

### Pretrained models

The code supports pretrained backbones from
[RobustBench](https://github.com/RobustBench/robustbench),
[Torchvision](https://pytorch.org/vision/stable/models.html), and
[timm](https://huggingface.co/timm). The default DCF configuration uses
**ResNet-50** with a batch size of **64**; model configurations are available in
[`configs/models/`](configs/models/).

## Quick Start

Run the following commands from the repository root after preparing the data.

### Run DCF on ImageNet-C

```bash
python CS.py \
  -acfg configs/adapter/imagenet/DCF.yaml \
  -dcfg configs/dataset/imagenet.yaml \
  -ocfg configs/order/imagenet/0.yaml \
  -mcfg configs/models/resnet50.yaml \
  SEED 0 \
  TEST.BATCH_SIZE 64 \
  CORRUPTION.SEVERITY "[5]" \
  DATA_DIR "/path/to/datasets"
```

### Evaluate long-horizon adaptation

```bash
python L-CS.py \
  -acfg configs/adapter/imagenet/DCF.yaml \
  -dcfg configs/dataset/imagenet.yaml \
  -ocfg configs/order/imagenet/0.yaml \
  -mcfg configs/models/resnet50.yaml \
  SEED 0 \
  TEST.BATCH_SIZE 64 \
  TEST.ROUNDS 20 \
  DATA_DIR "/path/to/datasets"
```

| Entry point | Evaluation setting |
| :--- | :--- |
| [`CS.py`](CS.py) | Continual adaptation across corruption domains |
| [`L-CS.py`](L-CS.py) | Long-horizon adaptation; configurable rounds, default `20` |
| [`run.sh`](run.sh) | Ready-to-run ImageNet-C example with DCF and ResNet-50 |

`L-CS.py` also supports label-shift and mixed-severity streams. Prefix the
evaluation command with the corresponding environment variables:

| Stream | Environment variables |
| :--- | :--- |
| Temporal corruption shifts | Default |
| Label shifts | `DCF_STREAM_SCENARIO=label_shift DCF_LABEL_SHIFT_ALPHA=0.1` |
| Mixed severity | `DCF_STREAM_SCENARIO=mixed_severity` |

For ImageNet-R or ImageNet-Sketch, use the matching adapter and dataset
configurations with the same entry point. For example:

```bash
python L-CS.py \
  -acfg configs/adapter/imagenet_r/DCF.yaml \
  -dcfg configs/dataset/imagenet_r.yaml \
  -mcfg configs/models/resnet50.yaml \
  SEED 0 \
  TEST.ROUNDS 20 \
  DATA_DIR "/path/to/datasets"
```

ImageNet-C supports both `.pt` cache files and raw-image streaming. Results are
saved under `results/`, including per-round domain accuracy in `results/detail/`.
Set `DCF_RESULTS_DIR` to choose another output directory.

### Run in the background

After setting `DATA_DIR` in `conf.py`:

```bash
nohup bash run.sh > run.log 2>&1 &
```

## Baselines

The repository also includes the following test-time adaptation methods:

| Method | Paper | Code |
| :--- | :--- | :--- |
| BN Adapt | [Adaptive Batch Normalization](https://arxiv.org/abs/1603.04779) | [`methods/BN.py`](methods/BN.py) |
| Tent | [Paper](https://openreview.net/pdf?id=uXl3bZLkr3c) | [Repository](https://github.com/DequanWang/tent) |
| CoTTA | [Paper](https://arxiv.org/abs/2203.13591) | [Repository](https://github.com/qinenergy/cotta) |
| SAR | [Paper](https://openreview.net/pdf?id=g2YraF75Tj) | [Repository](https://github.com/mr-eggplant/SAR) |
| RoTTA | [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Yuan_Robust_Test-Time_Adaptation_in_Dynamic_Scenarios_CVPR_2023_paper.pdf) | [Repository](https://github.com/BIT-DA/RoTTA) |
| TRIBE | [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/29435) | [Repository](https://github.com/Gorilla-Lab-SCUT/TRIBE) |
| DeYO | [Paper](https://openreview.net/pdf?id=9w3iw8wDuE)  | [`methods/DeYO.py`](methods/DeYO.py) |
| AEA | [Paper](https://openreview.net/pdf?id=sEMJ1PLSZR) | [Repository](https://github.com/wonjeongchoi/AEA) |

## Acknowledgements

We thank the authors of [RobustBench](https://github.com/RobustBench/robustbench)
and the baseline methods above for sharing their code and resources.

## Contact

For questions or suggestions, open a GitHub issue or contact
[im@xhy.im](mailto:im@xhy.im).
