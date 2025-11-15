## 💾 The code will be made public if the paper is accepted

## ⚙️ Prerequisites

To set up the environment, please follow these steps:

1.  **Create a Conda environment:**
    ```bash
    conda create -n tta python=3.8.1
    conda activate tta
    conda install -y ipython pip
    ```

2.  **Install required packages:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🛠️ Preparation

### 💾 Datasets

To run one of the following benchmark tests, you need to download the corresponding dataset.

*   **ImageNet → ImageNet-C**:
    *   Download the [ImageNet-C 🔗](https://github.com/hendrycks/robustness) dataset from [Zenodo 🔗](https://zenodo.org/record/2235448#.Yj2RO_co_mF).
*   **ImageNet → ImageNet-C-Bar**:
    *   Download the [ImageNet-C-Bar 🔗](https://github.com/hendrycks/robustness) dataset from the [Here🔗](https://dl.fbaipublicfiles.com/inc_bar/imagenet_c_bar.tar.gz).
*   **ImageNet → ImageNet-3DCC**:
    *   Download the [ImageNet-3DCC 🔗](https://github.com/hendrycks/robustness) dataset from the [EPFL-VILAB GitHub repository 🔗](https://github.com/EPFL-VILAB/3DCommonCorruptions?tab=readme-ov-file#3dcc-data).
*   **ImageNet → ImageNet-R**:
    *   Download the [ImageNet-R 🔗](https://github.com/hendrycks/imagenet-r) dataset from the [GitHub repository 🔗](https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar).
*   **ImageNet → ImageNet-Sketch**:
    *   Download the [ImageNet-Sketch 🔗](https://github.com/HaohanWang/ImageNet-Sketch) dataset from the [Google Drive 🔗](https://drive.google.com/open?id=1Mj0i5HBthqH1p_yeXzsg22gZduvgoNeA).

### 🧠 Models

For the Test-Time Adaptation (TTA) benchmarks, we utilize pre-trained models from:

*   [RobustBench](https://github.com/RobustBench/robustbench)
*   [Torchvision](https://pytorch.org/vision/0.14/models.html)
*   [Timm (Hugging Face)](https://huggingface.co/timm)

---

## ▶️ Run Experiments

We provide Python scripts and Bash scripts to run the experiments on a single **A100** GPU.

**Using Python Scripts:**

For example, to run the `IMAGENET → IMAGNET-C` benchmark with the `DCF` adapter, execute:

```bash
python L-CS.py -acfg configs/adapter/imagenet/DCF.yaml -dcfg configs/dataset/imagenet.yaml -ocfg configs/order/imagenet/0.yaml SEED 0
```

**Using Bash Scripts:**

For example, to run experiments defined in `run.sh` and log the output:

```bash
nohup bash run.sh > run.log 2>&1 &
```
This command runs the script in the background, detached from the terminal, and redirects standard output and standard error to `run.log`.

---

## 🏆 Competitors

This repository currently supports the following Test-Time Adaptation methods:

*   [**Tent**](https://openreview.net/pdf?id=uXl3bZLkr3c)
*   [**BN**](https://arxiv.org/pdf/1603.04779.pdf)
*   [**TEA**](https://openaccess.thecvf.com/content/CVPR2024/papers/Yuan_TEA_Test-time_Energy_Adaptation_CVPR_2024_paper.pdf)
*  [**AEA**](https://openreview.net/pdf?id=sEMJ1PLSZR)
*   [**CoTTA**](https://arxiv.org/abs/2203.13591) 
*   [**SAR**](https://openreview.net/pdf?id=g2YraF75Tj)
*   [**RoTTA**](https://openaccess.thecvf.com/content/CVPR2023/papers/Yuan_Robust_Test-Time_Adaptation_in_Dynamic_Scenarios_CVPR_2023_paper.pdf)
*   [**TRIBE**](https://ojs.aaai.org/index.php/AAAI/article/view/29435)

---

## 🙏 Acknowledgements

This project builds upon the excellent work from several open-source projects. We extend our sincere gratitude to their authors and contributors:

*   **RobustBench**: [Official Repository 🔗](https://github.com/RobustBench/robustbench)
*   **Tent**: [Official Repository 🔗](https://github.com/DequanWang/tent)
*   **TEA**: [Official Repository 🔗](https://github.com/yuanyige/tea)
*   **AEA**: [Official Repository 🔗](https://github.com/wonjeongchoi/AEA)
*   **CoTTA**: [Official Repository 🔗](https://github.com/qinenergy/cotta)
*   **SAR**: [Official Repository 🔗](https://github.com/mr-eggplant/SAR)
*   **RoTTA**: [Official Repository 🔗](https://github.com/BIT-DA/RoTTA)
*   **TRIBE**: [Official Repository 🔗](https://github.com/Gorilla-Lab-SCUT/TRIBE/)

---

## 📧 Contact

If you have any questions, suggestions, or bug reports regarding our work, please feel free to reach out:

*   **Primary Contact**: <a href="mailto:im@xhy.im">im@xhy.im</a>
*   Alternatively, you can open an issue in this GitHub repository.

---
