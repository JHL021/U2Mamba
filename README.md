# U²-Net with Mamba Integration 🐍

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the training code for an enhanced **U²-Net** model integrated with **Mamba (State Space Model)** layers, specifically designed for high-performance Salient Object Detection (SOD) tasks.

---

## 📖 Overview

This project implements a hybrid architecture that synergizes the multi-scale feature extraction capabilities of **U²-Net** with the efficient long-range dependency modeling of **Mamba**:

* **U²-Net:** Utilizes nested U-structures to capture fine-grained spatial details.
* **Mamba Layer:** Replaces standard bottlenecks with Selective State Space Models (SSM) to achieve global context awareness with linear $O(L)$ complexity.

---

## 🛠️ Requirements

The environment is optimized for **Python 3.10** and **CUDA 11.8+**.

```bash
### Install core dependencies
pip install torch torchvision torchmetrics numpy mamba-ssm
```
---

## 📊 Dataset Preparation
The pipeline is pre-configured for the DUTS dataset. Please organize your data as follows:
```bash
train_data/
└── DUTS-TR/
    └── DUTS-TR/
        ├── im_aug/    # Training images (.jpg)
        └── gt_aug/    # Corresponding ground truth masks (.png)
```
---

## 🏗️ Model Architecture
The primary innovation is the MambaLayer integration, which processes spatial data as follows:
1. Normalization: Applies LayerNorm for gradient stability.
2. Serialization: Flattens 2D feature maps $\in \mathbb{R}^{C \times H \times W}$ into 1D sequences.
3. SSM Modeling: Learns global topological relationships via Mamba kernels.


---

## 🚀 Training Instructions
1. Clone the Repository
git clone [https://github.com/your-username/u2net-mamba.git](https://github.com/your-username/u2net-mamba.git)
cd u2net-mamba

2. Run Training
Adjust hyperparameters in u2mamba_train.py (e.g., epoch_num, batch_size_train) as needed, then execute:
python u2mamba_train.py

---

## 📊 Performance Comparison

Comparison of our method and SOTA methods on baseline datasets:

| Model | DUTS-TE ($maxF_{\beta}$↑ / MAE↓) | PASCAL ($maxF_{\beta}$↑ / MAE↓) | DUT-OMRON ($maxF_{\beta}$↑ / MAE↓) | HKU-IS ($maxF_{\beta}$↑ / MAE↓) | ECSSD ($maxF_{\beta}$↑ / MAE↓) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| F3Net | 0.840 / 0.035 | 0.840 / 0.062 | 0.766 / 0.053 | 0.910 / 0.028 | 0.925 / 0.033 |
| RCSB | 0.855 / 0.034 | 0.842 / 0.058 | 0.773 / 0.045 | 0.923 / 0.027 | 0.823 / 0.033 |
| U²Net | 0.873 / 0.044 | 0.770 / 0.076 | 0.823 / 0.054 | 0.935 / 0.031 | 0.951 / 0.033 |
| MSENet | 0.877 / 0.034 | 0.862 / 0.060 | 0.798 / 0.045 | 0.927 / 0.026 | 0.941 / 0.033 |
| LDF | 0.855 / 0.034 | 0.848 / 0.060 | 0.773 / 0.051 | 0.914 / 0.027 | 0.930 / 0.034 |
| PoolNet | 0.809 / 0.040 | 0.822 / 0.074 | 0.747 / 0.056 | 0.899 / 0.032 | 0.915 / 0.039 |
| BBRF | 0.905 / 0.040 | 0.884 / 0.074 | 0.820 / 0.056 | 0.946 / 0.032 | 0.957 / 0.039 |
| MENet | 0.895 / 0.028 | 0.848 / 0.062 | 0.792 / 0.045 | 0.939 / 0.023 | 0.938 / 0.031 |
| VST | 0.877 / 0.037 | 0.850 / 0.067 | 0.800 / 0.058 | 0.937 / 0.030 | 0.944 / 0.034 |
| VST-S++ | 0.897 / 0.029 | 0.859 / 0.062 | 0.813 / 0.050 | 0.941 / 0.025 | 0.951 / 0.027 |
| **U²Mamba (Ours)** | **0.904 / 0.024** | **0.856 / 0.068** | **0.816 / 0.052** | **0.933 / 0.025** | **0.929 / 0.024** |

---

## 📊 Experimental Results

![Comparison of our method and SOTA methods](results.png)
