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
# Install core dependencies
pip install torch torchvision torchmetrics numpy mamba-ssm

📊 Dataset Preparation
The pipeline is pre-configured for the DUTS dataset. Please organize your data as follows:
train_data/
└── DUTS-TR/
    └── DUTS-TR/
        ├── im_aug/    # Training images (.jpg)
        └── gt_aug/    # Corresponding ground truth masks (.png)


🏗️ Model Architecture
The primary innovation is the MambaLayer integration, which processes spatial data as follows:
1. Normalization: Applies LayerNorm for gradient stability.
2. Serialization: Flattens 2D feature maps $\in \mathbb{R}^{C \times H \times W}$ into 1D sequences.
3. SSM Modeling: Learns global topological relationships via Mamba kernels.
$$ \text{Output} = \text{Mamba}(\text{LN}(\text{Flatten}(X))) $$


🚀 Training Instructions
1. Clone the Repository
git clone [https://github.com/your-username/u2net-mamba.git](https://github.com/your-username/u2net-mamba.git)
cd u2net-mamba

2. Run Training
Adjust hyperparameters in u2mamba_train.py (e.g., epoch_num, batch_size_train) as needed, then execute:
python u2mamba_train.py
