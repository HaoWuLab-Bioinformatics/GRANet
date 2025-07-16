# GRANet: Graph Residual Attention Network for Gene Regulatory Network Inference

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange.svg)](https://pytorch.org/)

## Overview
GRANet is a deep learning framework for inferring gene regulatory networks (GRNs) from single-cell RNA-seq data. It integrates:

- **Multi-head graph attention mechanisms**
- **Residual connections**
- **Multi-dimensional feature extraction**
- **Convolutional neural networks**

> **Abstract**:  
>
> ```
> The reconstruction of gene regulatory networks (GRNs) is crucial for uncovering regulatory relationships between genes and understanding the mechanisms of gene expression within cells. With advancements in single-cell RNA sequencing (scRNA-seq) technology, researchers have sought to infer GRNs at the single-cell level. However, existing methods primarily construct global models encompassing entire gene networks. While these approaches aim to capture genome-wide interactions, they frequently suffer from decreased accuracy due to challenges such as network scale, noise interference, and data sparsity.
> 
> This study proposes GRANet (Graph Residual Attention Network), a novel deep learning framework for inferring gene regulatory networks (GRNs). GRANet leverages residual attention mechanisms to adaptively learn complex gene regulatory relationships while integrating multi-dimensional biological features for a more comprehensive inference process. We evaluated GRANet across multiple datasets, benchmarking its performance against state-of-the-art methods. The experimental results demonstrate that GRANet consistently outperforms existing methods in GRN inference tasks.
> 
> In addition, in our case study on EGR1, CBFB, and ELF1, GRANet achieved high prediction accuracy, effectively identifying both known and novel regulatory interactions. These findings highlight GRANet's potential to advance research in gene regulation and disease mechanisms.
> ```

### Installation
```bash
git clone https://github.com/HaoWuLab-Bioinformatics/GRANet

pip install -r requirements.txt
```

## Usage

__GRANet requires single-cell RNA-seq data in the following format__:

Input: `N x G` expression matrix (N cells × G genes)

**Command to run model**

The model can be run directly using the main file in the directory, or through the command line:

`` python main.py``

##  Contact

For questions, contact:
Junliang Zhou - [Z1527304929@outlook.com](https://mailto:Z1527304929@outlook.com/)

