<h1 align="center">Clustering-based Adaptive Query Generation for Semantic Segmentation</h1>

This repository is an official Pytorch implementation of the paper [**"Clustering-Based Adaptive Query Generation for Semantic Segmentation"**](https://ieeexplore.ieee.org/abstract/document/10949765) <br>
Yeong Woo Kim and Wonjun Kim <br>
***IEEE Access***, Jan. 2024. </br>
<p align="center">
  <img src="https://github.com/DCVL-WSSS/ClusterCAM/assets/49578893/82ccf953-05b2-4b3e-9441-90b3a247a493" alt="The overall architecture of the proposed method."/>
</p>

*The overall architecture of the proposed method.*

## Installation
Since this code is based on [Mask2Former](https://github.com/facebookresearch/Mask2Former), please follow the [Installation Guide](https://github.com/facebookresearch/Mask2Former/blob/main/INSTALL.md) of Mask2Former.

## Dataset Preparation
Since this code is based on [Mask2Former](https://github.com/facebookresearch/Mask2Former), please follow the [Dataset Preparation Guide](https://github.com/facebookresearch/Mask2Former/tree/main/datasets) of Mask2Former.

## How to use it
### Train
```bash
sh train_ade.sh       # for the ADE20K experiment
sh train_city.sh  # for the Cityscapes experiment
```

## Results
### Quantitative results

### Qualitative results


## Acknowledgments
This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korean Government [Ministry of Science and ICT (MSIT)] under Grant RS-2023-NR076462.

Our implementation and experiments are built on top of open-source GitHub repositories. We thank all the authors who made their code public, which tremendously accelerates our project progress. If you find these works helpful, please consider citing them as well.

[Mask2Former]([https://github.com/facebookresearch/Mask2Former](https://github.com/facebookresearch/Mask2Former))  </br>

## Citation
```bibtex
Y. W. Kim and W. Kim, "Clustering-Based Adaptive Query Generation for Semantic Segmentation," in IEEE Signal Processing Letters, doi: 10.1109/LSP.2025.3558160.
keywords: {Decoding;Vectors;Transformers;Semantic segmentation;Training;Semantics;Floors;Convolution;Shape;Phase frequency detectors;Semantic segmentation;deep learning;learnable queries;clustering-based adaptive query generation},
```
