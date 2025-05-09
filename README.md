<h1 align="center">Clustering-based Adaptive Query Generation for Semantic Segmentation</h1>

This repository is an official Pytorch implementation of the paper [**"Clustering-Based Adaptive Query Generation for Semantic Segmentation"**](https://ieeexplore.ieee.org/abstract/document/10949765) <br>
Yeong Woo Kim and Wonjun Kim <br>
***IEEE Signal Processing Letters***, 2025. </br>
<p align="center">
  <img src="https://github.com/user-attachments/assets/c873d121-c254-4048-b634-4e30d672a251" alt="The overall architecture of the proposed method."/>
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
sh train_city.sh      # for the Cityscapes experiment
```

## Results
### Quantitative results

| Model | Dataset    | Backbone | mIoU |
| ---   | ---        | ---      | ---  |
| CQG   | ADE20K     | R50      | 48.7 |
| CQG   | ADE20K     | Swin-B   | 55.4 |
| CQG   | ADE20K     | Swin-L   | 56.4 |
| CQG   | Cistyscaes | R50      | 80.8 |


### Qualitative results
![ade](https://github.com/user-attachments/assets/438f04ef-d222-47d5-b9fb-909860104ecb)
![city](https://github.com/user-attachments/assets/79fc35db-3d1a-4b1a-855b-c2c8fff517ad)
## Acknowledgments
This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korean Government [Ministry of Science and ICT (MSIT)] under Grant RS-2023-NR076462.

Our implementation and experiments are built on top of open-source GitHub repositories. We thank all the authors who made their code public, which tremendously accelerates our project progress. If you find these works helpful, please consider citing them as well.

[Mask2Former] [https://github.com/facebookresearch/Mask2Former](https://github.com/facebookresearch/Mask2Former)  </br>

## Citation
```bibtex
@ARTICLE{10949765,
  author={Kim, Yeong Woo and Kim, Wonjun},
  journal={IEEE Signal Processing Letters}, 
  title={Clustering-Based Adaptive Query Generation for Semantic Segmentation}, 
  year={2025},
  volume={32},
  number={},
  pages={1580-1584},
  doi={10.1109/LSP.2025.3558160}}
```
