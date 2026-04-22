# HIA-DDP
The official code for the paper “HIA-DDP: Hard-Instance-Aware Dynamic Data Pruning with Diffusion-based Augmentation and Knowledge-Prior Guidance for 2D/3D Lung Nodule Segmentation” is provided in this repository. The `code` directory contains the implementations for both 2D and 3D lung nodule segmentation.

## Dataset

This is the generated 3D lung nodule dataset used in the paper:  
[https://drive.google.com/drive/folders/1DRobAG4q74TmFleKz4KJNf9G2zIrJeLc?usp=drive_link](https://drive.google.com/drive/folders/1DRobAG4q74TmFleKz4KJNf9G2zIrJeLc?usp=drive_link)

This is the generated 2D lung nodule dataset used in the paper:  
(under review)

This is the original dataset used in the paper:  
[https://data.mendeley.com/datasets/f32p8brkgd/4](https://data.mendeley.com/datasets/f32p8brkgd/4)

## Training

To train the model, run:

```bash
python trainLuna.py
```

## Acknowledgements

We would like to sincerely thank the authors of the following works for making their code and dataset publicly available. Their valuable resources provided important support for this project.

This repository benefited from the following resources:

- Jiang, W., Zhi, L., Zhang, S., Zhou, T.  
  *A Dual-Branch Framework With Prior Knowledge for Precise Segmentation of Lung Nodules in Challenging CT Scans*.  
  *IEEE Journal of Biomedical and Health Informatics*, 2024.

- Zhi, L., Jiang, W., Zhang, S., Zhou, T.  
  *Deep neural network pulmonary nodule segmentation methods for CT images: Literature review and experimental comparisons*.  
  *Mendeley Data*, V4, 2024.  
  DOI: 10.17632/f32p8brkgd.4

- Konz, N., Chen, Y., Dong, H., Mazurowski, M. A.  
  *Anatomically-Controllable Medical Image Generation with Segmentation-Guided Diffusion Models*.  
  *MICCAI*, 2024.

- Chen, Q., Chen, X., Song, H., Xiong, Z., Yuille, A., Wei, C., Zhou, Z.  
  *Towards Generalizable Tumor Synthesis*.  
  *CVPR*, 2024.
