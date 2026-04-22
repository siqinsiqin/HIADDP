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
