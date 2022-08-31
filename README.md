# [ECCV 2022]Ghost-free High Dynamic Range Imaging with Context-aware Transformer

By Zhen Liu<sup>1</sup>, Yinglong Wang<sup>2</sup>, Bing Zeng<sup>3</sup> and [Shuaicheng Liu](http://www.liushuaicheng.org/)<sup>3,1*</sup>

<sup>1</sup>Megvii Technology, <sup>2</sup>Noah’s Ark Lab, Huawei Technologies, <sup>3</sup>University of Electronic Science and Technology of China

This is the official MegEngine implementation of our ECCV2022 paper: *Ghost-free High Dynamic Range Imaging with Context-aware Transformer* ([HDR-Transformer](https://arxiv.org/abs/2208.05114)). The PyTorch version is available at [HDR-Transformer-PyTorch](https://github.com/liuzhen03/HDR-Transformer-PyTorch).

## News
* **2022.08.26** The PyTorch implementation is now avaible.
* **2022.08.11** The arXiv version of our paper is now available.
* **2022.07.19** The source code is now available.
* **2022.07.04** Our paper has been accepted by ECCV 2022.

## Abstract 
High dynamic range (HDR) deghosting algorithms aim to generate ghost-free HDR images with realistic details. Restricted by the locality of the receptive field, existing CNN-based methods are typically prone to producing ghosting artifacts and intensity distortions in the presence of large motion and severe saturation. In this paper, we propose a novel Context-Aware Vision Transformer (CA-ViT) for ghost-free high dynamic range imaging. The CA-ViT is designed as a dual-branch architecture, which can jointly capture both global and local dependencies. Specifically, the global branch employs a window-based Transformer encoder to model long-range object movements and intensity variations to solve ghosting. For the local branch, we design a local context extractor (LCE) to capture short-range image features and use the channel attention mechanism to select informative local details across the extracted features to complement the global branch. By incorporating the CA-ViT as basic components, we further build the HDR-Transformer, a hierarchical network to reconstruct high-quality ghost-free HDR images. Extensive experiments on three benchmark datasets show that our approach outperforms state-of-the-art methods qualitatively and quantitatively with considerably reduced computational budgets.

## Pipeline
![pipeline](https://user-images.githubusercontent.com/1344482/181019035-dc3b141d-0cd7-407e-83c9-8c6fbbc36d4f.JPG)
Illustration of the proposed CA-ViT. As shown in Fig (a), the CA-ViT is designed as a dual-branch architecture where the global branch models long-range dependency among image contexts through a multi-head Transformer encoder, and the local branch explores both intra-frame local details and inner-frame feature relationship through a local context extractor. Fig. (b) depicts the key insight of our HDR deghosting approach with CA-ViT. To remove the residual ghosting artifacts caused by large motions of the hand (marked with blue), long-range contexts (marked with red), which are required to hallucinate reasonable content in the ghosting area, are modeled by the self-attention in the global branch. Meanwhile, the well-exposed non-occluded local regions (marked with green) can be effectively extracted with convolutional layers and fused by the channel attention in the local branch.



## Usage

### Requirements
* Python 3.7.0
* MegEngine 1.8.3+
* CUDA 10.0 on Ubuntu 18.04

Install the require dependencies:
```bash
conda create -n hdr_transformer python=3.7
conda activate hdr_transformer
pip install -r requirements.txt
```

### Dataset
1. Download the dataset (include the training set and test set) from [Kalantari17's dataset](https://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/)
2. Move the dataset to `./data` and reorganize the directories as follows:
```
./data/Training
|--001
|  |--262A0898.tif
|  |--262A0899.tif
|  |--262A0900.tif
|  |--exposure.txt
|  |--HDRImg.hdr
|--002
...
./data/Test (include 15 scenes from `EXTRA` and `PAPER`)
|--001
|  |--262A2615.tif
|  |--262A2616.tif
|  |--262A2617.tif
|  |--exposure.txt
|  |--HDRImg.hdr
...
|--BarbequeDay
|  |--262A2943.tif
|  |--262A2944.tif
|  |--262A2945.tif
|  |--exposure.txt
|  |--HDRImg.hdr
...
```
3. Prepare the corpped training set by running:
```
cd ./dataset
python gen_crop_data.py
```

### Training & Evaluaton
```
cd HDR-Transformer
```
To train the model, run:
```
python train.py --model_dir experiments
```
To evaluate, run:
```
python evaluate.py --model_dir experiments --restore_file experiments/val_model_best.pth
```

## Results
![results](https://user-images.githubusercontent.com/1344482/181019317-94fa0ce6-a386-44a0-b59b-c10def8bc8ce.JPG)



## Acknowledgement
The MegEngine version of the Swin-Transformer is based on [Swin-Transformer-MegEngine](https://github.com/MegEngine/swin-transformer). Our work is inspired the following works and uses parts of their official implementations:

* [Swin-Transformer](https://github.com/microsoft/Swin-Transformer)
* [SwinIR](https://github.com/JingyunLiang/SwinIR)

We thank the respective authors for open sourcing their methods.

## Citation
```
@article{liu2022ghost,
  title={Ghost-free High Dynamic Range Imaging with Context-aware Transformer},
  author={Liu, Zhen and Wang, Yinglong and Zeng, Bing and Liu, Shuaicheng},
  journal={arXiv preprint arXiv:2208.05114},
  year={2022}
}
```

## Contact
If you have any questions, feel free to contact Zhen Liu at liuzhen03@megvii.com.
