# [ECCV 2022]Ghost-free High Dynamic Range Imaging with Context-aware Transformer

By Zhen Liu<sup>1</sup>, Yinglong Wang<sup>2</sup>, Bing Zeng<sup>3</sup> and [Shuaicheng Liu](http://www.liushuaicheng.org/)<sup>3,1*</sup>

<sup>1</sup>Megvii Technology, <sup>2</sup>Noah’s Ark Lab, Huawei Technologies, <sup>3</sup>University of Electronic Science and Technology of China

This is the official MegEngine implementation of our ECCV2022 paper: *Ghost-free High Dynamic Range Imaging with Context-aware Transformer* ([HDR-Transformer]()). The PyTorch version is coming soon.

## News
* **2022.07.19** The source code is now available.
* **2022.07.04** Our paper has been accepted by ECCV 2022.

## Usage

### Requirements
* Python 3.7.0
* MegEngine 1.8.3
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

## Acknowledgement
The MegEngine version of the Swin-Transformer is based on [Swin-Transformer-MegEngine](https://github.com/MegEngine/swin-transformer). Our work is inspired the following works and uses parts of their official implementations:

* [Swin-Transformer](https://github.com/microsoft/Swin-Transformer)
* [SwinIR](https://github.com/JingyunLiang/SwinIR)

We thank the respective authors for open sourcing their methods.

## Citation
[to be updated]

## Contact
If you have any questions, feel free to contact Zhen Liu at liuzhen03@megvii.com.
