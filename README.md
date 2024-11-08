## Introduction
Status: Archive (code is provided as-is, no updates expected)
### Inference code
Code for reproducing results in the paper __A deep learning-based framework for hidden danger detection of transmission lines in complex backgrounds__.

## Network Architecture
![pipeline](https://github.com/DearPerpetual/HDDet/blob/main/image.png)

## Results
<p align="center">
<img src="https://github.com/DearPerpetual/HDDet/blob/main/runs/detect/exp/1056.jpg", width="360">
<p align="center">
<img src="https://github.com/DearPerpetual/MFPLNet/blob/main/work_dirs/out/swin_t_tusimple/20240925_121139_lr_1e-03_b_8/visualization/clips_0530_00000_795.jpg", width="360">
</p>
<p align="center">
<img src="https://github.com/DearPerpetual/MFPLNet/blob/main/work_dirs/out/swin_t_tusimple/20240925_121139_lr_1e-03_b_8/visualization/clips_0530_00000_857.jpg", width="360">
</p>


## Require
Please `pip install` the following packages:
- torch
- torchvision
- scikit-learn
- opencv-python
- tqdm
- imgaug
- yapf
- pathspec
- timm
- mmcv
- albumentations

## Development Environment

Running on Ubuntu 16.04 system with pytorch.

## Inference
### step 1: Install python packages in requirement.txt.

### step 2: Download the weight `output/model.pth` to the root directory.

- Model weights and test results download link：[64ix](https://pan.baidu.com/s/1rFHj47XtQNIj9PRh3_YpVg).

### step 3: Run the following script to obtain detection results in the testing image.
  `python main.py configs/clrnet/clr_swin_t_tusimple.py --validate --load_from [weight_path] --view --gpus 0`
- for example:
  `python main.py configs/clrnet/clr_resnet18_tusimple.py --validate --load_from work_dirs/clr/r18_tusimple/20240326_114727_lr_1e-03_b_40/ckpt/4.pth --view --gpus 0`
- Test results：

![test](https://github.com/DearPerpetual/MFPLNet/blob/main/Val.png)

![000033](https://github.com/DearPerpetual/MFPLNet/blob/main/work_dirs/out/swin_t_tusimple/20240925_121139_lr_1e-03_b_8/visualization/clips_0530_00000_445.jpg)

__Note: The testing images are all shot by UAV and the resolution was adjusted to `360x540`.__
