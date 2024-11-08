## Introduction
Status: Archive (code is provided as-is, no updates expected)
### Inference code
Code for reproducing results in the paper __A deep learning-based framework for hidden danger detection of transmission lines in complex backgrounds__.

## Network Architecture
![pipeline](https://github.com/DearPerpetual/HDDet/blob/main/model.png)

## Results
<p>
<img src="https://github.com/DearPerpetual/HDDet/blob/main/runs/detect/exp/1056.jpg", width="360">
<img src="https://github.com/DearPerpetual/HDDet/blob/main/runs/detect/exp/162.jpg", width="360">
<img src="https://github.com/DearPerpetual/HDDet/blob/main/runs/detect/exp/58.jpg", width="360">
<img src="https://github.com/DearPerpetual/HDDet/blob/main/runs/detect/exp/84.jpg", width="360">
</p>


## Require
Please `pip install` the following packages:
- gitpython
- ipython
- python
- matplotlib
- tqdm
- torch
- tensorboard
- pandas
- seaborn
- PyYAML
- opencv-python

## Development Environment

Running on Ubuntu 16.04 system with pytorch.

## Inference
### step 1: Install python packages in requirement.txt.

### step 2: Download the weight `output/model.pth` to the root directory.

- Model weights and test results download link：[64ix](https://pan.baidu.com/s/1rFHj47XtQNIj9PRh3_YpVg).

### step 3: Run the following script to obtain detection results in the testing image.
  `python detect.py
  
- Test results：

![test](https://github.com/DearPerpetual/HDDet/blob/main/TestProcessing.png)

![000033](https://github.com/DearPerpetual/MFPLNet/blob/main/work_dirs/out/swin_t_tusimple/20240925_121139_lr_1e-03_b_8/visualization/clips_0530_00000_445.jpg)

__Note: The testing images are all shot by UAV and the resolution was adjusted to `360x540`.__
