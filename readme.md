# DREAM block for Pose-Robust Face Recognition
This is our implementation for our CVPR 2018 accepted paper *Pose-Robust Face Recognition via Deep Residual Equivariant Mapping* [paper on arxiv](https://arxiv.org/abs/1803.00839).

The code is wriiten by [Yu Rong](https://github.com/penincillin) and [Kaidi Cao](https://github.com/CarlyleCao)

## Prerequisites
- Linux or macOS
- Python 3
- NVIDIA GPU + CUDA CuDNN or CPU (GPU is prefered)
- opencv2.4 (opencv 2.4.13 is preferred)

## Getting Started
### Installation
- Install Anaconda 
    - [Anaconda3-4.2.0](https://repo.continuum.io/archive/Anaconda3-4.2.0-Linux-x86_64.sh) for Python 3
- Install Pytorch and torchvision through Anaconda (Please follow the guide in [Pytorch](pytorch.org))
- Clone this repo
```bash
git clone git@github.com:penincillin/DREAM.git
cd DREAM
```
### Prepare Data and Models
Big files like model.zip could be downloaded both from Google Drive and Baidu Yun. If you have problems with downloading those files, you could contact me :)
#### Face Alignment
All the face images should be aligned. Please follow the align protocol in dataset [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). After alignment, the face should be in the center of the image, and the size of image should be 178x218. Some aligned samples could be found in image/align_sample.

#### Datasets
In this paper, we use three face datasets. We train base model and DREAM block on [MS-Celeb-1M](https://www.microsoft.com/en-us/research/project/ms-celeb-1m-challenge-recognizing-one-million-celebrities-real-world/)  
We offer a subset of Ms-Celeb-1M with 10 celebrities, you could download from the following link  
Ms-Celeb-1M Subset (msceleb.zip): [Google Drive](https://drive.google.com/file/d/1om0pbwBX4RZHVuI3QXVrBj9mLtOK2PV8/view?usp=sharing) &nbsp; &nbsp; [Baidu Yun](https://pan.baidu.com/s/1Zviee1QXnB7noArpAoy7Iw)  

We evaluate our the performance of our models on CFP and IJB-A.
For CFP, we offer the code to get algined images from the original images (The code could only be runned on Linux). First, you need to download the original CFP dataset, and then download the image list from [Google Drive](https://drive.google.com/file/d/1B9QGThNd_-4Pg8O3si-EUYU9Px748p1C/view?usp=sharing) &nbsp; &nbsp; [Baidu Yun](https://pan.baidu.com/s/1U_CzmLsJ2OaX4rJeJ7r92g)

For IJBA, we provide the aligned images here. [Google Drive](https://drive.google.com/file/d/11p1eVSpyHZQUG0uBGyRoFnOXXTuZ501c/view?usp=sharing)　&nbsp; &nbsp;  [Baidu Yun](https://pan.baidu.com/s/1xLi6zDqwAeXEMV4aWi1k3g) 

#### Pretrained Models
We offer several pretrained models. They could be downloaded from [Google Drive](https://drive.google.com/open?id=1CrWbsyAvqTA14ET2wvks_4U2_h1P52qK) &nbsp; &nbsp; [Baidu Yun](https://pan.baidu.com/s/1LQmWZss0QoRc_chVIHsR_Q)

## Train DREAM Block
### stitch Training
Prepare the feature extracted from any face recognition model (You could use the pretrained model we prepared).   
We prepared a piece of sample data (stitching.zip) which could be download from [Google Drive](https://drive.google.com/file/d/1x1K8MxAnVtpfaN3DfO4bdcKH39mmplj-/view?usp=sharing) &nbsp; &nbsp; [Baidu Yun](https://pan.baidu.com/s/1QIEeE9RxRY6iK3wCpvUh2Q)  
- Download the sample data
```bash
mkdir data
mv stitching.zip data
cd data
unzip stitching.zip
```
- Train the model:
```bash
cd src/stitching
sh train_stitch.sh
```


### end2end Training
- Download the Ms-Celeb-1M Subset
```bash
mkdir data
mv msceleb.zip data
cd data
unzip msceleb.zip
```
- Train the model:
```bash
cd src/end2end
sh train.sh
```
### evaluate CFP
- Download the CFP dataset and preprocess the image. Then download the image list for evaluation
```bash
# make sure you are in the root directory of DREAM project
mkdir data
cd src/preprocess
sh align_cfp.sh
cd data/CFP
unzip CFP_protocol.zip
```
- Download pretrained model
```bash
# make sure you are in the root directory of DREAM project
cd ../ 
mv model.zip data
cd data
unzip model.zip
```
- Evaluate the pretrained model on CFP dataset
```bash
# make sure you are in the root directory of DREAM project
cd src/CFP
sh eval_cfp.sh
```

### evaluate IJBA
- Download the IJBA dataset(contact me to get the aligned images)
```bash
# make sure you are in the root directory of DREAM project
mkdir data
mv IJBA.zip data
cd data
unzip IJBA.zip
```
- Download pretrained models (If have downloaded the models, skip this step)
```bash
# make sure you are in the root directory of DREAM project
cd ../ 
mv model.zip data
cd data
unzip model.zip
```
- Evaluate the pretrained model on IJBA dataset
```bash
# make sure you are in the root directory of DREAM project
cd src/IJBA
sh eval_ijba.sh
```

## Citation
Please cite the paper in your publications if it helps your research:

    
    
    @inproceedings{cao2018Dream,
      author = {Kaidi Cao and Yu Rong and Cheng Li and Xiaoou Tang and Chen Change Loy},
      booktitle = {CVPR},
      title = {Pose-Robust Face Recognition via Deep Residual Equivariant Mapping},
      year = {2018}
      }

