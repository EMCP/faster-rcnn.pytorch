# An Updated *Faster* Pytorch Implementation of Faster R-CNN

## Introduction


This project is based on a *faster* pytorch implementation of faster R-CNN, which needed updating.  

* [jwyang/faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch), developed based on PyTorch v0.4.0 and v1.x

During my implementing, I referred the above implementation. However, my implementation has several unique and new features compared with the above implementation:

### What I did

* Drop Python v2.7.x support (Deprecated in 2020)
* Add Pytorch v1.5.x support
* Remove C++/Cuda NMS in favor of TorchVision python code version
* Removed a lot of unused or redundant configurations
* removed all but COCO based datasets

### What still needs to be done
* Migrate to torchvision based ResNet 101, instead of the original implementation

## Preparation


First of all, clone the code
```
git clone https://github.com/emcp/faster-rcnn.pytorch.git
```

Then, create a folder:
```
cd faster-rcnn.pytorch && mkdir data
```

### prerequisites

* Python 3.6+
* Pytorch 1.5+
* CUDA 10.0 or higher

### Data Preparation

* **COCO**: Please also follow the instructions in [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) to prepare the data.

### Pretrained Model

**If you want to use pytorch pre-trained models, please remember to transpose images from BGR to RGB, and also use the same data transformer (minus mean and normalize) as used in pretrained model.**

### Compilation

Install all the python dependencies using conda:
```
conda install environments.yml
```

## Train

Before training, set the right directory to save and load the trained models. Change the arguments "save_dir" and "load_dir" in config.json to adapt to your environment.

```
python train_net.py 
```

BATCH_SIZE and WORKER_NUMBER can be set adaptively according to your GPU memory size. **On Titan Xp with 12G memory, it can be up to 4**.

If you have multiple (say 8) Titan Xp GPUs, then just use them all! 

## Test

```
python test_net.py 
```

## Demo

If you want to run detection on your own images with a pre-trained model, download the pretrained model listed in above tables or train your own models at first, then add images to folder $ROOT/images, and then run

```
python demo.py 
```

Then you will find the detection results in folder $ROOT/images.

Below are some detection results:

<div style="color:#0000FF" align="center">
<img src="images/img3_det_res101.jpg" width="430"/> <img src="images/img4_det_res101.jpg" width="430"/>
</div>

## Webcam Demo

You can use a webcam in a real-time demo by running with modification under `dataset.json`
```
python demo.py
```
The demo is stopped by clicking the image window and then pressing the 'q' key.


## Citation

Please cite the original gentleman who contributed this wonderful work

    @article{jjfaster2rcnn,
        Author = {Jianwei Yang and Jiasen Lu and Dhruv Batra and Devi Parikh},
        Title = {A Faster Pytorch Implementation of Faster R-CNN},
        Journal = {https://github.com/jwyang/faster-rcnn.pytorch},
        Year = {2017}
    }

    @inproceedings{renNIPS15fasterrcnn,
        Author = {Shaoqing Ren and Kaiming He and Ross Girshick and Jian Sun},
        Title = {Faster {R-CNN}: Towards Real-Time Object Detection
                 with Region Proposal Networks},
        Booktitle = {Advances in Neural Information Processing Systems ({NIPS})},
        Year = {2015}
    }
