# pytorch-yolo-v3
Good news !!!

In this update, I add yolo-v3 model which gets 80.8 mAP ( with 416 input, and no multi-scale training trick ) on VOC2007-test. For more details, you could see code files including ```models/yolo_v3.py``` and ```tools.py```.

In addition, you can replace darknet-53 with darknet-19 as the backbone of yolo-v3, and this model can get 78.6 mAP on VOC2007-test.

# pytorch-yolo-v2
I really enjoy yolo. It is so amazing! So I try to reproduce it. And I almost got it!

But !! I don't plan to one hundred percent reproduce it, so my own yolo-v2 is a little different with origin version. Because all the tricks used in yolo-v2 may not ne suitable for other tasks. 

It is known to us that reproducing the model is easy while it is hard to reproduce the results shown in the paper.

As the Chinese new years(2020-1-24) is comming, I have no more energy to adjust my yolo-v2 which means my model can't get the same result as origin yolo-v2 (76.8 mAP with 416 and 78.6 with 608). If you really care about this point, my project can't satisfy you. And there are many other excellent projects reproducing origin yolo-v2, so just consider to clone them and try on your task.

Before I tell you guys how to use this project, I must say something about difference between origin yolo-v2 and mine:

- For objectness, I just regard it as a 0-1 bernoulli distribution (Origin yolo-v2 set IoU as the objectness label.). If there is an object, the label is 1, else it is 0. And the model uses sigmoid function and MSE loss function.

- For class prediction, I use cross-entropy funtion while origin yolo-v2 used MSE to regression it. I really can't understand why it used MSE for class. If anyone knows that, please tell me. Thanks a lot!

- For data augmentation, I copy the augmentation codes from the https://github.com/amdegroot/ssd.pytorch which is a superb project reproducing the SSD. If anyone is interested in SSD, just clone it to learn !(Don't forget to star it !)

My yolo-v2 got 74.4 mAP with 416 input and 76.8 mAP with 608 input on VOC2007 test, lower than origin yolo-v2 that got 76.8% mAP with the same image size. This is maybe because that there are two tricks that didn't work:

1. hi-res classifier

The darknet19-448 has been trained and got 75.52 top-1 acc. But it doesn't bring any improvement for my model.

2. multi-scale train

It doesn't work, too. Although it does make my model robust to more input size (from 320 to 608), it didn't bring any increase on mAP with 416 resolution.

My loss function and groundtruth creator both in the ```tools.py```, and you can try to change some parameters to improve the model.


## Installation
- Pytorch-gpu 1.1.0
- Tensorboard 1.14.
- python-opencv, python3, Anaconda3-5.1.0

## Dataset
As for now, I only train and test on PASCAL VOC2007 and 2012. 

### VOC Dataset
I copy the download files from the following excellent project:
https://github.com/amdegroot/ssd.pytorch

I have uploaded the VOC2007 and VOC2012 to BaiDuYunDisk, so for researchers in China, you can download them from BaiDuYunDisk:

Link：https://pan.baidu.com/s/1tYPGCYGyC0wjpC97H-zzMQ 

Password：4la9

You will get a ```VOCdevkit.zip```, then what you need to do is just to unzip it and put it into ```data/```. After that, the whole path to VOC dataset is ```data/VOCdevkit/VOC2007``` and ```data/VOCdevkit/VOC2012```.

#### Download VOC2007 trainval & test

```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2007.sh # <directory>
```

#### Download VOC2012 trainval
```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2012.sh # <directory>
```

### MSCOCO Dataset
I copy the download files from the following excellent project:
https://github.com/DeNA/PyTorch_YOLOv3

#### Download MSCOCO 2017 dataset
Just run data/scripts/COCO2017.sh


## Train
To run:
```Shell
python train_voc.py
```

You can run ```python train_voc.py -h``` to check all optional argument

## Test
To run:
```Shell
python test_voc.py --trained_model [ Please input the path to model dir. ]
```

## Evaluation
To run:
```Shell
python eval_voc.py --train_model [ Please input the path to model dir. ]
```

Finally, I have tried to train my yolo-v2 on MSCOCO-2017 datatset, but I didn't get a good result. My yolo-v2 got 31.5 AP50 on MSCOCO-valid dataset and its visualization results are a little poor. I haven't address this problem , but what I know is that MSCOCO is really a challenging dataset !