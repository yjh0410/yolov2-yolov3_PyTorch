# the whole project
In this project, you can enjoy yolo-v2, yolo-v3, tiny-yolo-v2 and tiny-yolo-v3. What I have to say is that I don't try to 100% reproduce official YOLO, because it is really difficult and I have not much computation resource. 

Recently, I made some improvement, and my yolo project is very close to official yolo models.

For now, my yolo-v2 gets 76.0 mAP with 416 input size. I set batchsize as 20, and I train yolo-v2 with 160 epoch same to official yolo-v2.

To get a higher mAP, I add more epochs(total 250 epochs) to train my model. The model is being trained. And my yolo-v3, too.

I will upload the new model again. Just hold on~

However, I have a qeustion: Is the mAP metric really good? Does it really suit object detection?

I find higher mAP doesn't mean better visualization...so weird.

# pytorch-yolo-v3
Good news !!!

In this update, I add yolo-v3 model which gets 81.0 mAP ( with 416 input, and no multi-scale training trick ) on VOC2007-test. For more details, you could see code files including ```models/yolo_v3.py``` and ```tools.py```.

In addition, you can replace darknet-53 with darknet-19 as the backbone of yolo-v3, and this model can get 78.6 mAP on VOC2007-test.

# pytorch-yolo-v2
I really enjoy yolo. It is so amazing! So I try to reproduce it. And I almost got it!

But !! I don't plan to one hundred percent reproduce it, so my own yolo-v2 is a little different with origin version. Because all the tricks used in yolo-v2 may not ne suitable for other tasks. 

It is known to us that reproducing the model is easy while it is hard to reproduce the results shown in the paper.

Before I tell you guys how to use this project, I must say something about difference between origin yolo-v2 and mine:

- For class prediction, I use cross-entropy funtion while origin yolo-v2 used MSE to regression it. I really can't understand why it used MSE for class. If anyone knows that, please tell me. Thanks a lot!

- For data augmentation, I copy the augmentation codes from the https://github.com/amdegroot/ssd.pytorch which is a superb project reproducing the SSD. If anyone is interested in SSD, just clone it to learn !(Don't forget to star it !)

My yolo-v2 got 76.0 mAP with 416 input size, lower than origin yolo-v2 that got 76.8% mAP with the same image size. Otherwise, I find the multi-scale trick doesn't always work...

My loss function and groundtruth creator both in the ```tools.py```, and you can try to change some parameters to improve the model.


## Installation
- Pytorch-gpu 1.1.0/1.2.0/1.3.0
- Tensorboard 1.14.
- python-opencv, python3.6/3.7

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
