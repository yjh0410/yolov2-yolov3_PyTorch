# the whole project
In this project, you can enjoy: 
- yolo-v2
- yolo-v3
- tiny-yolo-v2 
- tiny-yolo-v3

What I have to say is that I don't try to 100% reproduce the whole official YOLO project, because it is really hard to me. I have not much computation resource, so I can't train my yolov3 on COCO. It will cost more than two weeks...

Recently, I made some improvement, and my yolo project is very close to official yolo models.

I will upload the new model again. Just hold on~

However, I have a qeustion: Is the mAP metric really good? Does it really suit object detection?

I find higher mAP doesn't mean better visualization...so weird.


# YOLOv2
I really enjoy yolo. It is so amazing! So I try to reproduce it. And I think I achieve this goal:

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8> </th>     <td bgcolor=white> size </td><td bgcolor=white> Original (darknet) </td><td bgcolor=white> Ours (pytorch) 160peochs </td><td bgcolor=white> Ours (pytorch) 250epochs </td></tr>
<tr><th align="left" bgcolor=#f8f8f8> VOC07 test</th><td bgcolor=white> 416 </td><td bgcolor=white> 76.8 </td><td bgcolor=white> 76.0 </td><td bgcolor=white> 77.1 </td></tr>
<tr><th align="left" bgcolor=#f8f8f8> VOC07 test</th><td bgcolor=white> 544 </td><td bgcolor=white> 78.6 </td><td bgcolor=white> 77.0 </td><td bgcolor=white> 78.1 </td></tr>
</table></tbody>

With 160 training epochs, my yolo-v2 only gets 76.0 mAP with 416 input size and 77.0 mAP with 544 input size. To be better, I add another 90 epochs.
With 250 training epochs, my yolo-v2 performs very well !

During testing stage, I set conf thresh as 0.001 and set nms thresh as 0.5 to obtain above results. To make my model faster, I set conf thresh as 0.01. With this higher conf thresh, my yolo-v2 still performs very well and gets 77.0 mAP with 416 input size and 78.0 mAP with 544 input size.

I visualize some detection results whose score is over 0.3 on VOC 2007 test:
![Image](https://github.com/yjh0410/pytorch-yolo-v2-v3/blob/master/test_results/000000.jpg)
![Image](https://github.com/yjh0410/pytorch-yolo-v2-v3/blob/master/test_results/000003.jpg)
![Image](https://github.com/yjh0410/pytorch-yolo-v2-v3/blob/master/test_results/000006.jpg)
![Image](https://github.com/yjh0410/pytorch-yolo-v2-v3/blob/master/test_results/000029.jpg)
![Image](https://github.com/yjh0410/pytorch-yolo-v2-v3/blob/master/test_results/000030.jpg)
![Image](https://github.com/yjh0410/pytorch-yolo-v2-v3/blob/master/test_results/000039.jpg)
![Image](https://github.com/yjh0410/pytorch-yolo-v2-v3/blob/master/test_results/000065.jpg)
![Image](https://github.com/yjh0410/pytorch-yolo-v2-v3/blob/master/test_results/000070.jpg)
![Image](https://github.com/yjh0410/pytorch-yolo-v2-v3/blob/master/test_results/000072.jpg)

The COCO is coming ...
<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8> </th>     <td bgcolor=white> data </td><td bgcolor=white> AP </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white> AP_S </td><td bgcolor=white> AP_M </td><td bgcolor=white> AP_L </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> Original (darknet)</th><td bgcolor=white> COCO test-dev </td><td bgcolor=white> 21.6 </td><td bgcolor=white> 44.0 </td><td bgcolor=white> 19.2 </td><td bgcolor=white> 5.0 </td><td bgcolor=white> 22.4 </td><td bgcolor=white> 35.5 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> Ours (darknet)</th><td bgcolor=white> COCO eval-dev </td><td bgcolor=white> - </td><td bgcolor=white> - </td><td bgcolor=white> - </td><td bgcolor=white> - </td><td bgcolor=white> - </td><td bgcolor=white> - </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> Ours (darknet)</th><td bgcolor=white> COCO test-dev </td><td bgcolor=white> - </td><td bgcolor=white> - </td><td bgcolor=white> - </td><td bgcolor=white> - </td><td bgcolor=white> - </td><td bgcolor=white> - </td></tr>

</table></tbody>

## Tricks
Tricks in official paper:
- [x] batch norm
- [x] hi-res classifier
- [x] convolutional
- [x] anchor boxes
- [x] new network
- [x] dimension priors
- [x] location prediction
- [x] passthrough
- [x] multi-scale
- [x] hi-red detector

In TITAN Xp, my yolo-v2 runs at 100+ FPS, so it's very fast. I have no any TITAN X GPU, and I can't run my model in a X GPU. Sorry, guys~

Before I tell you how to use this project, I must say one important thing about difference between origin yolo-v2 and mine:

- For data augmentation, I copy the augmentation codes from the https://github.com/amdegroot/ssd.pytorch which is a superb project reproducing the SSD. If anyone is interested in SSD, just clone it to learn !(Don't forget to star it !)

So I don't write data augmentation by myself. I'm a little lazy~~

My loss function and groundtruth creator both in the ```tools.py```, and you can try to change any parameters to improve the model.

Next, I plan to train my yolo-v2 on COCO.

# YOLOv3
Besides YOLOv2, I also try to reproduce YOLOv3. Before this, I rebuilt a darknet53 network with PyTorch and pretrained it on ImageNet, so I don't select official darknet53 model file...Oh! I forgot to you guys that my darknet19 used in my YOLOv2 is also rebuilt by myself with PyTorch. The top-1 performance of my darknet19 and darknet53 is following:

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8> </th>     <td bgcolor=white> size </td><td bgcolor=white> Original (darknet) </td><td bgcolor=white> Ours (pytorch)  </td></tr>
<tr><th align="left" bgcolor=#f8f8f8> darknet19</th><td bgcolor=white> 224 </td><td bgcolor=white> 72.9 </td><td bgcolor=white> 72.96 </td></tr>
<tr><th align="left" bgcolor=#f8f8f8> darknet19</th><td bgcolor=white> 448 </td><td bgcolor=white> 76.5 </td><td bgcolor=white> 75.52 </td></tr>
<tr><th align="left" bgcolor=#f8f8f8> darknet53</th><td bgcolor=white> 224 </td><td bgcolor=white> 77.2 </td><td bgcolor=white> 75.42 </td></tr>
<tr><th align="left" bgcolor=#f8f8f8> darknet53</th><td bgcolor=white> 448 </td><td bgcolor=white> - </td><td bgcolor=white> 77.76 </td></tr>
</table></tbody>

Looks good !

I have only one GPU meaning training YOLOv3 on COCO will cost my lots of time(more than two weeks), so I only train my YOLOv3 on VOC. The resule is shown:

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8> </th>     <td bgcolor=white> size </td><td bgcolor=white> Original (darknet) </td><td bgcolor=white> Ours (pytorch) 250epochs </td></tr>
<tr><th align="left" bgcolor=#f8f8f8> VOC07 test</th><td bgcolor=white> 416 </td><td bgcolor=white> 80.25 </td><td bgcolor=white> 81.4 </td></tr>
</table></tbody>

I use the same training strategy to my YOLOv2. My data-processing code is a little different from official YOLOv3. For more details, you can check my code files.

# Tiny YOLOv2
Please hold on ...

# TIny YOLOv3
Please hold on ...

## Installation
- Pytorch-gpu 1.1.0/1.2.0/1.3.0
- Tensorboard 1.14.
- opencv-python, python3.6/3.7

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
Just run ```sh data/scripts/COCO2017.sh```. You will get COCO train2017, val2017, test2017.


## Train
### VOC
```Shell
python train_voc.py -v [select a model] -hr -ms --cuda
```
### COCO
```Shell
python train_coco.py -v [select a model] -hr -ms --cuda
```

You can run ```python train_voc.py -h``` to check all optional argument.

By default, I set num_workers in pytorch dataloader as 0 to guarantee my multi-scale trick. But the trick can't work when I add more wokers. I know little about multithreading. So sad...

## Test
### VOC
```Shell
python test_voc.py -v [select a model] --trained_model [ Please input the path to model dir. ] --cuda
```

### COCO
```Shell
python test_coco.py -v [select a model] --trained_model [ Please input the path to model dir. ] --cuda
```


## Evaluation
### VOC
```Shell
python eval_voc.py -v [select a model] --train_model [ Please input the path to model dir. ] --cuda
```

### COCO
To run on COCO_val:
```Shell
python eval_coco.py -v [select a model] --train_model [ Please input the path to model dir. ] --cuda
```

To run on COCO_test-dev(You must be sure that you have downloaded test2017):
```Shell
python eval_coco.py -v [select a model] --train_model [ Please input the path to model dir. ] --cuda -t
```
You will get a .json file which can be evaluated on COCO test server.