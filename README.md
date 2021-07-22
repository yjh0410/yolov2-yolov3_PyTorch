# Update
I'm trying to write a tutorial to guide users to train, test and eval their own dataset.

I wish this version of yolo can help everyone who is interested in Object Detection.

# This project
In this project, you can enjoy: 
- YOLOv2 with DarkNet-19
- YOLOv2 with ResNet-50
- YOLOv2Slim
- YOLOv3
- YOLOv3Spp
- YOLOv3Tiny


I just want to provide a good YOLO project for everyone who is interested in Object Detection.

# Weights
Google Drive: https://drive.google.com/drive/folders/1T5hHyGICbFSdu6u2_vqvxn_puotvPsbd?usp=sharing 

BaiDuYunDisk: https://pan.baidu.com/s/1tSylvzOVFReUAvaAxKRSwg 
Password d266

You can download all my models from the above links.

# DarkNet-19, DarkNet-53, CSPDarkNet-53
First of all, I reproduce the backbones used in YOLO.

## DarkNet-19

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8> </th>     <td bgcolor=white> size </td><td bgcolor=white> top1  </td></tr>
<tr><th align="left" bgcolor=#f8f8f8> darknet19</th><td bgcolor=white> 224 </td><td bgcolor=white> 72.96 </td></tr>
<tr><th align="left" bgcolor=#f8f8f8> darknet19</th><td bgcolor=white> 448 </td><td bgcolor=white> 75.52 </td></tr>
</table></tbody>

## DarkNet-53

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8> </th>     <td bgcolor=white> size </td><td bgcolor=white> top1  </td></tr>
<tr><th align="left" bgcolor=#f8f8f8> darknet53</th><td bgcolor=white> 224 </td><td bgcolor=white> 75.42 </td></tr>
<tr><th align="left" bgcolor=#f8f8f8> darknet53</th><td bgcolor=white> 448 </td><td bgcolor=white> 77.76 </td></tr>
</table></tbody>

## CSPDarkNet-53

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8> </th>     <td bgcolor=white> size </td><td bgcolor=white> top1 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> cspdarknet53</th><td bgcolor=white> 224 </td><td bgcolor=white> 75.7 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> cspdarknet53-448 </th><td bgcolor=white> 448 </td><td bgcolor=white> 76.9 </td></tr>
</table></tbody>

# YOLOv2

## YOLOv2 with DarkNet-19
### Tricks
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


Then I train my YOLOv2 on VOC dataset. I really enjoy yolo. It is so amazing! I think I have achieved this goal:

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

COCO:

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8> </th>     <td bgcolor=white> data </td><td bgcolor=white> AP </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white> AP_S </td><td bgcolor=white> AP_M </td><td bgcolor=white> AP_L </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> Original (darknet)</th><td bgcolor=white> COCO test-dev </td><td bgcolor=white> 21.6 </td><td bgcolor=white> 44.0 </td><td bgcolor=white> 19.2 </td><td bgcolor=white> 5.0 </td><td bgcolor=white> 22.4 </td><td bgcolor=white> 35.5 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> Ours (pytorch)</th><td bgcolor=white> COCO test-dev </td><td bgcolor=white> 26.8 </td><td bgcolor=white> 46.6 </td><td bgcolor=white> 26.8 </td><td bgcolor=white> 5.8 </td><td bgcolor=white> 27.4 </td><td bgcolor=white> 45.2 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> Ours (pytorch)</th><td bgcolor=white> COCO eval </td><td bgcolor=white> 26.6 </td><td bgcolor=white> 46.0 </td><td bgcolor=white> 26.7 </td><td bgcolor=white> 5.9 </td><td bgcolor=white> 27.8 </td><td bgcolor=white> 47.1 </td></tr>
</table></tbody>

In TITAN Xp, my yolo-v2 runs at 100+ FPS, so it's very fast. I have no any TITAN X GPU, and I can't run my model in a X GPU. Sorry, guys~

Before I tell you how to use this project, I must say one important thing about difference between origin yolo-v2 and mine:

- For data augmentation, I copy the augmentation codes from the https://github.com/amdegroot/ssd.pytorch which is a superb project reproducing the SSD. If anyone is interested in SSD, just clone it to learn !(Don't forget to star it !)

So I don't write data augmentation by myself. I'm a little lazy~~

My loss function and groundtruth creator both in the ```tools.py```, and you can try to change any parameters to improve the model.

## YOLOv2 with ResNet-50

I replace darknet-19 with resnet-50 and get a better result on COCO-val

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8> </th>     <td bgcolor=white> data </td><td bgcolor=white> AP </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white> AP_S </td><td bgcolor=white> AP_M </td><td bgcolor=white> AP_L </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> Our YOLOv2-320</th><td bgcolor=white> COCO eval </td><td bgcolor=white> 25.8 </td><td bgcolor=white> 44.6 </td><td bgcolor=white> 25.9 </td><td bgcolor=white> 4.6 </td><td bgcolor=white> 26.8 </td><td bgcolor=white> 47.9 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> Our YOLOv2-416</th><td bgcolor=white> COCO eval </td><td bgcolor=white> 29.0 </td><td bgcolor=white> 48.8 </td><td bgcolor=white> 29.7 </td><td bgcolor=white> 7.4 </td><td bgcolor=white> 31.9 </td><td bgcolor=white> 48.3 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> Our YOLOv2-512</th><td bgcolor=white> COCO eval </td><td bgcolor=white> 30.4 </td><td bgcolor=white> 51.6 </td><td bgcolor=white> 30.9 </td><td bgcolor=white> 10.1 </td><td bgcolor=white> 34.9 </td><td bgcolor=white> 46.6 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> Our YOLOv2-544</th><td bgcolor=white> COCO eval </td><td bgcolor=white> 30.4 </td><td bgcolor=white> 51.9 </td><td bgcolor=white> 30.9 </td><td bgcolor=white> 11.1 </td><td bgcolor=white> 35.8 </td><td bgcolor=white> 45.5 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> Our YOLOv2-608</th><td bgcolor=white> COCO eval </td><td bgcolor=white> 29.2 </td><td bgcolor=white> 51.6 </td><td bgcolor=white> 29.1 </td><td bgcolor=white> 13.6 </td><td bgcolor=white> 36.8 </td><td bgcolor=white> 40.5 </td></tr>
</table></tbody>

# YOLOv3

I train my YOLOv3 on VOC:

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8> </th>     <td bgcolor=white> size </td><td bgcolor=white> Original (darknet) </td><td bgcolor=white> Ours (pytorch) 250epochs </td></tr>
<tr><th align="left" bgcolor=#f8f8f8> VOC07 test</th><td bgcolor=white> 416 </td><td bgcolor=white> 80.25 </td><td bgcolor=white> 81.4 </td></tr>
</table></tbody>

I use the same training strategy to my YOLOv2. My data-processing code is a little different from official YOLOv3. For more details, you can check my code files.

COCO:

Original YOLOv3:

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8> </th>     <td bgcolor=white> data </td><td bgcolor=white> AP </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white> AP_S </td><td bgcolor=white> AP_M </td><td bgcolor=white> AP_L </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3-320</th><td bgcolor=white> COCO test-dev </td><td bgcolor=white> 28.2 </td><td bgcolor=white> 51.5 </td><td bgcolor=white> - </td><td bgcolor=white> - </td><td bgcolor=white> - </td><td bgcolor=white> - </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3-416</th><td bgcolor=white> COCO test-dev </td><td bgcolor=white> 31.0 </td><td bgcolor=white> 55.3 </td><td bgcolor=white> - </td><td bgcolor=white> - </td><td bgcolor=white> - </td><td bgcolor=white> - </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3-608</th><td bgcolor=white> COCO test-dev </td><td bgcolor=white> 33.0 </td><td bgcolor=white> 57.0 </td><td bgcolor=white> 34.4 </td><td bgcolor=white> 18.3 </td><td bgcolor=white> 35.4 </td><td bgcolor=white> 41.9 </td></tr>
</table></tbody>

Our YOLOv3_PyTorch:

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8> </th>     <td bgcolor=white> data </td><td bgcolor=white> AP </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white> AP_S </td><td bgcolor=white> AP_M </td><td bgcolor=white> AP_L </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3-320</th><td bgcolor=white> COCO test-dev </td><td bgcolor=white> 33.1 </td><td bgcolor=white> 54.1 </td><td bgcolor=white> 34.5 </td><td bgcolor=white> 12.1 </td><td bgcolor=white> 34.5 </td><td bgcolor=white> 49.6 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3-416</th><td bgcolor=white> COCO test-dev </td><td bgcolor=white> 36.0 </td><td bgcolor=white> 57.4 </td><td bgcolor=white> 37.0 </td><td bgcolor=white> 16.3 </td><td bgcolor=white> 37.5 </td><td bgcolor=white> 51.1 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3-608</th><td bgcolor=white> COCO test-dev </td><td bgcolor=white> 37.6 </td><td bgcolor=white> 59.4 </td><td bgcolor=white> 39.9 </td><td bgcolor=white> 20.4 </td><td bgcolor=white> 39.9 </td><td bgcolor=white> 48.2 </td></tr>
</table></tbody>

My YOLOv3 is very stronger and better, right?

I also visualize some detection results whose score is over 0.3 on COCO 2017-val:

![Image](https://github.com/yjh0410/pytorch-yolo-v2-v3/blob/master/test_results/COCO-val/000003.jpg)
![Image](https://github.com/yjh0410/pytorch-yolo-v2-v3/blob/master/test_results/COCO-val/000077.jpg)
![Image](https://github.com/yjh0410/pytorch-yolo-v2-v3/blob/master/test_results/COCO-val/003422.jpg)
![Image](https://github.com/yjh0410/pytorch-yolo-v2-v3/blob/master/test_results/COCO-val/003853.jpg)
![Image](https://github.com/yjh0410/pytorch-yolo-v2-v3/blob/master/test_results/COCO-val/003970.jpg)
![Image](https://github.com/yjh0410/pytorch-yolo-v2-v3/blob/master/test_results/COCO-val/004040.jpg)
![Image](https://github.com/yjh0410/pytorch-yolo-v2-v3/blob/master/test_results/COCO-val/004157.jpg)
![Image](https://github.com/yjh0410/pytorch-yolo-v2-v3/blob/master/test_results/COCO-val/004283.jpg)
![Image](https://github.com/yjh0410/pytorch-yolo-v2-v3/blob/master/test_results/COCO-val/004862.jpg)
![Image](https://github.com/yjh0410/pytorch-yolo-v2-v3/blob/master/test_results/COCO-val/004985.jpg)
![Image](https://github.com/yjh0410/pytorch-yolo-v2-v3/blob/master/test_results/COCO-val/004988.jpg)

So, just have fun !

# YOLOv3SPP
On COCO:

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8> </th>     <td bgcolor=white> data </td><td bgcolor=white> AP </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white> AP_S </td><td bgcolor=white> AP_M </td><td bgcolor=white> AP_L </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3Spp-320</th><td bgcolor=white> COCO eval </td><td bgcolor=white> 32.78 </td><td bgcolor=white> 53.79 </td><td bgcolor=white> 33.9 </td><td bgcolor=white> 12.4 </td><td bgcolor=white> 35.5 </td><td bgcolor=white> 50.6 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3Spp-416</th><td bgcolor=white> COCO eval </td><td bgcolor=white> 35.66 </td><td bgcolor=white> 57.09 </td><td bgcolor=white> 37.4 </td><td bgcolor=white> 16.8 </td><td bgcolor=white> 38.1 </td><td bgcolor=white> 50.7 </td></tr>


<tr><th align="left" bgcolor=#f8f8f8> YOLOv3Spp-608</th><td bgcolor=white> COCO eval </td><td bgcolor=white> 37.52 </td><td bgcolor=white> 59.44 </td><td bgcolor=white> 39.3 </td><td bgcolor=white> 21.5 </td><td bgcolor=white> 40.6 </td><td bgcolor=white> 49.6 </td></tr>

</table></tbody>

My YOLOv3SPP is not better than my YOLOv3. It is a little confusing ...

# YOLOv2Slim
I build a very simple lightweight backbone: darknet_tiny

![Image](https://github.com/yjh0410/pytorch-yolo-v2-v3/blob/master/img_file/darknet_tiny.png)

I replace the darknet19 used in YOLOv2 with darknet_tiny.

My SlimYOLOv2 is fast and strong. On VOC, it gets 70.7 mAP and 100+ FPS on 1660ti GPU.

Just enjoy it.

And, I'm still trying to make it faster without too much drop of precision.

# YOLOv3Tiny
We evaluate our YOLOv3Tiny on COCO-val with inputsize 608:

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8> </th>     <td bgcolor=white> data </td><td bgcolor=white> AP </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white> AP_S </td><td bgcolor=white> AP_M </td><td bgcolor=white> AP_L </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> (official) YOLOv3Tiny </th><td bgcolor=white> COCO test-dev </td><td bgcolor=white> - </td><td bgcolor=white> 33.1 </td><td bgcolor=white> - </td><td bgcolor=white>- </td><td bgcolor=white> - </td><td bgcolor=white> - </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> (Our) YOLOv3Tiny </th><td bgcolor=white> COCO val </td><td bgcolor=white> 15.9 </td><td bgcolor=white> 33.8 </td><td bgcolor=white> 12.8 </td><td bgcolor=white> 7.6 </td><td bgcolor=white> 17.7 </td><td bgcolor=white> 22.4 </td></tr>

</table></tbody>


# Installation
- Pytorch-gpu 1.1.0/1.2.0/1.3.0
- Tensorboard 1.14.
- opencv-python, python3.6/3.7

# Dataset

## VOC Dataset
I copy the download files from the following excellent project:
https://github.com/amdegroot/ssd.pytorch

I have uploaded the VOC2007 and VOC2012 to BaiDuYunDisk, so for researchers in China, you can download them from BaiDuYunDisk:

Link：https://pan.baidu.com/s/1tYPGCYGyC0wjpC97H-zzMQ 

Password：4la9

You will get a ```VOCdevkit.zip```, then what you need to do is just to unzip it and put it into ```data/```. After that, the whole path to VOC dataset is ```data/VOCdevkit/VOC2007``` and ```data/VOCdevkit/VOC2012```.

### Download VOC2007 trainval & test

```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2007.sh # <directory>
```

### Download VOC2012 trainval
```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2012.sh # <directory>
```

## MSCOCO Dataset
I copy the download files from the following excellent project:
https://github.com/DeNA/PyTorch_YOLOv3

### Download MSCOCO 2017 dataset
Just run ```sh data/scripts/COCO2017.sh```. You will get COCO train2017, val2017, test2017.


# Train
## VOC
```Shell
python train.py -d voc --cuda -v [select a model] -hr -ms --ema
```

You can run ```python train.py -h``` to check all optional argument.

## COCO
If you have only one gpu:
```Shell
python train.py -d coco --cuda -v [select a model] -hr -ms --ema
```

If you have multi gpus like 8, and you put 4 images on each gpu:
```Shell
python -m torch.distributed.launch --nproc_per_node=8 train.py -d coco --cuda -v [select a model] -hr -ms --ema \
                                                                        -dist \
                                                                        --sybn \
                                                                        --num_gpu 8\
                                                                        --batch_size 4
```

# Test
## VOC
```Shell
python test.py -d voc --cuda -v [select a model] --trained_model [ Please input the path to model dir. ]
```

## COCO
```Shell
python test.py -d coco-val --cuda -v [select a model] --trained_model [ Please input the path to model dir. ]
```


# Evaluation
## VOC
```Shell
python eval.py -d voc --cuda -v [select a model] --train_model [ Please input the path to model dir. ]
```

## COCO
To run on COCO_val:
```Shell
python eval.py -d coco-val --cuda -v [select a model] --train_model [ Please input the path to model dir. ]
```

To run on COCO_test-dev(You must be sure that you have downloaded test2017):
```Shell
python eval.py -d coco-test --cuda -v [select a model] --train_model [ Please input the path to model dir. ]
```
You will get a .json file which can be evaluated on COCO test server.
