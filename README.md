# Update
Recently, I optimize my YOLO project:

https://github.com/yjh0410/PyTorch_YOLO-Family

In my new YOLO project, you can enjoy: 
- a new and stronger YOLOv1
- a new and stronger YOLOv2
- YOLOv3
- YOLOv3 with SPP
- YOLOv3 with DilatedEncoder
- YOLOv4
- YOLO-Tiny
- YOLO-Nano


# This project
In this project, you can enjoy: 
- YOLOv2 with DarkNet-19
- YOLOv2 with ResNet-50
- YOLOv2Slim
- YOLOv3
- YOLOv3-Spp
- YOLOv3-Tiny


I just want to provide a good YOLO project for everyone who is interested in Object Detection.

# Weights
Google Drive: https://drive.google.com/drive/folders/1T5hHyGICbFSdu6u2_vqvxn_puotvPsbd?usp=sharing 

BaiDuYunDisk: https://pan.baidu.com/s/1tSylvzOVFReUAvaAxKRSwg 
Password d266

You can download all my models from the above links.

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

## VOC2007

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8> </th>     <td bgcolor=white> size </td><td bgcolor=white> Original (darknet) </td><td bgcolor=white> Ours (pytorch) 160peochs </td><td bgcolor=white> Ours (pytorch) 250epochs </td></tr>
<tr><th align="left" bgcolor=#f8f8f8> VOC07 test</th><td bgcolor=white> 416 </td><td bgcolor=white> 76.8 </td><td bgcolor=white> 76.0 </td><td bgcolor=white> 77.1 </td></tr>
<tr><th align="left" bgcolor=#f8f8f8> VOC07 test</th><td bgcolor=white> 544 </td><td bgcolor=white> 78.6 </td><td bgcolor=white> 77.0 </td><td bgcolor=white> 78.1 </td></tr>
</table></tbody>

## COCO

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8> </th>     <td bgcolor=white> data </td><td bgcolor=white> AP </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white> AP_S </td><td bgcolor=white> AP_M </td><td bgcolor=white> AP_L </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> Original (darknet)</th><td bgcolor=white> COCO test-dev </td><td bgcolor=white> 21.6 </td><td bgcolor=white> 44.0 </td><td bgcolor=white> 19.2 </td><td bgcolor=white> 5.0 </td><td bgcolor=white> 22.4 </td><td bgcolor=white> 35.5 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> Ours (pytorch)</th><td bgcolor=white> COCO test-dev </td><td bgcolor=white> 26.8 </td><td bgcolor=white> 46.6 </td><td bgcolor=white> 26.8 </td><td bgcolor=white> 5.8 </td><td bgcolor=white> 27.4 </td><td bgcolor=white> 45.2 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> Ours (pytorch)</th><td bgcolor=white> COCO eval </td><td bgcolor=white> 26.6 </td><td bgcolor=white> 46.0 </td><td bgcolor=white> 26.7 </td><td bgcolor=white> 5.9 </td><td bgcolor=white> 27.8 </td><td bgcolor=white> 47.1 </td></tr>
</table></tbody>


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

## VOC2007

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8> </th>     <td bgcolor=white> size </td><td bgcolor=white> Original (darknet) </td><td bgcolor=white> Ours (pytorch) 250epochs </td></tr>
<tr><th align="left" bgcolor=#f8f8f8> VOC07 test</th><td bgcolor=white> 416 </td><td bgcolor=white> 80.25 </td><td bgcolor=white> 81.4 </td></tr>
</table></tbody>

# COCO

Official YOLOv3:

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8> </th>     <td bgcolor=white> data </td><td bgcolor=white> AP </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white> AP_S </td><td bgcolor=white> AP_M </td><td bgcolor=white> AP_L </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3-320</th><td bgcolor=white> COCO test-dev </td><td bgcolor=white> 28.2 </td><td bgcolor=white> 51.5 </td><td bgcolor=white> - </td><td bgcolor=white> - </td><td bgcolor=white> - </td><td bgcolor=white> - </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3-416</th><td bgcolor=white> COCO test-dev </td><td bgcolor=white> 31.0 </td><td bgcolor=white> 55.3 </td><td bgcolor=white> - </td><td bgcolor=white> - </td><td bgcolor=white> - </td><td bgcolor=white> - </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3-608</th><td bgcolor=white> COCO test-dev </td><td bgcolor=white> 33.0 </td><td bgcolor=white> 57.0 </td><td bgcolor=white> 34.4 </td><td bgcolor=white> 18.3 </td><td bgcolor=white> 35.4 </td><td bgcolor=white> 41.9 </td></tr>
</table></tbody>

Our YOLOv3:

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8> </th>     <td bgcolor=white> data </td><td bgcolor=white> AP </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white> AP_S </td><td bgcolor=white> AP_M </td><td bgcolor=white> AP_L </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3-320</th><td bgcolor=white> COCO test-dev </td><td bgcolor=white> 33.1 </td><td bgcolor=white> 54.1 </td><td bgcolor=white> 34.5 </td><td bgcolor=white> 12.1 </td><td bgcolor=white> 34.5 </td><td bgcolor=white> 49.6 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3-416</th><td bgcolor=white> COCO test-dev </td><td bgcolor=white> 36.0 </td><td bgcolor=white> 57.4 </td><td bgcolor=white> 37.0 </td><td bgcolor=white> 16.3 </td><td bgcolor=white> 37.5 </td><td bgcolor=white> 51.1 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3-608</th><td bgcolor=white> COCO test-dev </td><td bgcolor=white> 37.6 </td><td bgcolor=white> 59.4 </td><td bgcolor=white> 39.9 </td><td bgcolor=white> 20.4 </td><td bgcolor=white> 39.9 </td><td bgcolor=white> 48.2 </td></tr>
</table></tbody>

# YOLOv3SPP
## COCO:

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8> </th>     <td bgcolor=white> data </td><td bgcolor=white> AP </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white> AP_S </td><td bgcolor=white> AP_M </td><td bgcolor=white> AP_L </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3Spp-320</th><td bgcolor=white> COCO eval </td><td bgcolor=white> 32.78 </td><td bgcolor=white> 53.79 </td><td bgcolor=white> 33.9 </td><td bgcolor=white> 12.4 </td><td bgcolor=white> 35.5 </td><td bgcolor=white> 50.6 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3Spp-416</th><td bgcolor=white> COCO eval </td><td bgcolor=white> 35.66 </td><td bgcolor=white> 57.09 </td><td bgcolor=white> 37.4 </td><td bgcolor=white> 16.8 </td><td bgcolor=white> 38.1 </td><td bgcolor=white> 50.7 </td></tr>


<tr><th align="left" bgcolor=#f8f8f8> YOLOv3Spp-608</th><td bgcolor=white> COCO eval </td><td bgcolor=white> 37.52 </td><td bgcolor=white> 59.44 </td><td bgcolor=white> 39.3 </td><td bgcolor=white> 21.5 </td><td bgcolor=white> 40.6 </td><td bgcolor=white> 49.6 </td></tr>

</table></tbody>

# YOLOv3Tiny
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
