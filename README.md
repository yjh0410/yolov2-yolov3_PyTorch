# pytorch-yolo-v2
I really enjoy yolo. It is so amazing! So I try to reproduce it. And I almost got it!

But !! I don't plan to one hundred percent reproduce it, so my own yolo-v2 is a little different with origin version. Because all the tricks used in yolo-v2 may not ne suitable for other tasks. 

It is known to us that reproduce the model is easy while it is hard to reproduce the results shown in the paper.

As the Chinese new years(2020-1-24) is comming, I have no more energy to adjust my yolo-v2 which means my model can't get the same result as origin yolo-v2 (76.8 mAP with 416 and 78.6 with 608). If you really care about this point, my project can't satisfy you. And there are many other excellent projects reproducing origin yolo-v2, so just consider to clone them and try on your task.

Before I tell you guys how to use this project, I must say something about difference between origin yolo-v2 and mine:

- For objectness, I just regard it as a 0-1 bernoulli distribution (Origin yolo-v2 set IoU as the objectness label.). If there is an object, the label is 1, else it is 0. And the model uses sigmoid function and BCE loss function.

- For class prediction, I use cross-entropy funtion while origin yolo-v2 used MSE to regression it. I really can't understand why it used MSE for class. If anyone knows that, please tell me. Thanks a lot!

- For data augmentation, I copy the augmentation codes from the https://github.com/amdegroot/ssd.pytorch which is a superb project reproducing the SSD. If anyone is interested in SSD, just clone it to learn !(Don't forget to star it !)

My yolo-v2 got 73.0 mAP on VOC2007 test with input resolution 416 image size, lower than origin yolo-v2 that got 76.8% mAP with the same image size. This is maybe because that there are two tricks I didn't use:

1. hi-res classifier

The darknet19-448 has been trained and got 75.52 top-1 acc. But it doesn't bring any improvement to my model.

2. multi-scale train

It doesn't work, too. Although it does make my model robust to more input size (from 320 to 608), it didn't bring any increase on mAP with 416 resolution.

## Installation
- Pytorch-gpu 1.1.0
- Tensorboard 1.14.
- python-opencv, python3, Anaconda3-5.1.0

## Dataset
As for now, I only train and test on PASCAL VOC2007 and 2012. 

### VOC Dataset
I copy the download files from the following excellent project:
https://github.com/amdegroot/ssd.pytorch

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

## Train
To run:
```Shell
python train_voc.py
```

Optional argument:
```Shell
-h, --help            show this help message and exit
  -v VERSION, --version VERSION
                        yolo_v2
  -d DATASET, --dataset DATASET
                        VOC or COCO dataset
  -hr HIGH_RESOLUTION, --high_resolution HIGH_RESOLUTION
                        1: use high resolution to pretrain; 0: else not.
  -ms MULTI_SCALE, --multi_scale MULTI_SCALE
                        1: use multi-scale trick; 0: else not
  -fl USE_FOCAL, --use_focal USE_FOCAL
                        0: use focal loss; 1: else not;
  --batch_size BATCH_SIZE
                        Batch size for training
  --lr LR               initial learning rate
  -wp WARM_UP, --warm_up WARM_UP
                        yes or no to choose using warmup strategy to train
  --wp_epoch WP_EPOCH   The upper bound of warm-up
  --dataset_root DATASET_ROOT
                        Location of VOC root directory
  --num_classes NUM_CLASSES
                        The number of dataset classes
  --momentum MOMENTUM   Momentum value for optim
  --weight_decay WEIGHT_DECAY
                        Weight decay for SGD
  --gamma GAMMA         Gamma update for SGD
  --num_workers NUM_WORKERS
                        Number of workers used in dataloading
  --gpu_ind GPU_IND     To choose your gpu.
  --save_folder SAVE_FOLDER
                        Gamma update for SGD
```

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