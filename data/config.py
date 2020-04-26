# config.py
import os.path

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))
MEANS = (104, 117, 123)
# ANCHOR SIZE is from the anchor config of YOLO-v2. But, following is searched by myself!!!
ANCHOR_SIZE = [[1.19, 1.98], [2.79, 4.59], [4.53, 8.92], [8.06, 5.29], [10.32, 10.65]] 
      
ANCHOR_SIZE_COCO = [[0.53, 0.79], [1.71, 2.36], [2.89, 6.44], [6.33, 3.79], [9.03, 9.74]]

# multi level anchor box config for VOC and COCO
MULTI_ANCHOR_SIZE = [[32.64, 47.68], [50.24, 108.16], [126.72, 96.32],     
                     [78.4, 201.92], [178.24, 178.56], [129.6, 294.72],     
                     [331.84, 194.56], [227.84, 325.76], [365.44, 358.72]]   

MULTI_ANCHOR_SIZE_COCO = [[12.48, 19.2], [31.36, 46.4],[46.4, 113.92],
                          [97.28, 55.04], [133.12, 127.36], [79.04, 224.],
                          [301.12, 150.4 ], [172.16, 285.76], [348.16, 341.12]]

IGNORE_THRESH = 0.5


# yolo-v2 config
# I hope you know that higher mAP, not always better results.
voc_ab = {
    'num_classes': 20,
    'lr_epoch': (80, 120, 160), # set (150, 200, 250), if you want to get a higher mAP
    'max_epoch': 160, # set 250, if you want to get a higher mAP
    'min_dim': [416, 416],
    'ms_channels':[128, 256, 512],
    'stride': 32,
    'strides': [16, 32],
    'multi_scale': [[320, 320], [352, 352], [384, 384], [416, 416], [448, 448],
                 [480, 480], [512, 512], [544, 544], [576, 576], [608, 608]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

coco_ab = {
    'num_classes': 80,
    'lr_epoch': (80, 120, 160), # set (150, 200, 250), if you want to get a higher mAP
    'max_epoch': 160, # set 250, if you want to get a higher mAP
    'min_dim': [416, 416],
    'ms_channels':[128, 256, 512],
    'stride': 32,
    'strides': [16, 32],
    'multi_scale': [[320, 320], [352, 352], [384, 384], [416, 416], [448, 448],
                 [480, 480], [512, 512], [544, 544], [576, 576], [608, 608]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'COCO',
}
