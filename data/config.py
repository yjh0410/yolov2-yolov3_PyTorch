# config.py

# YOLOv2 with darknet-19
yolov2_d19_cfg = {
    # network
    'backbone': 'd19',
    # for multi-scale trick
    'train_size': 640,
    'val_size': 416,
    'random_size_range': [10, 19],
    # anchor size
    'anchor_size_voc': [[1.19, 1.98], [2.79, 4.59], [4.53, 8.92], [8.06, 5.29], [10.32, 10.65]],
    'anchor_size_coco': [[0.53, 0.79], [1.71, 2.36], [2.89, 6.44], [6.33, 3.79], [9.03, 9.74]],
    # train
    'lr_epoch': (150, 200),
    'max_epoch': 250,
    'ignore_thresh': 0.5
}

# YOLOv2 with resnet-50
yolov2_r50_cfg = {
    # network
    'backbone': 'r50',
    # for multi-scale trick
    'train_size': 640,
    'val_size': 416,
    'random_size_range': [10, 19],
    # anchor size
    'anchor_size_voc': [[1.19, 1.98], [2.79, 4.59], [4.53, 8.92], [8.06, 5.29], [10.32, 10.65]],
    'anchor_size_coco': [[0.53, 0.79], [1.71, 2.36], [2.89, 6.44], [6.33, 3.79], [9.03, 9.74]],
    # train
    'lr_epoch': (150, 200),
    'max_epoch': 250,
    'ignore_thresh': 0.5
}

# YOLOv2Slim
yolov2_slim_cfg = {
    # network
    'backbone': 'dtiny',
    # for multi-scale trick
    'train_size': 640,
    'val_size': 416,
    'random_size_range': [10, 19],
    # anchor size
    'anchor_size_voc': [[1.19, 1.98], [2.79, 4.59], [4.53, 8.92], [8.06, 5.29], [10.32, 10.65]],
    'anchor_size_coco': [[0.53, 0.79], [1.71, 2.36], [2.89, 6.44], [6.33, 3.79], [9.03, 9.74]],
    # train
    'lr_epoch': (90, 120),
    'max_epoch': 150,
    'ignore_thresh': 0.5
}

# YOLOv3 / YOLOv3Spp
yolov3_d53_cfg = {
    # network
    'backbone': 'd53',
    # for multi-scale trick
    'train_size': 640,
    'val_size': 416,
    'random_size_range': [10, 19],
    # anchor size
    'anchor_size_voc': [[32.64, 47.68], [50.24, 108.16], [126.72, 96.32],     
                        [78.4, 201.92], [178.24, 178.56], [129.6, 294.72],     
                        [331.84, 194.56], [227.84, 325.76], [365.44, 358.72]],
    'anchor_size_coco': [[12.48, 19.2], [31.36, 46.4],[46.4, 113.92],
                         [97.28, 55.04], [133.12, 127.36], [79.04, 224.],
                         [301.12, 150.4 ], [172.16, 285.76], [348.16, 341.12]],
    # train
    'lr_epoch': (150, 200),
    'max_epoch': 250,
    'ignore_thresh': 0.5
}

# YOLOv3Tiny
yolov3_tiny_cfg = {
    # network
    'backbone': 'd-light',
    # for multi-scale trick
    'train_size': 640,
    'val_size': 416,
    'random_size_range':[10, 19],
    # anchor size
    'anchor_size_voc': [[34.01, 61.79],   [86.94, 109.68],  [93.49, 227.46],     
                        [246.38, 163.33], [178.68, 306.55], [344.89, 337.14]],
    'anchor_size_coco': [[15.09, 23.25],  [46.36, 61.47],   [68.41, 161.84],
                         [168.88, 93.59], [154.96, 257.45], [334.74, 302.47]],
    # train
    'lr_epoch': (150, 200),
    'max_epoch': 250,
    'ignore_thresh': 0.5
}
