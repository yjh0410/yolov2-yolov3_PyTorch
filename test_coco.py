import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from data.cocodataset import *
from data import config, BaseTransform, VOCAnnotationTransform, VOCDetection, VOC_ROOT, VOC_CLASSES
import numpy as np
import cv2
import time
from decimal import *


parser = argparse.ArgumentParser(description='YOLO-v2 Detection')
parser.add_argument('-v', '--version', default='yolo_v2',
                    help='yolo_v2, yolo_v3, tiny_yolo_v2, tiny_yolo_v3')
parser.add_argument('-d', '--dataset', default='COCO',
                    help='we use VOC-test or COCO-val to test.')
parser.add_argument('--trained_model', default='weights_yolo_v2/yolo_v2_72.2.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--visual_threshold', default=0.3, type=float,
                    help='Final confidence threshold')
parser.add_argument('--dataset_root', default='./data/COCO/', 
                    help='Location of VOC root directory')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='use cuda.')
parser.add_argument('-f', default=None, type=str, 
                    help="Dummy arg so we can load in Jupyter Notebooks")
parser.add_argument('--debug', action='store_true', default=False,
                    help='debug mode where only one image is trained')


args = parser.parse_args()

coco_class_labels = ('background',
                        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                        'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign',
                        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                        'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella',
                        'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
                        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass',
                        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                        'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk',
                        'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book',
                        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

coco_class_index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
                    21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67,
                    70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

def test_net(net, device, testset, transform, thresh, mode='voc'):
    class_color = [(np.random.randint(255),np.random.randint(255),np.random.randint(255)) for _ in range(80)]
    num_images = len(testset)
    for index in range(num_images):
        print('Testing image {:d}/{:d}....'.format(index+1, num_images))
        if args.version == 'COCO':
            img, _ = testset.pull_image(index)
        elif args.version == 'VOC':
            img = testset.pull_image(index)
        # img_id, annotation = testset.pull_anno(i)
        x = torch.from_numpy(transform(img)[0][:, :, (2, 1, 0)]).permute(2, 0, 1)
        x = x.unsqueeze(0).to(device)

        t0 = time.clock()
        y = net(x)      # forward pass
        detections = y
        print("detection time used ", Decimal(time.clock()) - Decimal(t0), "s")
        # scale each detection back up to the image
        scale = np.array([[img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]]])
        bbox_pred, scores, cls_inds = detections
        # map the boxes to origin image scale
        bbox_pred *= scale

        for i, box in enumerate(bbox_pred):
            cls_indx = cls_inds[i]
            xmin, ymin, xmax, ymax = box
            if scores[i] > thresh:
                box_w = int(xmax - xmin)
                cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), class_color[int(cls_indx)], 2)
                cv2.rectangle(img, (int(xmin), int(abs(ymin)-15)), (int(xmin+box_w*0.55), int(ymin)), class_color[int(cls_indx)], -1)
                cls_id = coco_class_index[int(cls_indx)]
                cls_name = coco_class_labels[cls_id]
                mess = '%s: %.3f' % (cls_name, scores[i])
                cv2.putText(img, mess, (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
        cv2.imshow('detection', img)
        cv2.waitKey(0)
        # print('Saving the' + str(index) + '-th image ...')
        # cv2.imwrite('test_images/' + args.dataset+ '3/' + str(index).zfill(6) +'.jpg', img)



def test():
    # get device
    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # load net
    num_classes = 80
    if args.dataset == 'COCO':
        cfg = config.coco_ab
        testset = COCODataset(
                    data_dir=args.dataset_root,
                    json_file='instances_val2017.json',
                    name='val2017',
                    img_size=cfg['min_dim'][0],
                    debug=args.debug)
    elif args.dataset == 'VOC':
        cfg = config.voc_ab
        testset = VOCDetection(VOC_ROOT, [('2007', 'test')], None, VOCAnnotationTransform())


    if args.version == 'yolo_v2':
        from models.yolo_v2 import myYOLOv2
        net = myYOLOv2(device, input_size=cfg['min_dim'], num_classes=num_classes, anchor_size=config.ANCHOR_SIZE_COCO)
        print('Let us test yolo-v2 on the MSCOCO dataset ......')
    
    elif args.version == 'yolo_v3':
        from models.yolo_v3 import myYOLOv3
        net = myYOLOv3(device, input_size=cfg['min_dim'], num_classes=num_classes, anchor_size=config.MULTI_ANCHOR_SIZE_COCO)

    elif args.version == 'tiny_yolo_v2':
        from models.tiny_yolo_v2 import YOLOv2tiny    
        net = YOLOv2tiny(device, input_size=cfg['min_dim'], num_classes=num_classes, anchor_size=config.ANCHOR_SIZE_COCO)
   
    elif args.version == 'tiny_yolo_v3':
        from models.tiny_yolo_v3 import YOLOv3tiny
    
        net = YOLOv3tiny(device, input_size=cfg['min_dim'], num_classes=num_classes, anchor_size=config.MULTI_ANCHOR_SIZE_COCO)

    net.load_state_dict(torch.load(args.trained_model, map_location='cuda'))
    net.to(device).eval()
    print('Finished loading model!')

    # evaluation
    test_net(net, device, testset,
             BaseTransform(net.input_size, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)),
             thresh=args.visual_threshold)

if __name__ == '__main__':
    test()