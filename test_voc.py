import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from data import VOC_ROOT, VOC_CLASSES
from data import VOCAnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES
from data import config
import numpy as np
import cv2
import tools
import time
from decimal import *


parser = argparse.ArgumentParser(description='YOLO Detection')
parser.add_argument('-v', '--version', default='yolo_v2',
                    help='yolo_v2, yolo_v3, tiny_yolo_v2, tiny_yolo_v3')
parser.add_argument('-d', '--dataset', default='VOC',
                    help='VOC or COCO dataset')
parser.add_argument('--trained_model', default='weight/voc/',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--visual_threshold', default=0.3, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', action='store_true', default=False, 
                    help='use cuda.')
parser.add_argument('--voc_root', default=VOC_ROOT, 
                    help='Location of VOC root directory')
parser.add_argument('-f', default=None, type=str, 
                    help="Dummy arg so we can load in Jupyter Notebooks")

args = parser.parse_args()

def test_net(net, device, testset, transform, thresh, mode='voc'):
    num_images = len(testset)
    for index in range(num_images):
        print('Testing image {:d}/{:d}....'.format(index+1, num_images))
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

        CLASSES = VOC_CLASSES
        class_color = tools.CLASS_COLOR
        for i, box in enumerate(bbox_pred):
            cls_indx = cls_inds[i]
            xmin, ymin, xmax, ymax = box
            if scores[i] > thresh:
                cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), class_color[int(cls_indx)], 2)
                cv2.rectangle(img, (int(xmin), int(abs(ymin)-20)), (int(xmax), int(ymin)), class_color[int(cls_indx)], -1)
                mess = '%s' % (CLASSES[int(cls_indx)])
                cv2.putText(img, mess, (int(xmin), int(ymin-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
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
    num_classes = len(VOC_CLASSES)
    testset = VOCDetection(args.voc_root, [('2007', 'test')], None, VOCAnnotationTransform())

    cfg = config.voc_ab
    if args.version == 'yolo_v2':
        from models.yolo_v2 import myYOLOv2
        net = myYOLOv2(device, input_size=cfg['min_dim'], num_classes=num_classes, anchor_size=config.ANCHOR_SIZE)
        print('Let us test yolo-v2 on the VOC0712 dataset ......')

    elif args.version == 'yolo_v3':
        from models.yolo_v3 import myYOLOv3
        net = myYOLOv3(device, input_size=cfg['min_dim'], num_classes=num_classes, anchor_size=config.MULTI_ANCHOR_SIZE)
    
    elif args.version == 'tiny_yolo_v2':
        from models.tiny_yolo_v2 import YOLOv2tiny    
        net = YOLOv2tiny(device, input_size=cfg['min_dim'], num_classes=num_classes, anchor_size=config.ANCHOR_SIZE)
        print('Let us test tiny-yolo-v2 on the VOC0712 dataset ......')
   
    elif args.version == 'tiny_yolo_v3':
        from models.tiny_yolo_v3 import YOLOv3tiny
    
        net = YOLOv3tiny(device, input_size=cfg['min_dim'], num_classes=num_classes, anchor_size=config.TINY_MULTI_ANCHOR_SIZE)
        print('Let us test tiny-yolo-v3 on the VOC0712 dataset ......')

    net.load_state_dict(torch.load(args.trained_model, map_location=device))
    net.to(device).eval()
    print('Finished loading model!')

    # evaluation
    test_net(net, device, testset,
             BaseTransform(net.input_size, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)),
             thresh=args.visual_threshold)

if __name__ == '__main__':
    test()