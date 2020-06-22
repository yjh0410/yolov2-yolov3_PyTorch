import argparse
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from data import BaseTransform, VOC_CLASSES
from data import config
import numpy as np
import cv2
import tools
import time
from decimal import *

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


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Demo Detection')

    parser.add_argument('-v', '--version', default='yolo_v2',
                        help='yolo_v2 and tiny_yolo_v2.')
    parser.add_argument('--trained_model', default='weights/',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--mode', default='image',
                        type=str, help='Use the data from image, video or camera')
    parser.add_argument('--setup', default='VOC',
                        type=str, help='Use the VOC setup or COCO')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Use cuda')
    parser.add_argument('--path_to_img', default='data/demo/Images/',
                        type=str, help='The path to image files')
    parser.add_argument('--path_to_vid', default='data/demo/video/',
                        type=str, help='The path to video files')
    parser.add_argument('--path_to_saveVid', default='data/video/result.avi',
                        type=str, help='The path to save the detection results video')
    parser.add_argument('--visual_threshold', default=0.3,
                        type=float, help='visual threshold')
    
    return parser.parse_args()
                    

def vis(img, bbox_pred, scores, cls_inds, class_color, setup='VOC', thresh=0.3):
        
    for i, box in enumerate(bbox_pred):
        if scores[i] > thresh:
            cls_indx = cls_inds[i]
            if setup == 'VOC':
                cls_name = VOC_CLASSES[int(cls_indx)]
                mess = '%s: %.3f' % (cls_name, scores[i])
            elif setup == 'COCO':
                cls_id = coco_class_index[int(cls_indx)]
                cls_name = coco_class_labels[cls_id]
                mess = '%s: %.3f' % (cls_name, scores[i])
            # bounding box
            xmin, ymin, xmax, ymax = box
            box_w = int(xmax - xmin)
            # print(xmin, ymin, xmax, ymax)
            cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), class_color[int(cls_indx)], 2)
            cv2.rectangle(img, (int(xmin), int(abs(ymin)-15)), (int(xmin+box_w*0.55), int(ymin)), class_color[int(cls_indx)], -1)

            cv2.putText(img, mess, (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
    return img

def detect(net, device, transform, thresh, mode='image', path_to_img=None, path_to_vid=None, path_to_save=None, setup='VOC'):
    if setup == 'VOC':
        class_color = [(np.random.randint(255),np.random.randint(255),np.random.randint(255)) for _ in range(20)]
    elif setup == 'COCO':
        class_color = [(np.random.randint(255),np.random.randint(255),np.random.randint(255)) for _ in range(80)]
    # ------------------------- Camera ----------------------------
    # I'm not sure whether this 'camera' mode works ...
    if mode == 'camera':
        print('use camera !!!')
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        while True:
            ret, frame = cap.read()
            cv2.imshow('current frame', frame)
            if cv2.waitKey(1) == ord('q'):
                break
            x = torch.from_numpy(transform(frame)[0][:, :, (2, 1, 0)]).permute(2, 0, 1)
            x = x.unsqueeze(0).to(device)

            torch.cuda.synchronize()
            t0 = time.time()
            detections = net(x)      # forward pass
            torch.cuda.synchronize()
            t1 = time.time()
            print("detection time used ", t1-t0, "s")
            # scale each detection back up to the image
            scale = np.array([[frame.shape[1], frame.shape[0],
                                frame.shape[1], frame.shape[0]]])
            bbox_pred, scores, cls_inds = detections
            # map the boxes to origin image scale
            bbox_pred *= scale

            frame_processed = vis(frame, bbox_pred, scores, cls_inds, class_color, setup, thresh=thresh)
            cv2.imshow('detection result', frame_processed)
            cv2.waitKey(1)
        cap.release()
        cv2.destroyAllWindows()

    # ------------------------- Image ----------------------------
    elif mode == 'image':
        for file in os.listdir(path_to_img):
            img = cv2.imread(path_to_img + '/' + file, cv2.IMREAD_COLOR)
            x = torch.from_numpy(transform(img)[0][:, :, (2, 1, 0)]).permute(2, 0, 1)
            x = x.unsqueeze(0).to(device)

            torch.cuda.synchronize()
            t0 = time.time()
            detections = net(x)      # forward pass
            torch.cuda.synchronize()
            t1 = time.time()
            print("detection time used ", t1-t0, "s")
            # scale each detection back up to the image
            scale = np.array([[img.shape[1], img.shape[0],
                                img.shape[1], img.shape[0]]])
            bbox_pred, scores, cls_inds = detections
            # map the boxes to origin image scale
            bbox_pred *= scale

            img_processed = vis(img, bbox_pred, scores, cls_inds, class_color=class_color, setup=setup, thresh=thresh)
            cv2.imshow('detection result', img_processed)
            cv2.waitKey(0)

    # ------------------------- Video ---------------------------
    elif mode == 'video':
        video = cv2.VideoCapture(path_to_vid)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output000.avi',fourcc, 40.0, (1280,720))
        while(True):
            ret, frame = video.read()
            
            if ret:
                # ------------------------- Detection ---------------------------
                t0 = time.time()
                x = torch.from_numpy(transform(frame)[0][:, :, (2, 1, 0)]).permute(2, 0, 1)
                x = x.unsqueeze(0).to(device)

                torch.cuda.synchronize()
                t0 = time.time()
                detections = net(x)      # forward pass
                torch.cuda.synchronize()
                t1 = time.time()
                print("detection time used ", t1-t0, "s")
                # scale each detection back up to the image
                scale = np.array([[frame.shape[1], frame.shape[0],
                                    frame.shape[1], frame.shape[0]]])
                bbox_pred, scores, cls_inds = detections
                # map the boxes to origin image scale
                bbox_pred *= scale
                
                frame_processed = vis(frame, bbox_pred, scores, cls_inds, class_color=class_color, setup=setup, thresh=thresh)
                out.write(frame_processed)
                cv2.imshow('detection result', frame_processed)
                cv2.waitKey(1)
            else:
                break
        video.release()
        out.release()
        cv2.destroyAllWindows()

def run():
    args = parse_args()

    if args.cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.setup == 'VOC':
        cfg = config.voc_ab
        num_classes = 20
    elif args.setup == 'COCO':
        cfg = config.coco_ab
        num_classes = 80
    else:
        print('Only support VOC and COCO !!!')
        exit(0)

    if args.version == 'yolo_v2':
        from models.yolo_v2 import myYOLOv2
        if args.setup == 'VOC':
            anchor_size = config.ANCHOR_SIZE
        else:
            anchor_size = config.ANCHOR_SIZE_COCO

        net = myYOLOv2(device, input_size=cfg['min_dim'], num_classes=num_classes, trainable=False, anchor_size=anchor_size)

    elif args.version == 'yolo_v3':
        from models.yolo_v3 import myYOLOv3
        if args.setup == 'VOC':
            anchor_size = config.MULTI_ANCHOR_SIZE
        else:
            anchor_size = config.MULTI_ANCHOR_SIZE_COCO

        net = myYOLOv3(device, input_size=cfg['min_dim'], num_classes=num_classes, trainable=False, anchor_size=anchor_size)

    elif args.version == 'slim_yolo_v2':
        from models.slim_yolo_v2 import SlimYOLOv2
        if args.setup == 'VOC':
            anchor_size = config.ANCHOR_SIZE
        else:
            anchor_size = config.ANCHOR_SIZE_COCO

        net = SlimYOLOv2(device, input_size=cfg['min_dim'], num_classes=num_classes, trainable=False, anchor_size=anchor_size)

    elif args.version == 'tiny_yolo_v3':
        from models.tiny_yolo_v3 import YOLOv3tiny
        if args.setup == 'VOC':
            anchor_size = config.TINY_MULTI_ANCHOR_SIZE
        else:
            anchor_size = config.TINY_MULTI_ANCHOR_SIZE_COCO

        net = YOLOv3tiny(device, input_size=cfg['min_dim'], num_classes=num_classes, trainable=False, anchor_size=anchor_size)
    
    net.load_state_dict(torch.load(args.trained_model, map_location='cuda'))
    net.to(device).eval()
    print('Finished loading model!')

    # run
    if args.mode == 'camera':
        detect(net, device, BaseTransform(net.input_size, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)), 
                    thresh=args.visual_threshold, mode=args.mode, setup=args.setup)
    elif args.mode == 'image':
        detect(net, device, BaseTransform(net.input_size, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)), 
                    thresh=args.visual_threshold, mode=args.mode, path_to_img=args.path_to_img, setup=args.setup)
    elif args.mode == 'video':
        detect(net, device, BaseTransform(net.input_size, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)),
                    thresh=args.visual_threshold, mode=args.mode, path_to_vid=args.path_to_vid, path_to_save=args.path_to_saveVid, setup=args.setup)


if __name__ == '__main__':
    run()
