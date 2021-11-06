import os
import argparse
import torch
import torch.backends.cudnn as cudnn
from data.voc0712 import VOC_CLASSES, VOCDetection
from data.coco2017 import COCODataset, coco_class_index, coco_class_labels
from data import config, BaseTransform
import numpy as np
import cv2
import time


parser = argparse.ArgumentParser(description='YOLO Detection')
# basic
parser.add_argument('-size', '--input_size', default=416, type=int,
                    help='input_size')
parser.add_argument('--cuda', action='store_true', default=False, 
                    help='use cuda.')
# model
parser.add_argument('-v', '--version', default='yolo_v2',
                    help='yolov2_d19, yolov2_r50, yolov2_slim, yolov3, yolov3_spp, yolov3_tiny')
parser.add_argument('--trained_model', default='weight/',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--conf_thresh', default=0.1, type=float,
                    help='Confidence threshold')
parser.add_argument('--nms_thresh', default=0.50, type=float,
                    help='NMS threshold')
# dataset
parser.add_argument('-root', '--data_root', default='/mnt/share/ssd2/dataset',
                    help='dataset root')
parser.add_argument('-d', '--dataset', default='voc',
                    help='voc or coco')
# visualize
parser.add_argument('-vs', '--visual_threshold', default=0.25, type=float,
                    help='Final confidence threshold')
parser.add_argument('--show', action='store_true', default=False,
                    help='show the visulization results.')


args = parser.parse_args()


def plot_bbox_labels(img, bbox, label=None, cls_color=None, text_scale=0.4):
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
    # plot bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), cls_color, 2)
    
    if label is not None:
        # plot title bbox
        cv2.rectangle(img, (x1, y1-t_size[1]), (int(x1 + t_size[0] * text_scale), y1), cls_color, -1)
        # put the test on the title bbox
        cv2.putText(img, label, (int(x1), int(y1 - 5)), 0, text_scale, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    return img


def visualize(img, 
              bboxes, 
              scores, 
              cls_inds, 
              vis_thresh, 
              class_colors, 
              class_names, 
              class_indexs=None, 
              dataset_name='voc'):
    ts = 0.4
    for i, bbox in enumerate(bboxes):
        if scores[i] > vis_thresh:
            cls_id = int(cls_inds[i])
            if dataset_name == 'coco':
                cls_color = class_colors[cls_id]
                cls_id = class_indexs[cls_id]
            else:
                cls_color = class_colors[cls_id]
                
            if len(class_names) > 1:
                mess = '%s: %.2f' % (class_names[cls_id], scores[i])
            else:
                cls_color = [255, 0, 0]
                mess = None
            img = plot_bbox_labels(img, bbox, mess, cls_color, text_scale=ts)

    return img
        

def test(net, 
         device, 
         dataset, 
         transform, 
         vis_thresh, 
         class_colors=None, 
         class_names=None, 
         class_indexs=None, 
         dataset_name='voc'):

    num_images = len(dataset)
    save_path = os.path.join('det_results/', args.dataset, args.version)
    os.makedirs(save_path, exist_ok=True)

    for index in range(num_images):
        print('Testing image {:d}/{:d}....'.format(index+1, num_images))
        image, _ = dataset.pull_image(index)
        h, w, _ = image.shape
        scale = np.array([[w, h, w, h]])

        # to tensor
        x = torch.from_numpy(transform(image)[0][:, :, (2, 1, 0)]).permute(2, 0, 1)
        x = x.unsqueeze(0).to(device)

        t0 = time.time()
        # forward
        bboxes, scores, cls_inds = net(x)
        print("detection time used ", time.time() - t0, "s")
        
        # rescale
        bboxes *= scale

        # vis detection
        img_processed = visualize(
                            img=image,
                            bboxes=bboxes,
                            scores=scores,
                            cls_inds=cls_inds,
                            vis_thresh=vis_thresh,
                            class_colors=class_colors,
                            class_names=class_names,
                            class_indexs=class_indexs,
                            dataset_name=dataset_name
                            )
        if args.show:
            cv2.imshow('detection', img_processed)
            cv2.waitKey(0)
        # save result
        cv2.imwrite(os.path.join(save_path, str(index).zfill(6) +'.jpg'), img_processed)


if __name__ == '__main__':
    # cuda
    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # input size
    input_size = args.input_size

    # dataset
    if args.dataset == 'voc':
        print('test on voc ...')
        data_dir = os.path.join(args.data_root, 'VOCdevkit')
        class_names = VOC_CLASSES
        class_indexs = None
        num_classes = 20
        dataset = VOCDetection(root=data_dir, 
                                image_sets=[('2007', 'test')])

    elif args.dataset == 'coco':
        print('test on coco-val ...')
        data_dir = os.path.join(args.data_root, 'COCO')
        class_names = coco_class_labels
        class_indexs = coco_class_index
        num_classes = 80
        dataset = COCODataset(
                    data_dir=data_dir,
                    json_file='instances_val2017.json',
                    name='val2017')

    class_colors = [(np.random.randint(255), 
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(num_classes)]

    # model
    model_name = args.version
    print('Model: ', model_name)

    # load model and config file
    if model_name == 'yolov2_d19':
        from models.yolov2_d19 import YOLOv2D19 as yolo_net
        cfg = config.yolov2_d19_cfg

    elif model_name == 'yolov2_r50':
        from models.yolov2_r50 import YOLOv2R50 as yolo_net
        cfg = config.yolov2_r50_cfg

    elif model_name == 'yolov2_slim':
        from models.yolov2_slim import YOLOv2Slim as yolo_net
        cfg = config.yolov2_slim_cfg

    elif model_name == 'yolov3':
        from models.yolov3 import YOLOv3 as yolo_net
        cfg = config.yolov3_d53_cfg

    elif model_name == 'yolov3_spp':
        from models.yolov3_spp import YOLOv3Spp as yolo_net
        cfg = config.yolov3_d53_cfg

    elif model_name == 'yolov3_tiny':
        from models.yolov3_tiny import YOLOv3tiny as yolo_net
        cfg = config.yolov3tiny_cfg
    else:
        print('Unknown model name...')
        exit(0)

    # build model
    anchor_size = cfg['anchor_size_voc'] if args.dataset == 'voc' else cfg['anchor_size_coco']
    net = yolo_net(device=device, 
                   input_size=input_size, 
                   num_classes=num_classes, 
                   trainable=False, 
                   conf_thresh=args.conf_thresh,
                   nms_thresh=args.nms_thresh,
                   anchor_size=anchor_size)

    # load weight
    net.load_state_dict(torch.load(args.trained_model, map_location=device))
    net.to(device).eval()
    print('Finished loading model!')

    # evaluation
    test(net=net, 
        device=device, 
        dataset=dataset,
        transform=BaseTransform(input_size),
        vis_thresh=args.visual_threshold,
        class_colors=class_colors,
        class_names=class_names,
        class_indexs=class_indexs,
        dataset_name=args.dataset
        )
