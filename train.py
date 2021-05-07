from __future__ import division

import os
import random
import argparse
import time
import math
import numpy as np

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from data import VOC_CLASSES, VOC_ROOT, VOCDetection
from data import coco_root, COCODataset
from data import config
from data import BaseTransform, detection_collate

import tools

from utils.augmentations import SSDAugmentation
from utils.cocoapi_evaluator import COCOAPIEvaluator
from utils.vocapi_evaluator import VOCAPIEvaluator

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Detection')
    parser.add_argument('-v', '--version', default='yolo_v2',
                        help='yolov2_d19, yolov2_r50, yolov2_slim, yolov3, yolov3_spp, yolov3_x, yolov3_tiny')
    parser.add_argument('-d', '--dataset', default='voc',
                        help='voc or coco')
    parser.add_argument('-hr', '--high_resolution', action='store_true', default=False,
                        help='use high resolution to pretrain.')  
    parser.add_argument('-ms', '--multi_scale', action='store_true', default=False,
                        help='use multi-scale trick')                  
    parser.add_argument('--batch_size', default=32, type=int, 
                        help='Batch size for training')
    parser.add_argument('--lr', default=1e-3, type=float, 
                        help='initial learning rate')
    parser.add_argument('-cos', '--cos', action='store_true', default=False,
                        help='use cos lr')
    parser.add_argument('-no_wp', '--no_warm_up', action='store_true', default=False,
                        help='yes or no to choose using warmup strategy to train')
    parser.add_argument('--wp_epoch', type=int, default=2,
                        help='The upper bound of warm-up')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='start epoch to train')
    parser.add_argument('-r', '--resume', default=None, type=str, 
                        help='keep training')
    parser.add_argument('--momentum', default=0.9, type=float, 
                        help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float, 
                        help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, 
                        help='Gamma update for SGD')
    parser.add_argument('--num_workers', default=8, type=int, 
                        help='Number of workers used in dataloading')
    parser.add_argument('--eval_epoch', type=int,
                            default=10, help='interval between evaluations')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda.')
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='debug mode where only one image is trained')
    parser.add_argument('--save_folder', default='weights/', type=str, 
                        help='Gamma update for SGD')

    return parser.parse_args()


def train():
    args = parse_args()
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")
    # cuda
    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model_name = args.version
    print('Model: ' model_name)

    # config file
    if model_name == 'yolov2_d19':
        cfg = config.yolov2_d19_train_cfg
    elif model_name == 'yolov2_r50':
        cfg = config.yolov2_r50_train_cfg
    elif model_name == 'yolov2_slim':
        cfg = config.yolov2_slim_train_cfg
    elif model_name == 'yolov3' or model_name == 'yolov3_spp':
        cfg = config.yolov3_d53_train_cfg
    elif model_name == 'yolov3_x':
        cfg = config.yolov3_cspd53_train_cfg
    elif model_name == 'yolov3_tiny':
        cfg = config.yolov3tiny_train_cfg

    # path to save model
    path_to_save = os.path.join(args.save_folder, args.dataset, args.version)
    os.makedirs(path_to_save, exist_ok=True)

    # use hi-res backbone
    if args.high_resolution:
        print('use hi-res backbone')
        hr = True
    else:
        hr = False
    
    # multi-scale
    if args.multi_scale:
        print('use the multi-scale trick ...')
        train_size = cfg['train_size']
        val_size = cfg['val_size']
    else:
        train_size = val_size = cfg['val_size']

    # dataset and evaluator
    if args.dataset == 'voc':
        data_dir = VOC_ROOT
        num_classes = 20
        dataset = VOCDetection(root=data_dir, 
                                transform=SSDAugmentation(train_size)
                                )

        evaluator = VOCAPIEvaluator(data_root=data_dir,
                                    img_size=val_size,
                                    device=device,
                                    transform=BaseTransform(val_size),
                                    labelmap=VOC_CLASSES
                                    )

    elif args.dataset == 'coco':
        data_dir = coco_root
        num_classes = 80
        dataset = COCODataset(
                    data_dir=data_dir,
                    img_size=train_size[0],
                    transform=SSDAugmentation(train_size),
                    debug=args.debug)


        evaluator = COCOAPIEvaluator(
                        data_dir=data_dir,
                        img_size=val_size,
                        device=device,
                        transform=BaseTransform(val_size)
                        )
    
    else:
        print('unknow dataset !! Only support voc and coco !!')
        exit(0)
    
    print('Training model on:', dataset.name)
    print('The dataset size:', len(dataset))
    print("----------------------------------------------------------")

    # dataloader
    dataloader = torch.utils.data.DataLoader(
                    dataset, 
                    batch_size=args.batch_size, 
                    shuffle=True, 
                    collate_fn=detection_collate,
                    num_workers=args.num_workers,
                    pin_memory=True
                    )

    # build model
    if model_name == 'yolov2_d19':
        from models.yolov2_d19 import YOLOv2D19
        anchor_size = cfg['anchor_size_voc'] if args.dataset == 'voc' else cfg['anchor_size_coco']
        # model
        yolo_net = YOLOv2D19(device=device, 
                             input_size=train_size, 
                             num_classes=num_classes, 
                             trainable=True, 
                             anchor_size=anchor_size, 
                             hr=hr)
        print('Let us train yolov2_d19... ')

    elif model_name == 'yolov2_r50':
        from models.yolov2_r50 import YOLOv2R50
        anchor_size = cfg['anchor_size_voc'] if args.dataset == 'voc' else cfg['anchor_size_coco']
        # model
        yolo_net = YOLOv2R50(device=device, 
                             input_size=train_size, 
                             num_classes=num_classes, 
                             trainable=True, 
                             anchor_size=anchor_size, 
                             hr=hr)
        print('Let us train yolov2_r50... ')

    elif model_name == 'yolov3':
        from models.yolov3 import YOLOv3
        anchor_size = cfg['anchor_size_voc'] if args.dataset == 'voc' else cfg['anchor_size_coco']
        # model
        yolo_net = YOLOv3(device=device, 
                          input_size=train_size, 
                          num_classes=num_classes, 
                          trainable=True, 
                          anchor_size=anchor_size, 
                          hr=hr)
        print('Let us train yolov3...')

    elif model_name == 'yolov3_spp':
        from models.yolov3_spp import YOLOv3Spp
        anchor_size = cfg['anchor_size_voc'] if args.dataset == 'voc' else cfg['anchor_size_coco']
        # model
        yolo_net = YOLOv3Spp(device=device, 
                             input_size=train_size, 
                             num_classes=num_classes, 
                             trainable=True, 
                             anchor_size=anchor_size, 
                             hr=hr)
        print('Let us train yolov3_spp...')

    elif model_name == 'slim_yolo_v2':
        from models.slim_yolo_v2 import SlimYOLOv2
        anchor_size = ANCHOR_SIZE if args.dataset == 'voc' else ANCHOR_SIZE_COCO
    
        yolo_net = SlimYOLOv2(device, input_size=train_size, num_classes=num_classes, trainable=True, anchor_size=anchor_size, hr=hr)
        print('Let us train slim_yolo_v2 on the %s dataset ......' % (args.dataset))

    elif model_name == 'tiny_yolo_v3':
        from models.tiny_yolo_v3 import YOLOv3tiny
        anchor_size = TINY_MULTI_ANCHOR_SIZE if args.dataset == 'voc' else TINY_MULTI_ANCHOR_SIZE_COCO
    
        yolo_net = YOLOv3tiny(device, input_size=train_size, num_classes=num_classes, trainable=True, anchor_size=anchor_size, hr=hr)
        print('Let us train tiny_yolo_v3 on the %s dataset ......' % (args.dataset))

    else:
        print('Unknown version !!!')
        exit()

    model = yolo_net
    model.to(device).train()

    # use tfboard
    if args.tfboard:
        print('use tensorboard')
        from torch.utils.tensorboard import SummaryWriter
        c_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        log_path = os.path.join('log/coco/', args.version, c_time)
        os.makedirs(log_path, exist_ok=True)

        writer = SummaryWriter(log_path)
    
    # keep training
    if args.resume is not None:
        print('keep training model: %s' % (args.resume))
        model.load_state_dict(torch.load(args.resume, map_location=device))

    # optimizer setup
    base_lr = args.lr
    tmp_lr = base_lr
    optimizer = optim.SGD(model.parameters(), 
                            lr=args.lr, 
                            momentum=args.momentum,
                            weight_decay=args.weight_decay
                            )

    max_epoch = cfg['max_epoch']
    epoch_size = len(dataset) // args.batch_size

    # start training loop
    t0 = time.time()

    for epoch in range(args.start_epoch, max_epoch):

        # use cos lr
        if args.cos and epoch > 20 and epoch <= max_epoch - 20:
            # use cos lr
            tmp_lr = 0.00001 + 0.5*(base_lr-0.00001)*(1+math.cos(math.pi*(epoch-20)*1./ (max_epoch-20)))
            set_lr(optimizer, tmp_lr)

        elif args.cos and epoch > max_epoch - 20:
            tmp_lr = 0.00001
            set_lr(optimizer, tmp_lr)
        
        # use step lr
        else:
            if epoch in cfg['lr_epoch']:
                tmp_lr = tmp_lr * 0.1
                set_lr(optimizer, tmp_lr)
    

        for iter_i, (images, targets) in enumerate(dataloader):
            # WarmUp strategy for learning rate
            if not args.no_warm_up:
                if epoch < args.wp_epoch:
                    tmp_lr = base_lr * pow((iter_i+epoch*epoch_size)*1. / (args.wp_epoch*epoch_size), 4)
                    # tmp_lr = 1e-6 + (base_lr-1e-6) * (iter_i+epoch*epoch_size) / (epoch_size * (args.wp_epoch))
                    set_lr(optimizer, tmp_lr)

                elif epoch == args.wp_epoch and iter_i == 0:
                    tmp_lr = base_lr
                    set_lr(optimizer, tmp_lr)
        
            # to device
            images = images.to(device)

            # multi-scale trick
            if iter_i % 10 == 0 and iter_i > 0 and args.multi_scale:
                # randomly choose a new size
                size = random.randint(10, 19) * 32
                train_size = [size, size]
                model.set_grid(train_size)
            if args.multi_scale:
                # interpolate
                images = torch.nn.functional.interpolate(images, size=train_size, mode='bilinear', align_corners=False)
            
            # make labels
            targets = [label.tolist() for label in targets]
            if model_name == 'yolo_v2' or model_name == 'slim_yolo_v2':
                targets = tools.gt_creator(input_size=train_size, 
                                           stride=yolo_net.stride, 
                                           label_lists=targets, 
                                           anchor_size=anchor_size
                                           )
            else:
                targets = tools.multi_gt_creator(input_size=train_size, 
                                                 strides=yolo_net.stride, 
                                                 label_lists=targets, 
                                                 anchor_size=anchor_size
                                                 )
            targets = torch.tensor(targets).float().to(device)

            # forward and loss
            conf_loss, cls_loss, txtytwth_loss, total_loss = model(images, target=targets)

            # backprop
            total_loss.backward()        
            optimizer.step()
            optimizer.zero_grad()

            # display
            if iter_i % 10 == 0:
                if args.tfboard:
                    # viz loss
                    writer.add_scalar('object loss', conf_loss.item(), iter_i + epoch * epoch_size)
                    writer.add_scalar('class loss', cls_loss.item(), iter_i + epoch * epoch_size)
                    writer.add_scalar('local loss', txtytwth_loss.item(), iter_i + epoch * epoch_size)
                
                t1 = time.time()
                print('[Epoch %d/%d][Iter %d/%d][lr %.6f]'
                    '[Loss: obj %.2f || cls %.2f || bbox %.2f || total %.2f || size %d || time: %.2f]'
                        % (epoch+1, max_epoch, iter_i, epoch_size, tmp_lr,
                            conf_loss.item(), cls_loss.item(), txtytwth_loss.item(), total_loss.item(), train_size[0], t1-t0),
                        flush=True)

                t0 = time.time()

        # evaluation
        if (epoch + 1) % args.eval_epoch == 0:
            model.trainable = False
            model.set_grid(val_size)
            model.eval()

            # evaluate
            evaluator.evaluate(model)

            # convert to training mode.
            model.trainable = True
            model.set_grid(train_size)
            model.train()

        # save model
        if (epoch + 1) % 10 == 0:
            print('Saving state, epoch:', epoch + 1)
            torch.save(model.state_dict(), os.path.join(path_to_save, 
                        args.version + '_' + repr(epoch + 1) + '.pth')
                        )  


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    train()
