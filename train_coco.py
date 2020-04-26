from __future__ import division

from utils.cocoapi_evaluator import COCOAPIEvaluator
from data import *
from utils.augmentations import SSDAugmentation
from utils import get_device
from data.cocodataset import *
import tools

import os
import random
import argparse
import time
import math
import numpy as np

import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Detection')
    parser.add_argument('-v', '--version', default='yolo_v2',
                        help='yolo_v2, yolo_v3, tiny_yolo_v2, tiny_yolo_v3')
    parser.add_argument('-d', '--dataset', default='VOC',
                        help='VOC or COCO dataset')
    parser.add_argument('-hr', '--high_resolution', action='store_true', default=False,
                        help='use high resolution to pretrain.')  
    parser.add_argument('-ms', '--multi_scale', action='store_true', default=False,
                        help='use multi-scale trick')                  
    parser.add_argument('-fl', '--use_focal', action='store_true', default=False,
                        help='use focal loss')
    parser.add_argument('--batch_size', default=32, type=int, 
                        help='Batch size for training')
    parser.add_argument('--lr', default=1e-3, type=float, 
                        help='initial learning rate')
    parser.add_argument('--obj', default=5.0, type=float,
                        help='the weight of obj loss')
    parser.add_argument('--noobj', default=1.0, type=float,
                        help='the weight of noobj loss')
    parser.add_argument('-cos', '--cos', action='store_true', default=False,
                        help='use cos lr')
    parser.add_argument('-no_wp', '--no_warm_up', action='store_true', default=False,
                        help='yes or no to choose using warmup strategy to train')
    parser.add_argument('--wp_epoch', type=int, default=2,
                        help='The upper bound of warm-up')
    parser.add_argument('--dataset_root', default='./data/COCO/', 
                        help='Location of VOC root directory')
    parser.add_argument('--num_classes', default=80, type=int, 
                        help='The number of dataset classes')
    parser.add_argument('--momentum', default=0.9, type=float, 
                        help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float, 
                        help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, 
                        help='Gamma update for SGD')
    parser.add_argument('--n_cpu', default=8, type=int, 
                        help='Number of workers used in dataloading')
    parser.add_argument('--eval_epoch', type=int,
                            default=10, help='interval between evaluations')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda.')
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='debug mode where only one image is trained')
    parser.add_argument('--save_folder', default='weights_yolo_v2/coco/', type=str, 
                        help='Gamma update for SGD')

    return parser.parse_args()


def train():
    args = parse_args()
    data_dir = args.dataset_root

    path_to_save = os.path.join(args.save_folder, args.version)
    os.makedirs(path_to_save, exist_ok=True)

    hr = False  
    if args.high_resolution:
        print('use hi-res backbone')
        hr = True
    
    cfg = coco_ab

    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # build model
    if args.version == 'yolo_v2':
        from models.yolo_v2 import myYOLOv2
        total_anchor_size = tools.get_total_anchor_size(name='COCO')
    
        yolo_net = myYOLOv2(device, input_size=cfg['min_dim'], num_classes=args.num_classes, trainable=True, anchor_size=total_anchor_size, hr=hr)
        print('Let us train yolo-v2 on the COCO dataset ......')

    elif args.version == 'yolo_v3':
        from models.yolo_v3 import myYOLOv3
        total_anchor_size = tools.get_total_anchor_size(multi_scale=True, name='COCO')
        
        yolo_net = myYOLOv3(device, input_size=cfg['min_dim'], num_classes=args.num_classes, trainable=True, anchor_size=total_anchor_size, hr=hr)
        print('Let us train yolo-v3 on the COCO dataset ......')

    elif args.version == 'tiny_yolo_v2':
        from models.tiny_yolo_v2 import YOLOv2tiny
        total_anchor_size = tools.get_total_anchor_size(name='COCO')
    
        yolo_net = YOLOv2tiny(device, input_size=cfg['min_dim'], num_classes=args.num_classes, trainable=True, anchor_size=total_anchor_size, hr=hr)
        print('Let us train tiny-yolo-v2 on the COCO dataset ......')

    elif args.version == 'tiny_yolo_v3':
        from models.tiny_yolo_v3 import YOLOv3tiny
        total_anchor_size = tools.get_total_anchor_size(multi_scale=True, name='COCO')
    
        yolo_net = YOLOv3tiny(device, input_size=cfg['min_dim'], num_classes=args.num_classes, trainable=True, anchor_size=total_anchor_size, hr=hr)
        print('Let us train tiny-yolo-v3 on the COCO dataset ......')

    else:
        print('Unknown version !!!')
        exit()

    if args.multi_scale:
        print('Let us use the multi-scale trick.')
        ms_inds = range(len(cfg['multi_scale']))
        dataset = COCODataset(
                    data_dir=data_dir,
                    img_size=608,
                    transform=SSDAugmentation([608, 608], mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)),
                    debug=args.debug)
    else:
        dataset = COCODataset(
                    data_dir=data_dir,
                    img_size=cfg['min_dim'][0],
                    transform=SSDAugmentation(cfg['min_dim'], mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)),
                    debug=args.debug)
    
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")
    print('Loading the MSCOCO dataset...')
    print('Training model on:', dataset.name)
    print('The dataset size:', len(dataset))
    print('The obj weight : ', args.obj)
    print('The noobj weight : ', args.noobj)
    print("----------------------------------------------------------")


    # use tfboard
    if args.tfboard:
        print('use tensorboard')
        from torch.utils.tensorboard import SummaryWriter
        c_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        log_path = os.path.join('log/voc/', args.version, c_time)
        os.makedirs(log_path, exist_ok=True)

        writer = SummaryWriter(log_path)


    print('Let us train yolo-v2 on the MSCOCO dataset ......')
    
    model = yolo_net
    model.to(device).train()

    dataloader = torch.utils.data.DataLoader(
                    dataset, 
                    batch_size=args.batch_size, 
                    shuffle=True, 
                    collate_fn=detection_collate,
                    num_workers=args.n_cpu)

    evaluator = COCOAPIEvaluator(
                    data_dir=data_dir,
                    img_size=cfg['min_dim'],
                    device=device,
                    transform=BaseTransform(cfg['min_dim'], MEANS)
                    )

    # optimizer setup
    base_lr = args.lr
    tmp_lr = base_lr
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr)

    input_size = cfg['min_dim']
    max_epoch = cfg['max_epoch']
    epoch_size = len(dataset) // args.batch_size

    # start training loop
    t0 = time.time()

    for epoch in range(max_epoch):

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
    
        # COCO evaluation
        if (epoch + 1) % args.eval_epoch == 0:
            model.trainable = False
            ap50_95, ap50 = evaluator.evaluate(model)
            print('ap50 : ', ap50)
            print('ap50_95 : ', ap50_95)
            model.trainable = True
            model.train()
            writer.add_scalar('val/COCOAP50', ap50, epoch + 1)
            writer.add_scalar('val/COCOAP50_95', ap50_95, epoch + 1)


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
        
            # multi-scale trick
            if iter_i % 10 == 0 and iter_i > 0 and args.multi_scale:
                ms_ind = random.sample(ms_inds, 1)[0]
                input_size = cfg['multi_scale'][int(ms_ind)]
            
            # multi scale
            if args.multi_scale:
                images = torch.nn.functional.interpolate(images, size=input_size, mode='bilinear', align_corners=True)

            targets = [label.tolist() for label in targets]
            if args.version == 'yolo_v2' or args.version == 'tiny_yolo_v2':
                targets = tools.gt_creator(input_size, yolo_net.stride, targets, name='COCO')
            elif args.version == 'yolo_v3' or args.version == 'tiny_yolo_v3':
                targets = tools.multi_gt_creator(input_size, yolo_net.stride, targets, name='COCO')

            targets = torch.tensor(targets).float().to(device)

            out = model(images.to(device))


            obj_loss, class_loss, box_loss = tools.loss(out, targets, num_classes=args.num_classes, 
                                                        obj=args.obj,
                                                        noobj=args.noobj)

            total_loss = obj_loss + class_loss + box_loss

            # backprop
            total_loss.backward()        
            optimizer.step()
            optimizer.zero_grad()

            if iter_i % 10 == 0:
                if args.tfboard:
                    # viz loss
                    writer.add_scalar('object loss', obj_loss.item(), iter_i + epoch * epoch_size)
                    writer.add_scalar('class loss', class_loss.item(), iter_i + epoch * epoch_size)
                    writer.add_scalar('local loss', box_loss.item(), iter_i + epoch * epoch_size)
                
                t1 = time.time()
                print('[Epoch %d/%d][Iter %d/%d][lr %.6f]'
                    '[Loss: obj %.2f || cls %.2f || bbox %.2f || total %.2f || size %d || time: %.2f]'
                        % (epoch+1, max_epoch, iter_i, epoch_size, tmp_lr,
                            obj_loss.item(), class_loss.item(), box_loss.item(), total_loss.item(), input_size[0], t1-t0),
                        flush=True)

                t0 = time.time()


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
