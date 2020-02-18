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
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.autograd import Variable
import torch.optim as optim

from models.yolo_v2 import myYOLOv2

parser = argparse.ArgumentParser(description='YOLO-v1 Detection')
parser.add_argument('-v', '--version', default='yolo_v2',
                    help='yolo_v2')
parser.add_argument('-d', '--dataset', default='VOC',
                    help='VOC or COCO dataset')
parser.add_argument('-hr', '--high_resolution', type=int, default=0,
                    help='0: use high resolution to pretrain; 1: else not.')   
parser.add_argument('-ms', '--multi_scale', type=int, default=0,
                    help='1: use multi-scale trick; 0: else not')                                   
parser.add_argument('-fl', '--use_focal', type=int, default=0,
                    help='0: use focal loss; 1: else not;')
parser.add_argument('--batch_size', default=64, type=int, 
                    help='Batch size for training')
parser.add_argument('--lr', default=1e-4, type=float, 
                    help='initial learning rate')
parser.add_argument('--obj', default=1.0, type=float,
                    help='the weight of obj loss')
parser.add_argument('--noobj', default=0.5, type=float,
                    help='the weight of noobj loss')
parser.add_argument('-wp', '--warm_up', type=str, default='yes',
                    help='yes or no to choose using warmup strategy to train')
parser.add_argument('--wp_epoch', type=int, default=4,
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
parser.add_argument('--gpu_ind', default=0, type=int, 
                    help='To choose your gpu.')
parser.add_argument('--debug', action='store_true', default=False,
                    help='debug mode where only one image is trained')
parser.add_argument('--save_folder', default='weights_yolo_v2/coco/', type=str, 
                    help='Gamma update for SGD')


args = parser.parse_args()
data_dir = args.dataset_root

def train(net, device):
    global cfg, hr
    # set GPU

    use_focal = False
    if args.use_focal == 1:
        print("Let's use focal loss for objectness !!!")
        use_focal = True

    if args.multi_scale == 1:
        print('Let us use the multi-scale trick.')
        ms_inds = range(len(cfg['multi_scale']))
        dataset = COCODataset(
                    data_dir=data_dir,
                    img_size=608,
                    transform=SSDAugmentation([608, 608], MEANS),
                    debug=args.debug)
    else:
        dataset = COCODataset(
                    data_dir=data_dir,
                    img_size=cfg['min_dim'][0],
                    transform=SSDAugmentation(cfg['min_dim'], MEANS),
                    debug=args.debug)
    
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")
    print('Loading the MSCOCO dataset...')
    print('Training model on:', dataset.name)
    print('The dataset size:', len(dataset))
    print('The obj weight : ', args.obj)
    print('The noobj weight : ', args.noobj)
    print("----------------------------------------------------------")

    input_size = cfg['min_dim']
    num_classes = args.num_classes
    batch_size = args.batch_size
    save_folder = args.save_folder

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    # using tfboard
    from tensorboardX import SummaryWriter
    c_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    log_path = 'log/yolo_v2/coco/' + c_time
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    writer = SummaryWriter(log_path)

    if args.high_resolution == 1:
        hr = True

    print('Let us train yolo-v2 on the MSCOCO dataset ......')
    
    model = net
    model.to(device).train()

    dataloader = torch.utils.data.DataLoader(
                    dataset, 
                    batch_size=batch_size, 
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
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
    #                       dampening=0, weight_decay=args.weight_decay)
    # optimizer = optim.Adam(model.parameters())
    lr = args.lr
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr)

    step_index = 0
    epoch_size = len(dataset) // args.batch_size
    # each part of loss weight
    obj_w = 1.0
    cla_w = 1.0
    box_w = 2.0

    # start training loop
    iteration = 0
    for epoch in range(cfg['max_epoch']):
        batch_iterator = iter(dataloader)

        # No WarmUp strategy or WarmUp tage has finished.
        if epoch in cfg['lr_epoch']:
            step_index += 1
            lr = adjust_learning_rate(optimizer, args.gamma, step_index)

        # COCO evaluation
        if (epoch + 1) % args.eval_epoch == 0:
            model.trainable = False
            ap50_95, ap50 = evaluator.evaluate(model)
            print('ap50 : ', ap50)
            print('ap90_95 : ', ap50_95)
            model.trainable = True
            model.train()
            writer.add_scalar('val/COCOAP50', ap50, epoch + 1)
            writer.add_scalar('val/COCOAP50_95', ap50_95, epoch + 1)

        # subdivision loop
        optimizer.zero_grad()
        for images, targets in batch_iterator:
            iteration += 1
        
            # multi-scale trick
            if iteration % 10 == 0 and args.multi_scale == 1:
                ms_ind = random.sample(ms_inds, 1)[0]
                input_size = cfg['multi_scale'][int(ms_ind)]
            
            # multi scale
            if args.multi_scale == 1:
                images = torch.nn.functional.interpolate(images, size=input_size, mode='bilinear', align_corners=True)

            targets = [label.tolist() for label in targets]
            if args.version == 'yolo_v2':
                targets = tools.gt_creator(input_size, yolo_net.stride, args.num_classes, targets, name='COCO')
            elif args.version == 'yolo_v3':
                targets = tools.multi_gt_creator(input_size, yolo_net.stride, args.num_classes, targets, name='COCO')

            targets = torch.tensor(targets).float().to(device)

            t0 = time.time()
            out = model(images.to(device))
            obj_loss, class_loss, box_loss = tools.loss(out, targets, num_classes=args.num_classes, 
                                                        use_focal=use_focal,
                                                        obj=args.obj,
                                                        noobj=args.noobj)
            total_loss = obj_w * obj_loss + cla_w * class_loss + box_w * box_loss

            # viz loss
            writer.add_scalar('object loss', obj_loss.item(), iteration)
            writer.add_scalar('class loss', class_loss.item(), iteration)
            writer.add_scalar('local loss', box_loss.item(), iteration)
            writer.add_scalar('total loss', total_loss.item(), iteration)
            # backprop
            total_loss.backward()        
            optimizer.step()
            t1 = time.time()

            if iteration % 10 == 0:
                print('timer: %.4f sec.' % (t1 - t0))
                # print(obj_loss.item(), class_loss.item(), box_loss.item())
                print('Epoch[%d / %d]' % (epoch+1, cfg['max_epoch']) + ' || iter ' + repr(iteration) + \
                      ' || Loss: %.4f ||' % (total_loss.item()) + ' || lr: %.8f ||' % (lr) + ' || input size: %d ||' % input_size[0], end=' ')

        if (epoch + 1) % 10 == 0:
            print('Saving state, epoch:', epoch + 1)
            torch.save(yolo_net.state_dict(), save_folder+ '/' + args.version + '_' +
                    repr(epoch + 1) + '.pth')

def adjust_learning_rate(optimizer, gamma, step):
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def warmup_strategy(optimizer, epoch_size, iteration):
    lr = 1e-5 + (args.lr-1e-5) * iteration / (epoch_size * (args.wp_epoch)) 
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr
if __name__ == '__main__':
    global hr, cfg

    hr = False
    device = get_device(args.gpu_ind)
    
    if args.high_resolution == 1:
        hr = True
    
    cfg = coco_ab

    if args.version == 'yolo_v2':
        from models.yolo_v2 import myYOLOv2
        total_anchor_size = tools.get_total_anchor_size(name='COCO')
        yolo_net = myYOLOv2(device, input_size=cfg['min_dim'], num_classes=args.num_classes, trainable=True, anchor_size=total_anchor_size, hr=hr)
        print('Let us train yolo-v2 on the MSCOCO dataset ......')

    elif args.version == 'yolo_v3':
        from models.yolo_v3 import myYOLOv3
        total_anchor_size = tools.get_total_anchor_size(multi_scale=True, name='COCO')
        
        yolo_net = myYOLOv3(device, input_size=cfg['min_dim'], num_classes=args.num_classes, trainable=True, anchor_size=total_anchor_size, hr=hr)
        print('Let us train yolo-v3 on the MSCOCO dataset ......')

    train(yolo_net, device)
