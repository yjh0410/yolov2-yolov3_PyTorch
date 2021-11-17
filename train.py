from __future__ import division

import os
import random
import argparse
import time
import cv2
import numpy as np

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from data.voc0712 import VOCDetection
from data.coco2017 import COCODataset
from data import config
from data import BaseTransform, detection_collate

import tools

from utils import distributed_utils
from utils.com_paras_flops import FLOPs_and_Params
from utils.augmentations import SSDAugmentation, ColorAugmentation
from utils.cocoapi_evaluator import COCOAPIEvaluator
from utils.vocapi_evaluator import VOCAPIEvaluator
from utils.modules import ModelEMA


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Detection')
    # basic
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda.')
    parser.add_argument('--batch_size', default=16, type=int, 
                        help='Batch size for training')
    parser.add_argument('--lr', default=1e-3, type=float, 
                        help='initial learning rate')
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
    parser.add_argument('--num_workers', default=8, type=int, 
                        help='Number of workers used in dataloading')
    parser.add_argument('--num_gpu', default=1, type=int, 
                        help='Number of GPUs to train')
    parser.add_argument('--eval_epoch', type=int,
                            default=10, help='interval between evaluations')
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    parser.add_argument('--save_folder', default='weights/', type=str, 
                        help='Gamma update for SGD')
    parser.add_argument('--vis', action='store_true', default=False,
                        help='visualize target.')

    # model
    parser.add_argument('-v', '--version', default='yolo_v2',
                        help='yolov2_d19, yolov2_r50, yolov2_slim, yolov3, yolov3_spp, yolov3_tiny')
    
    # dataset
    parser.add_argument('-root', '--data_root', default='/mnt/share/ssd2/dataset',
                        help='dataset root')
    parser.add_argument('-d', '--dataset', default='voc',
                        help='voc or coco')
    
    # train trick
    parser.add_argument('-hr', '--high_resolution', action='store_true', default=False,
                        help='use hi-res pre-trained backbone.')  
    parser.add_argument('--no_warmup', action='store_true', default=False,
                        help='do not use warmup')
    parser.add_argument('-ms', '--multi_scale', action='store_true', default=False,
                        help='use multi-scale trick')      
    parser.add_argument('--mosaic', action='store_true', default=False,
                        help='use mosaic augmentation')
    parser.add_argument('--ema', action='store_true', default=False,
                        help='use ema training trick')

    # DDP train
    parser.add_argument('-dist', '--distributed', action='store_true', default=False,
                        help='distributed training')
    parser.add_argument('--local_rank', type=int, default=0, 
                        help='local_rank')
    parser.add_argument('--sybn', action='store_true', default=False, 
                        help='use sybn.')

    return parser.parse_args()


def train():
    args = parse_args()
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")

    # set distributed
    local_rank = 0
    if args.distributed:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = torch.distributed.get_rank()
        print(local_rank)
        torch.cuda.set_device(local_rank)

    # cuda
    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

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
        train_size = val_size = cfg['train_size']

    # Model ENA
    if args.ema:
        print('use EMA trick ...')

    # dataset and evaluator
    if args.dataset == 'voc':
        data_dir = os.path.join(args.data_root, 'VOCdevkit')
        num_classes = 20
        dataset = VOCDetection(data_dir=data_dir, 
                                transform=SSDAugmentation(train_size))

        evaluator = VOCAPIEvaluator(data_root=data_dir,
                                    img_size=val_size,
                                    device=device,
                                    transform=BaseTransform(val_size))

    elif args.dataset == 'coco':
        data_dir = os.path.join(args.data_root, 'COCO')
        num_classes = 80
        dataset = COCODataset(
                    data_dir=data_dir,
                    transform=SSDAugmentation(train_size))

        evaluator = COCOAPIEvaluator(
                        data_dir=data_dir,
                        img_size=val_size,
                        device=device,
                        transform=BaseTransform(val_size))
    
    else:
        print('unknow dataset !! Only support voc and coco !!')
        exit(0)
    
    print('Training model on:', dataset.name)
    print('The dataset size:', len(dataset))
    print("----------------------------------------------------------")

    # build model
    anchor_size = cfg['anchor_size_voc'] if args.dataset == 'voc' else cfg['anchor_size_coco']
    net = yolo_net(device=device, 
                   input_size=train_size, 
                   num_classes=num_classes, 
                   trainable=True, 
                   anchor_size=anchor_size, 
                   hr=hr)
    model = net

    # SyncBatchNorm
    if args.sybn and args.cuda and args.num_gpu > 1:
        print('use SyncBatchNorm ...')
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = model.to(device)
    # compute FLOPs and Params
    FLOPs_and_Params(model=model, size=train_size)

    # distributed
    if args.distributed and args.num_gpu > 1:
        print('using DDP ...')
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        # dataloader
        dataloader = torch.utils.data.DataLoader(
                        dataset=dataset, 
                        batch_size=args.batch_size, 
                        collate_fn=detection_collate,
                        num_workers=args.num_workers,
                        pin_memory=True,
                        sampler=torch.utils.data.distributed.DistributedSampler(dataset)
                        )

    else:
        model = model.train().to(device)
        # dataloader
        dataloader = torch.utils.data.DataLoader(
                        dataset=dataset, 
                        shuffle=True,
                        batch_size=args.batch_size, 
                        collate_fn=detection_collate,
                        num_workers=args.num_workers,
                        pin_memory=True
                        )

    # keep training
    if args.resume is not None:
        print('keep training model: %s' % (args.resume))
        model.load_state_dict(torch.load(args.resume, map_location=device))

    # EMA
    ema = ModelEMA(model) if args.ema else None

    # use tfboard
    if args.tfboard:
        print('use tensorboard')
        from torch.utils.tensorboard import SummaryWriter
        c_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        log_path = os.path.join('log/', args.dataset, c_time)
        os.makedirs(log_path, exist_ok=True)

        tblogger = SummaryWriter(log_path)
    
    # optimizer setup
    base_lr = args.lr
    tmp_lr = base_lr
    optimizer = optim.SGD(model.parameters(), 
                            lr=base_lr, 
                            momentum=args.momentum,
                            weight_decay=args.weight_decay
                            )

    batch_size = args.batch_size
    max_epoch = cfg['max_epoch']
    epoch_size = len(dataset) // (batch_size * args.num_gpu)

    best_map = -100.
    warmup = not args.no_warmup

    t0 = time.time()
    # start training loop
    for epoch in range(args.start_epoch, max_epoch):
        if args.distributed:
            dataloader.sampler.set_epoch(epoch)        

        # use step lr
        if epoch in cfg['lr_epoch']:
            tmp_lr = tmp_lr * 0.1
            set_lr(optimizer, tmp_lr)
    
        for iter_i, (images, targets) in enumerate(dataloader):
            # WarmUp strategy for learning rate
            ni = iter_i + epoch * epoch_size
            # warmup
            if epoch < args.wp_epoch and warmup:
                nw = args.wp_epoch * epoch_size
                tmp_lr = base_lr * pow(ni / nw, 4)
                set_lr(optimizer, tmp_lr)

            elif epoch == args.wp_epoch and iter_i == 0 and warmup:
                # warmup is over
                warmup = False
                tmp_lr = base_lr
                set_lr(optimizer, tmp_lr)

            # multi-scale trick
            if iter_i % 10 == 0 and iter_i > 0 and args.multi_scale:
                # randomly choose a new size
                r = cfg['random_size_range']
                train_size = random.randint(r[0], r[1]) * 32
                model.set_grid(train_size)
            if args.multi_scale:
                # interpolate
                images = torch.nn.functional.interpolate(images, size=train_size, mode='bilinear', align_corners=False)
            
            targets = [label.tolist() for label in targets]
            # visualize labels
            if args.vis:
                vis_data(images, targets, train_size)
                continue
            # make labels
            if model_name == 'yolov2_d19' or model_name == 'yolov2_r50' or model_name == 'yolov2_slim':
                targets = tools.gt_creator(input_size=train_size, 
                                           stride=net.stride, 
                                           label_lists=targets, 
                                           anchor_size=anchor_size
                                           )
            else:
                targets = tools.multi_gt_creator(input_size=train_size, 
                                                 strides=net.stride, 
                                                 label_lists=targets, 
                                                 anchor_size=anchor_size
                                                 )

            # to device
            images = images.float().to(device)
            targets = torch.tensor(targets).float().to(device)

            # forward
            conf_loss, cls_loss, box_loss, iou_loss = model(images, target=targets)

            # compute loss
            total_loss = conf_loss + cls_loss + box_loss + iou_loss

            loss_dict = dict(conf_loss=conf_loss,
                             cls_loss=cls_loss,
                             box_loss=box_loss,
                             iou_loss=iou_loss,
                             total_loss=total_loss
                            )

            loss_dict_reduced = distributed_utils.reduce_loss_dict(loss_dict)

            # check NAN for loss
            if torch.isnan(total_loss):
                print('nan')
                continue

            # backprop
            total_loss.backward()        
            optimizer.step()
            optimizer.zero_grad()

            # ema
            if args.ema:
                ema.update(model)

            # display
            if iter_i % 10 == 0:
                if args.tfboard:
                    # viz loss
                    tblogger.add_scalar('conf loss',  loss_dict_reduced['conf_loss'].item(),  iter_i + epoch * epoch_size)
                    tblogger.add_scalar('cls loss',  loss_dict_reduced['cls_loss'].item(),  iter_i + epoch * epoch_size)
                    tblogger.add_scalar('box loss',  loss_dict_reduced['box_loss'].item(),  iter_i + epoch * epoch_size)
                    tblogger.add_scalar('iou loss',  loss_dict_reduced['iou_loss'].item(),  iter_i + epoch * epoch_size)
                
                t1 = time.time()
                outstream = ('[Epoch %d/%d][Iter %d/%d][lr %.6f]'
                        '[Loss: conf %.2f || cls %.2f || box %.2f || iou %.2f || size %d || time: %.2f]'
                        % (epoch+1, 
                           max_epoch, 
                           iter_i, 
                           epoch_size, 
                           tmp_lr,
                           loss_dict_reduced['conf_loss'].item(),
                           loss_dict_reduced['cls_loss'].item(), 
                           loss_dict_reduced['box_loss'].item(),
                           loss_dict_reduced['iou_loss'].item(),
                           train_size, 
                           t1-t0))

                print(outstream, flush=True)

                t0 = time.time()

        # evaluation
        if (epoch + 1) % args.eval_epoch == 0:
            if args.ema:
                model_eval = ema.ema
            else:
                model_eval = model.module if args.distributed else model

            # set eval mode
            model_eval.trainable = False
            model_eval.set_grid(val_size)
            model_eval.eval()

            if local_rank == 0:
                # evaluate
                evaluator.evaluate(model_eval)

                cur_map = evaluator.map if args.dataset == 'voc' else evaluator.ap50_95
                if cur_map > best_map:
                    # update best-map
                    best_map = cur_map
                    # save model
                    print('Saving state, epoch:', epoch + 1)
                    torch.save(model_eval.state_dict(), os.path.join(path_to_save, 
                                args.version + '_' + repr(epoch + 1) + '_' + str(round(best_map, 2)) + '.pth')
                                )  
                if args.tfboard:
                    if args.dataset == 'voc':
                        tblogger.add_scalar('07test/mAP', evaluator.map, epoch)
                    elif args.dataset == 'coco':
                        tblogger.add_scalar('val/AP50_95', evaluator.ap50_95, epoch)
                        tblogger.add_scalar('val/AP50', evaluator.ap50, epoch)

            # wait for all processes to synchronize
            if args.distributed:
                dist.barrier()

            # set train mode.
            model_eval.trainable = True
            model_eval.set_grid(train_size)
            model_eval.train()
    
    if args.tfboard:
        tblogger.close()


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def vis_data(images, targets, input_size):
    # vis data
    mean=(0.406, 0.456, 0.485)
    std=(0.225, 0.224, 0.229)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)

    img = images[0].permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
    img = ((img * std + mean)*255).astype(np.uint8)
    img = img.copy()

    for box in targets[0]:
        xmin, ymin, xmax, ymax = box[:-1]
        # print(xmin, ymin, xmax, ymax)
        xmin *= input_size
        ymin *= input_size
        xmax *= input_size
        ymax *= input_size
        cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)

    cv2.imshow('img', img)
    cv2.waitKey(0)


if __name__ == '__main__':
    train()
