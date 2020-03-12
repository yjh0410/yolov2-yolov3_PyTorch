import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models
from utils import Conv2d, reorg_layer
from backbone import *
import os
import numpy as np

class myYOLOv3(nn.Module):
    def __init__(self, device, input_size=None, num_classes=20, trainable=False, conf_thresh=0.001, nms_thresh=0.50, anchor_size=None, hr=False):
        super(myYOLOv3, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.stride = [8, 16, 32]
        self.anchor_size = torch.tensor(anchor_size).view(3, len(anchor_size) // 3, 2)
        self.anchor_number = self.anchor_size.size(1)
        if not trainable:
            self.anchor_size[0, :] *= 4
            self.anchor_size[1, :] *= 2
            self.anchor_size[2, :] *= 1
            self.input_size = input_size
            self.grid_cell, self.all_anchor_wh, self.stride_tensor = self.init_grid(input_size)
            self.scale = np.array([[input_size[1], input_size[0], input_size[1], input_size[0]]])
            self.scale_torch = torch.tensor(self.scale.copy()).float()

        # backbone darknet-53
        self.backbone = darknet53(pretrained=trainable, hr=hr)
        
        # s = 32
        self.conv_set_3 = nn.Sequential(
            Conv2d(1024, 512, 1, leakyReLU=True),
            Conv2d(512, 1024, 3, padding=1, leakyReLU=True),
            Conv2d(1024, 512, 1, leakyReLU=True),
            Conv2d(512, 1024, 3, padding=1, leakyReLU=True),
            Conv2d(1024, 512, 1, leakyReLU=True),
        )
        self.conv_1x1_3 = Conv2d(512, 256, 1, leakyReLU=True)
        self.extra_conv_3 = Conv2d(512, 1024, 3, padding=1, leakyReLU=True)
        self.pred_3 = nn.Conv2d(1024, self.anchor_number*(1 + 4 + self.num_classes), 1)

        # s = 16
        self.conv_set_2 = nn.Sequential(
            Conv2d(768, 256, 1, leakyReLU=True),
            Conv2d(256, 512, 3, padding=1, leakyReLU=True),
            Conv2d(512, 256, 1, leakyReLU=True),
            Conv2d(256, 512, 3, padding=1, leakyReLU=True),
            Conv2d(512, 256, 1, leakyReLU=True),
        )
        self.conv_1x1_2 = Conv2d(256, 128, 1, leakyReLU=True)
        self.extra_conv_2 = Conv2d(256, 512, 3, padding=1, leakyReLU=True)
        self.pred_2 = nn.Conv2d(512, self.anchor_number*(1 + 4 + self.num_classes), 1)

        # s = 8
        self.conv_set_1 = nn.Sequential(
            Conv2d(384, 128, 1, leakyReLU=True),
            Conv2d(128, 256, 3, padding=1, leakyReLU=True),
            Conv2d(256, 128, 1, leakyReLU=True),
            Conv2d(128, 256, 3, padding=1, leakyReLU=True),
            Conv2d(256, 128, 1, leakyReLU=True),
        )
        self.extra_conv_1 = Conv2d(128, 256, 3, padding=1, leakyReLU=True)
        self.pred_1 = nn.Conv2d(256, self.anchor_number*(1 + 4 + self.num_classes), 1)
    
    def init_grid(self, input_size):
        s = self.stride
        total = sum([(self.input_size[1]//s) * (self.input_size[0]//s) for s in self.stride])
        # [1, H*W, 1, 2]
        grid_cell = torch.zeros(1, total, self.anchor_number, 2).to(self.device)
        # [1, 1, anchor_n, 2]
        all_anchor_wh = torch.zeros(1, total, self.anchor_number, 2).to(self.device)

        stride_tensor = torch.zeros(1, total, self.anchor_number).to(self.device).float()

        start_index = 0
        for i, s in enumerate(self.stride):
            ws = self.input_size[1] // s
            hs = self.input_size[0] // s
            for ys in range(hs):
                for xs in range(ws):
                    index = ys * ws + xs + start_index
                    grid_cell[:, index, :, :] = torch.tensor([xs, ys]).float()
                    stride_tensor[:, index, :] = torch.ones(self.anchor_number) * s
            all_anchor_wh[:, start_index:index] = self.anchor_size[i]
            start_index += ws * hs

        return grid_cell, all_anchor_wh, stride_tensor.view(1, -1)
        
    def decode_boxes(self, xywh_pred):
        """
            Input:
                xywh_pred : [B, H*W, anchor_n, 4] containing [tx, ty, tw, th]
            Output:
                bbox_pred : [B, H*W, anchor_n, 4] containing [c_x, c_y, w, h]
        """
        # b_x = sigmoid(tx) + gride_x,  b_y = sigmoid(ty) + gride_y
        B, HW, ab_n, _ = xywh_pred.size()
        c_xy_pred = torch.sigmoid(xywh_pred[:, :, :, :2]) + self.grid_cell
        # b_w = anchor_w * exp(tw),     b_h = anchor_h * exp(th)
        b_wh_pred = torch.exp(xywh_pred[:, :, :, 2:]) * self.all_anchor_wh
        # [B, H*W, anchor_n, 4] -> [B, H*W*anchor_n, 4]
        bbox_pred = torch.cat([c_xy_pred, b_wh_pred], -1).view(B, HW*ab_n, 4)

        # [center_x, center_y, w, h] -> [xmin, ymin, xmax, ymax]
        output = torch.zeros(bbox_pred.size())
        output[:, :, 0] = (bbox_pred[:, :, 0] - bbox_pred[:, :, 2] / 2) * self.stride_tensor
        output[:, :, 1] = (bbox_pred[:, :, 1] - bbox_pred[:, :, 3] / 2) * self.stride_tensor
        output[:, :, 2] = (bbox_pred[:, :, 0] + bbox_pred[:, :, 2] / 2) * self.stride_tensor
        output[:, :, 3] = (bbox_pred[:, :, 1] + bbox_pred[:, :, 3] / 2) * self.stride_tensor
        
        return output

    def clip_boxes(self, boxes, im_shape):
        """
        Clip boxes to image boundaries.
        """
        if boxes.shape[0] == 0:
            return boxes

        # x1 >= 0
        boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
        # y1 >= 0
        boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
        # x2 < im_shape[1]
        boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
        # y2 < im_shape[0]
        boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
        return boxes

    def nms(self, dets, scores):
        """"Pure Python NMS baseline."""
        x1 = dets[:, 0]  #xmin
        y1 = dets[:, 1]  #ymin
        x2 = dets[:, 2]  #xmax
        y2 = dets[:, 3]  #ymax

        areas = (x2 - x1) * (y2 - y1)                 # the size of bbox
        order = scores.argsort()[::-1]                        # sort bounding boxes by decreasing order

        keep = []                                             # store the final bounding boxes
        while order.size > 0:
            i = order[0]                                      #the index of the bbox with highest confidence
            keep.append(i)                                    #save it to keep
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(1e-28, xx2 - xx1)
            h = np.maximum(1e-28, yy2 - yy1)
            inter = w * h

            # Cross Area / (bbox + particular area - Cross Area)
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            #reserve all the boundingbox whose ovr less than thresh
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep

    def postprocess(self, all_local, all_conf, exchange=True, im_shape=None):
        """
        bbox_pred: (HxW*anchor_n, 4), bsize = 1
        prob_pred: (HxW*anchor_n, num_classes), bsize = 1
        """
        bbox_pred = all_local
        prob_pred = all_conf

        cls_inds = np.argmax(prob_pred, axis=1)
        prob_pred = prob_pred[(np.arange(prob_pred.shape[0]), cls_inds)]
        scores = prob_pred.copy()
        
        # threshold
        keep = np.where(scores >= self.conf_thresh)
        bbox_pred = bbox_pred[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        # NMS
        keep = np.zeros(len(bbox_pred), dtype=np.int)
        for i in range(self.num_classes):
            inds = np.where(cls_inds == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bbox_pred[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        bbox_pred = bbox_pred[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        if im_shape != None:
            # clip
            bbox_pred = self.clip_boxes(bbox_pred, im_shape)

        return bbox_pred, scores, cls_inds

    def forward(self, x):
        # backbone
        fmp_1, fmp_2, fmp_3 = self.backbone(x)

        # detection head
        # multi scale feature map fusion
        fmp_3 = self.conv_set_3(fmp_3)
        fmp_3_up = F.interpolate(self.conv_1x1_3(fmp_3), scale_factor=2.0, mode='bilinear', align_corners=True)

        fmp_2 = torch.cat([fmp_2, fmp_3_up], 1)
        fmp_2 = self.conv_set_2(fmp_2)
        fmp_2_up = F.interpolate(self.conv_1x1_2(fmp_2), scale_factor=2.0, mode='bilinear', align_corners=True)

        fmp_1 = torch.cat([fmp_1, fmp_2_up], 1)
        fmp_1 = self.conv_set_1(fmp_1)

        # head
        # s = 32
        fmp_3 = self.extra_conv_3(fmp_3)
        pred_3 = self.pred_3(fmp_3)

        # s = 16
        fmp_2 = self.extra_conv_2(fmp_2)
        pred_2 = self.pred_2(fmp_2)

        # s = 8
        fmp_1 = self.extra_conv_1(fmp_1)
        pred_1 = self.pred_1(fmp_1)

        # fp = self.conv_set(fp)
        # fp = self.branch(fp)
        # prediction = self.pred(fp)

        preds = [pred_1, pred_2, pred_3]
        total_obj_pred = []
        total_cls_pred = []
        total_xywh_pred = []
        B = HW = 0
        for pred in preds:
            B_, abC_, H_, W_ = pred.size()

            # [B, anchor_n * C, H, W] -> [B, H, W, anchor_n * C] -> [B, H*W, anchor_n*C]
            pred = pred.permute(0, 2, 3, 1).contiguous().view(B_, H_*W_, abC_)

            # Divide prediction to obj_pred, xywh_pred and cls_pred   
            # [B, H*W*anchor_n, 1]
            obj_pred = pred[:, :, :1 * self.anchor_number].contiguous().view(B_, H_*W_*self.anchor_number, 1)
            # [B, H*W*anchor_n, num_cls]
            cls_pred = pred[:, :, 1 * self.anchor_number : (1 + self.num_classes) * self.anchor_number].contiguous().view(B_, H_*W_*self.anchor_number, self.num_classes)
            # [B, H*W*anchor_n, 4]
            xywh_pred = pred[:, :, (1 + self.num_classes) * self.anchor_number:].contiguous()

            total_obj_pred.append(obj_pred)
            total_cls_pred.append(cls_pred)
            total_xywh_pred.append(xywh_pred)
            B = B_
            HW += H_*W_
        
        obj_pred = torch.cat(total_obj_pred, 1)
        cls_pred = torch.cat(total_cls_pred, 1)
        xywh_pred = torch.cat(total_xywh_pred, 1)

        # test
        if not self.trainable:
            xywh_pred = xywh_pred.view(B, HW*self.anchor_number, 4).view(B, HW, self.anchor_number, 4)
            with torch.no_grad():
                # batch size = 1                
                all_obj = torch.sigmoid(obj_pred)[0]           # 0 is because that these is only 1 batch.
                all_bbox = self.decode_boxes(xywh_pred)[0] / self.scale_torch
                all_class = (torch.softmax(cls_pred[0, :, :], 1) * all_obj)
                # separate box pred and class conf
                all_obj = all_obj.to('cpu').numpy()
                all_class = all_class.to('cpu').numpy()
                all_bbox = all_bbox.to('cpu').numpy()

                bboxes, scores, cls_inds = self.postprocess(all_bbox, all_class)
                # clip the boxes
                bboxes *= self.scale
                bboxes = self.clip_boxes(bboxes, self.input_size) / self.scale

    
                # print(len(all_boxes))
                return bboxes, scores, cls_inds

        xywh_pred = xywh_pred.view(B, -1, 4)
        final_prediction = torch.cat([obj_pred, cls_pred, xywh_pred], -1)

        return final_prediction
