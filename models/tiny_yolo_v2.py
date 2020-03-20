import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models
from utils import Conv2d, reorg_layer
from backbone import *
import os
import numpy as np
import tools

class myYOLOv2(nn.Module):
    def __init__(self, device, input_size=None, num_classes=20, trainable=False, conf_thresh=0.01, nms_thresh=0.5, anchor_size=None, hr=False):
        super(myYOLOv2, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.anchor_size = torch.tensor(anchor_size)
        self.anchor_number = len(anchor_size)
        self.stride = 32
        # if not trainable:
        self.grid_cell, self.all_anchor_wh = self.set_init(input_size)
        self.input_size = input_size
        self.scale = np.array([[input_size[1], input_size[0], input_size[1], input_size[0]]])
        self.scale_torch = torch.tensor(self.scale.copy()).float()

        # backbone darknet-19
        self.backbone = darknet_tiny(pretrained=trainable, hr=hr)
        
        # detection head
        self.convsets_1 = nn.Sequential(
            Conv2d(512, 256, 1, leakyReLU=True),
            Conv2d(256, 512, 3, 1, leakyReLU=True)
        )

        self.route_layer = Conv2d(256, 32, 1, leakyReLU=True)
        self.reorg = reorg_layer(stride=2)

        self.convsets_2 = Conv2d(640, 512, 3, 1, leakyReLU=True)
        
        # prediction layer
        self.pred_ = nn.Conv2d(512, self.anchor_number*(1 + 4 + self.num_classes), 1)

    def set_init(self, input_size):
        s = self.stride
        ws = input_size[1] // s
        hs = input_size[0] // s
        # [1, H*W, 1, 2]
        grid_cell = torch.zeros(1, hs*ws, self.anchor_number, 2).to(self.device)
        # [1, 1, anchor_n, 2]
        all_anchor_wh = torch.zeros(1, hs*ws, self.anchor_number, 2).to(self.device)
        
        for ys in range(hs):
            for xs in range(ws):
                index = ys * ws + xs
                grid_cell[:, index, :, :] = torch.tensor([xs, ys]).float()
        all_anchor_wh[:, :] = self.anchor_size

        return grid_cell, all_anchor_wh
        
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
        # [H*W, anchor_n, 4] -> [H*W*anchor_n, 4]
        bbox_pred = torch.cat([c_xy_pred, b_wh_pred], -1).view(B, HW*ab_n, 4)

        # [center_x, center_y, w, h] -> [xmin, ymin, xmax, ymax]
        output = torch.zeros(bbox_pred.size())
        output[:, :, 0] = (bbox_pred[:, :, 0] - bbox_pred[:, :, 2] / 2) * self.stride
        output[:, :, 1] = (bbox_pred[:, :, 1] - bbox_pred[:, :, 3] / 2) * self.stride
        output[:, :, 2] = (bbox_pred[:, :, 0] + bbox_pred[:, :, 2] / 2) * self.stride
        output[:, :, 3] = (bbox_pred[:, :, 1] + bbox_pred[:, :, 3] / 2) * self.stride
        
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
        _, fp_1, fp_2 = self.backbone(x)

        # head
        fp_2 = self.convsets_1(fp_2)

        # route from 16th layer in darknet
        fp_1 = self.reorg(self.route_layer(fp_1))

        # route concatenate
        fp = torch.cat([fp_1, fp_2], dim=1)
        fp = self.convsets_2(fp)
        prediction = self.pred_(fp)

        B, abC, H, W = prediction.size()

        # [B, anchor_n * C, H, W] -> [B, H, W, anchor_n * C] -> [B, H*W, anchor_n*C]
        prediction = prediction.permute(0, 2, 3, 1).contiguous().view(B, H*W, abC)

        # Divide prediction to obj_pred, xywh_pred and cls_pred   
        # [B, H*W*anchor_n, 1]
        obj_pred = prediction[:, :, :1 * self.anchor_number].contiguous().view(B, H*W*self.anchor_number, 1)
        # [B, H*W, anchor_n, num_cls]
        cls_pred = prediction[:, :, 1 * self.anchor_number : (1 + self.num_classes) * self.anchor_number].contiguous().view(B, H*W*self.anchor_number, self.num_classes)
        # [B, H*W, anchor_n, 4]
        xywh_pred = prediction[:, :, (1 + self.num_classes) * self.anchor_number:].contiguous()
        
        # test
        if not self.trainable:
            xywh_pred = xywh_pred.view(B, H*W, self.anchor_number, 4)
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

        xywh_pred = xywh_pred.view(B, H*W*self.anchor_number, 4)
        final_prediction = torch.cat([obj_pred, cls_pred, xywh_pred], -1)

        return final_prediction
