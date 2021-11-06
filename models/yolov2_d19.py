import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.modules import Conv, reorg_layer
from backbone import *
import numpy as np
import tools

class YOLOv2D19(nn.Module):
    def __init__(self, device, input_size=None, num_classes=20, trainable=False, conf_thresh=0.001, nms_thresh=0.5, anchor_size=None, hr=False):
        super(YOLOv2D19, self).__init__()
        self.device = device
        self.input_size = input_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.anchor_size = torch.tensor(anchor_size)
        self.num_anchors = len(anchor_size)
        self.stride = 32
        self.grid_cell, self.all_anchor_wh = self.create_grid(input_size)

        # backbone darknet-19
        self.backbone = darknet19(pretrained=trainable, hr=hr)
        
        # detection head
        self.convsets_1 = nn.Sequential(
            Conv(1024, 1024, k=3, p=1),
            Conv(1024, 1024, k=3, p=1)
        )

        self.route_layer = Conv(512, 64, k=1)
        self.reorg = reorg_layer(stride=2)

        self.convsets_2 = Conv(1280, 1024, k=3, p=1)
        
        # prediction layer
        self.pred = nn.Conv2d(1024, self.num_anchors*(1 + 4 + self.num_classes), kernel_size=1)


    def create_grid(self, input_size):
        w, h = input_size, input_size
        # generate grid cells
        ws, hs = w // self.stride, h // self.stride
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
        grid_xy = grid_xy.view(1, hs*ws, 1, 2).to(self.device)

        # generate anchor_wh tensor
        anchor_wh = self.anchor_size.repeat(hs*ws, 1, 1).unsqueeze(0).to(self.device)

        return grid_xy, anchor_wh


    def set_grid(self, input_size):
        self.input_size = input_size
        self.grid_cell, self.all_anchor_wh = self.create_grid(input_size)


    def decode_xywh(self, txtytwth_pred):
        """
            Input:
                txtytwth_pred : [B, H*W, anchor_n, 4] containing [tx, ty, tw, th]
            Output:
                xywh_pred : [B, H*W*anchor_n, 4] containing [xmin, ymin, xmax, ymax]
        """
        # b_x = sigmoid(tx) + gride_x,  b_y = sigmoid(ty) + gride_y
        B, HW, ab_n, _ = txtytwth_pred.size()
        xy_pred = torch.sigmoid(txtytwth_pred[:, :, :, :2]) + self.grid_cell
        # b_w = anchor_w * exp(tw),     b_h = anchor_h * exp(th)
        wh_pred = torch.exp(txtytwth_pred[:, :, :, 2:]) * self.all_anchor_wh
        # [H*W, anchor_n, 4] -> [H*W*anchor_n, 4]
        xywh_pred = torch.cat([xy_pred, wh_pred], -1).view(B, HW*ab_n, 4) * self.stride

        return xywh_pred


    def decode_boxes(self, txtytwth_pred, requires_grad=False):
        """
            Input:
                txtytwth_pred : [B, H*W, anchor_n, 4] containing [tx, ty, tw, th]
            Output:
                x1y1x2y2_pred : [B, H*W*anchor_n, 4] containing [xmin, ymin, xmax, ymax]
        """
        # [H*W*anchor_n, 4]
        xywh_pred = self.decode_xywh(txtytwth_pred)

        # [center_x, center_y, w, h] -> [xmin, ymin, xmax, ymax]
        x1y1x2y2_pred = torch.zeros_like(xywh_pred, requires_grad=requires_grad)
        x1y1x2y2_pred[:, :, 0] = (xywh_pred[:, :, 0] - xywh_pred[:, :, 2] / 2)
        x1y1x2y2_pred[:, :, 1] = (xywh_pred[:, :, 1] - xywh_pred[:, :, 3] / 2)
        x1y1x2y2_pred[:, :, 2] = (xywh_pred[:, :, 0] + xywh_pred[:, :, 2] / 2)
        x1y1x2y2_pred[:, :, 3] = (xywh_pred[:, :, 1] + xywh_pred[:, :, 3] / 2)
        
        return x1y1x2y2_pred


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


    def postprocess(self, bboxes, scores):
        """
        bboxes: (HxW, 4), bsize = 1
        scores: (HxW, num_classes), bsize = 1
        """

        cls_inds = np.argmax(scores, axis=1)
        scores = scores[(np.arange(scores.shape[0]), cls_inds)]
        
        # threshold
        keep = np.where(scores >= self.conf_thresh)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        # NMS
        keep = np.zeros(len(bboxes), dtype=np.int)
        for i in range(self.num_classes):
            inds = np.where(cls_inds == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        return bboxes, scores, cls_inds


    def forward(self, x, target=None):
        # backbone主干网络
        _, c4, c5 = self.backbone(x)

        # head
        p5 = self.convsets_1(c5)

        # 处理c4特征
        p4 = self.reorg(self.route_layer(c4))

        # 融合
        p5 = torch.cat([p4, p5], dim=1)

        # head
        p5 = self.convsets_2(p5)

        # 预测
        prediction = self.pred(p5)

        B, abC, H, W = prediction.size()

        # [B, num_anchor * C, H, W] -> [B, H, W, num_anchor * C] -> [B, H*W, num_anchor*C]
        prediction = prediction.permute(0, 2, 3, 1).contiguous().view(B, H*W, abC)

        # 从pred中分离出objectness预测、类别class预测、bbox的txtytwth预测  
        # [B, H*W*num_anchor, 1]
        conf_pred = prediction[:, :, :1 * self.num_anchors].contiguous().view(B, H*W*self.num_anchors, 1)
        # [B, H*W, num_anchor, num_cls]
        cls_pred = prediction[:, :, 1 * self.num_anchors : (1 + self.num_classes) * self.num_anchors].contiguous().view(B, H*W*self.num_anchors, self.num_classes)
        # [B, H*W, num_anchor, 4]
        txtytwth_pred = prediction[:, :, (1 + self.num_classes) * self.num_anchors:].contiguous()
        
        # train
        if self.trainable:
            txtytwth_pred = txtytwth_pred.view(B, H*W, self.num_anchors, 4)
            # decode bbox
            x1y1x2y2_pred = (self.decode_boxes(txtytwth_pred) / self.input_size).view(-1, 4)
            x1y1x2y2_gt = target[:, :, 7:].view(-1, 4)

            # 计算预测框和真实框之间的IoU
            iou_pred = tools.iou_score(x1y1x2y2_pred, x1y1x2y2_gt).view(B, -1, 1)

            # 将IoU作为置信度的学习目标
            with torch.no_grad():
                gt_conf = iou_pred.clone()

            txtytwth_pred = txtytwth_pred.view(B, H*W*self.num_anchors, 4)
            # 将IoU作为置信度的学习目标 
            # [obj, cls, txtytwth, x1y1x2y2] -> [conf, obj, cls, txtytwth]
            target = torch.cat([gt_conf, target[:, :, :7]], dim=2)

            # 计算损失
            conf_loss, cls_loss, bbox_loss, iou_loss = tools.loss(pred_conf=conf_pred, 
                                                                  pred_cls=cls_pred,
                                                                  pred_txtytwth=txtytwth_pred,
                                                                  pred_iou=iou_pred,
                                                                  label=target
                                                                  )

            return conf_loss, cls_loss, bbox_loss, iou_loss   

        # test
        else:
            txtytwth_pred = txtytwth_pred.view(B, H*W, self.num_anchors, 4)
            with torch.no_grad():
                # batch size = 1
                # 测试时，笔者默认batch是1，
                # 因此，我们不需要用batch这个维度，用[0]将其取走。
                # [B, H*W*num_anchor, 1] -> [H*W*num_anchor, 1]
                conf_pred = torch.sigmoid(conf_pred)[0]
                # [B, H*W*num_anchor, 4] -> [H*W*num_anchor, 4]
                bboxes = torch.clamp((self.decode_boxes(txtytwth_pred) / self.input_size)[0], 0., 1.)
                # [B, H*W*num_anchor, C] -> [H*W*num_anchor, C], 
                scores = torch.softmax(cls_pred[0, :, :], dim=1) * conf_pred

                # 将预测放在cpu处理上，以便进行后处理
                scores = scores.to('cpu').numpy()
                bboxes = bboxes.to('cpu').numpy()

                # 后处理
                bboxes, scores, cls_inds = self.postprocess(bboxes, scores)

                return bboxes, scores, cls_inds
