import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.modules import Conv, SPP
from backbone import build_backbone
import tools


# YOLOv3 SPP
class YOLOv3Spp(nn.Module):
    def __init__(self,
                 device,
                 input_size=None,
                 num_classes=20,
                 trainable=False,
                 conf_thresh=0.001,
                 nms_thresh=0.50,
                 anchor_size=None):
        super(YOLOv3Spp, self).__init__()
        self.device = device
        self.input_size = input_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.stride = [8, 16, 32]
        self.anchor_size = torch.tensor(anchor_size).view(3, len(anchor_size) // 3, 2)
        self.num_anchors = self.anchor_size.size(1)

        self.grid_cell, self.stride_tensor, self.all_anchors_wh = self.create_grid(input_size)

        # backbone
        self.backbone = build_backbone(model_name='darknet53', pretrained=trainable)
        
        # s = 32
        self.conv_set_3 = nn.Sequential(
            SPP(),
            Conv(1024*4, 512, k=1),
            Conv(512, 1024, k=3, p=1),
            Conv(1024, 512, k=1),
            Conv(512, 1024, k=3, p=1),
            Conv(1024, 512, k=1)
        )
        self.conv_1x1_3 = Conv(512, 256, k=1)
        self.extra_conv_3 = Conv(512, 1024, k=3, p=1)
        self.pred_3 = nn.Conv2d(1024, self.num_anchors*(1 + 4 + self.num_classes), kernel_size=1)

        # s = 16
        self.conv_set_2 = nn.Sequential(
            Conv(768, 256, k=1),
            Conv(256, 512, k=3, p=1),
            Conv(512, 256, k=1),
            Conv(256, 512, k=3, p=1),
            Conv(512, 256, k=1)
        )
        self.conv_1x1_2 = Conv(256, 128, k=1)
        self.extra_conv_2 = Conv(256, 512, k=3, p=1)
        self.pred_2 = nn.Conv2d(512, self.num_anchors*(1 + 4 + self.num_classes), kernel_size=1)

        # s = 8
        self.conv_set_1 = nn.Sequential(
            Conv(384, 128, k=1),
            Conv(128, 256, k=3, p=1),
            Conv(256, 128, k=1),
            Conv(128, 256, k=3, p=1),
            Conv(256, 128, k=1)
        )
        self.extra_conv_1 = Conv(128, 256, k=3, p=1)
        self.pred_1 = nn.Conv2d(256, self.num_anchors*(1 + 4 + self.num_classes), kernel_size=1)

    
        self.init_yolo()


    def init_yolo(self):  
        # Init head
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        # init obj&cls pred
        for pred in [self.pred_1, self.pred_2, self.pred_3]:
            nn.init.constant_(pred.bias[..., :self.num_anchors], bias_value)
            nn.init.constant_(pred.bias[..., self.num_anchors : (1 + self.num_classes) * self.num_anchors], bias_value)


    def create_grid(self, input_size):
        total_grid_xy = []
        total_stride = []
        total_anchor_wh = []
        w, h = input_size, input_size
        for ind, s in enumerate(self.stride):
            # generate grid cells
            ws, hs = w // s, h // s
            grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
            grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
            grid_xy = grid_xy.view(1, hs*ws, 1, 2)

            # generate stride tensor
            stride_tensor = torch.ones([1, hs*ws, self.num_anchors, 2]) * s

            # generate anchor_wh tensor
            anchor_wh = self.anchor_size[ind].repeat(hs*ws, 1, 1)

            total_grid_xy.append(grid_xy)
            total_stride.append(stride_tensor)
            total_anchor_wh.append(anchor_wh)

        total_grid_xy = torch.cat(total_grid_xy, dim=1).to(self.device)
        total_stride = torch.cat(total_stride, dim=1).to(self.device)
        total_anchor_wh = torch.cat(total_anchor_wh, dim=0).to(self.device).unsqueeze(0)

        return total_grid_xy, total_stride, total_anchor_wh


    def set_grid(self, input_size):
        self.input_size = input_size
        self.grid_cell, self.stride_tensor, self.all_anchors_wh = self.create_grid(input_size)


    def decode_xywh(self, txtytwth_pred):
        """
            Input:
                txtytwth_pred : [B, H*W, anchor_n, 4] containing [tx, ty, tw, th]
            Output:
                xywh_pred : [B, H*W*anchor_n, 4] containing [x, y, w, h]
        """
        # b_x = sigmoid(tx) + gride_x,  b_y = sigmoid(ty) + gride_y
        B, HW, ab_n, _ = txtytwth_pred.size()
        c_xy_pred = (torch.sigmoid(txtytwth_pred[:, :, :, :2]) + self.grid_cell) * self.stride_tensor
        # b_w = anchor_w * exp(tw),     b_h = anchor_h * exp(th)
        b_wh_pred = torch.exp(txtytwth_pred[:, :, :, 2:]) * self.all_anchors_wh
        # [B, H*W, anchor_n, 4] -> [B, H*W*anchor_n, 4]
        xywh_pred = torch.cat([c_xy_pred, b_wh_pred], -1).view(B, HW*ab_n, 4)

        return xywh_pred


    def decode_boxes(self, txtytwth_pred):
        """
            Input: \n
                txtytwth_pred : [B, H*W, anchor_n, 4] \n
            Output: \n
                x1y1x2y2_pred : [B, H*W*anchor_n, 4] \n
        """
        # txtytwth -> cxcywh
        xywh_pred = self.decode_xywh(txtytwth_pred)

        # cxcywh -> x1y1x2y2
        x1y1x2y2_pred = torch.zeros_like(xywh_pred)
        x1y1_pred = xywh_pred[..., :2] - xywh_pred[..., 2:] * 0.5
        x2y2_pred = xywh_pred[..., :2] + xywh_pred[..., 2:] * 0.5
        x1y1x2y2_pred = torch.cat([x1y1_pred, x2y2_pred], dim=-1)
        
        return x1y1x2y2_pred


    def nms(self, dets, scores):
        """"Pure Python NMS baseline."""
        x1 = dets[:, 0]  #xmin
        y1 = dets[:, 1]  #ymin
        x2 = dets[:, 2]  #xmax
        y2 = dets[:, 3]  #ymax

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(1e-10, xx2 - xx1)
            h = np.maximum(1e-10, yy2 - yy1)
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


    @torch.no_grad()
    def inference(self, x):
        B = x.size(0)
        # backbone
        feats = self.backbone(x)
        c3, c4, c5 = feats['layer1'], feats['layer2'], feats['layer3']

        # FPN
        p5 = self.conv_set_3(c5)
        p5_up = F.interpolate(self.conv_1x1_3(p5), scale_factor=2.0, mode='bilinear', align_corners=True)

        p4 = torch.cat([c4, p5_up], 1)
        p4 = self.conv_set_2(p4)
        p4_up = F.interpolate(self.conv_1x1_2(p4), scale_factor=2.0, mode='bilinear', align_corners=True)

        p3 = torch.cat([c3, p4_up], 1)
        p3 = self.conv_set_1(p3)

        # head
        # s = 32
        p5 = self.extra_conv_3(p5)
        pred_3 = self.pred_3(p5)

        # s = 16
        p4 = self.extra_conv_2(p4)
        pred_2 = self.pred_2(p4)

        # s = 8
        p3 = self.extra_conv_1(p3)
        pred_1 = self.pred_1(p3)

        preds = [pred_1, pred_2, pred_3]
        total_conf_pred = []
        total_cls_pred = []
        total_reg_pred = []
        for pred in preds:
            C = pred.size(1)

            # [B, anchor_n * C, H, W] -> [B, H, W, anchor_n * C] -> [B, H*W, anchor_n*C]
            pred = pred.permute(0, 2, 3, 1).contiguous().view(B, -1, C)

            # [B, H*W*anchor_n, 1]
            conf_pred = pred[:, :, :1 * self.num_anchors].contiguous().view(B, -1, 1)
            # [B, H*W*anchor_n, num_cls]
            cls_pred = pred[:, :, 1 * self.num_anchors : (1 + self.num_classes) * self.num_anchors].contiguous().view(B, -1, self.num_classes)
            # [B, H*W*anchor_n, 4]
            reg_pred = pred[:, :, (1 + self.num_classes) * self.num_anchors:].contiguous()

            total_conf_pred.append(conf_pred)
            total_cls_pred.append(cls_pred)
            total_reg_pred.append(reg_pred)
        
        conf_pred = torch.cat(total_conf_pred, dim=1)
        cls_pred = torch.cat(total_cls_pred, dim=1)
        reg_pred = torch.cat(total_reg_pred, dim=1)
        # decode bbox
        reg_pred = reg_pred.view(B, -1, self.num_anchors, 4)
        box_pred = self.decode_boxes(reg_pred)

        # batch size = 1
        conf_pred = conf_pred[0]
        cls_pred = cls_pred[0]
        box_pred = box_pred[0]

        # score
        scores = torch.sigmoid(conf_pred) * torch.softmax(cls_pred, dim=-1)

        # normalize bbox
        bboxes = torch.clamp(box_pred / self.input_size, 0., 1.)

        # to cpu
        scores = scores.to('cpu').numpy()
        bboxes = bboxes.to('cpu').numpy()

        # post-process
        bboxes, scores, cls_inds = self.postprocess(bboxes, scores)

        return bboxes, scores, cls_inds
        

    def forward(self, x, target=None):
        if not self.trainable:
            return self.inference(x)
        else:
            # backbone
            B = x.size(0)
            # backbone
            feats = self.backbone(x)
            c3, c4, c5 = feats['layer1'], feats['layer2'], feats['layer3']

            # FPN
            p5 = self.conv_set_3(c5)
            p5_up = F.interpolate(self.conv_1x1_3(p5), scale_factor=2.0, mode='bilinear', align_corners=True)

            p4 = torch.cat([c4, p5_up], 1)
            p4 = self.conv_set_2(p4)
            p4_up = F.interpolate(self.conv_1x1_2(p4), scale_factor=2.0, mode='bilinear', align_corners=True)

            p3 = torch.cat([c3, p4_up], 1)
            p3 = self.conv_set_1(p3)

            # head
            # s = 32
            p5 = self.extra_conv_3(p5)
            pred_3 = self.pred_3(p5)

            # s = 16
            p4 = self.extra_conv_2(p4)
            pred_2 = self.pred_2(p4)

            # s = 8
            p3 = self.extra_conv_1(p3)
            pred_1 = self.pred_1(p3)

            preds = [pred_1, pred_2, pred_3]
            total_conf_pred = []
            total_cls_pred = []
            total_reg_pred = []
            for pred in preds:
                C = pred.size(1)

                # [B, anchor_n * C, H, W] -> [B, H, W, anchor_n * C] -> [B, H*W, anchor_n*C]
                pred = pred.permute(0, 2, 3, 1).contiguous().view(B, -1, C)

                # [B, H*W*anchor_n, 1]
                conf_pred = pred[:, :, :1 * self.num_anchors].contiguous().view(B, -1, 1)
                # [B, H*W*anchor_n, num_cls]
                cls_pred = pred[:, :, 1 * self.num_anchors : (1 + self.num_classes) * self.num_anchors].contiguous().view(B, -1, self.num_classes)
                # [B, H*W*anchor_n, 4]
                reg_pred = pred[:, :, (1 + self.num_classes) * self.num_anchors:].contiguous()

                total_conf_pred.append(conf_pred)
                total_cls_pred.append(cls_pred)
                total_reg_pred.append(reg_pred)
            
            conf_pred = torch.cat(total_conf_pred, dim=1)
            cls_pred = torch.cat(total_cls_pred, dim=1)
            reg_pred = torch.cat(total_reg_pred, dim=1)

            # decode bbox
            reg_pred = reg_pred.view(B, -1, self.num_anchors, 4)
            x1y1x2y2_pred = (self.decode_boxes(reg_pred) / self.input_size).view(-1, 4)
            reg_pred = reg_pred.view(B, -1, 4)
            x1y1x2y2_gt = target[:, :, 7:].view(-1, 4)
                
            # set conf target
            iou_pred = tools.iou_score(x1y1x2y2_pred, x1y1x2y2_gt).view(B, -1, 1)
            gt_conf = iou_pred.clone().detach()

            # [obj, cls, txtytwth, scale_weight, x1y1x2y2] -> [conf, obj, cls, txtytwth, scale_weight]
            target = torch.cat([gt_conf, target[:, :, :7]], dim=2)

            # loss
            (
                conf_loss,
                cls_loss,
                bbox_loss,
                iou_loss
            ) = tools.loss(pred_conf=conf_pred,
                            pred_cls=cls_pred,
                            pred_txtytwth=reg_pred,
                            pred_iou=iou_pred,
                            label=target
                            )

            return conf_loss, cls_loss, bbox_loss, iou_loss   
