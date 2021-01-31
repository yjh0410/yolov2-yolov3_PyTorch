"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data import VOCDetection
import sys
import os
import time
import numpy as np
import pickle

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


class VOCAPIEvaluator():
    """ VOC AP Evaluation class """
    def __init__(self, data_root, img_size, device, transform, labelmap, set_type='test', year='2007', display=False):
        self.data_root = data_root
        self.img_size = img_size
        self.device = device
        self.transform = transform
        self.labelmap = labelmap
        self.set_type = set_type
        self.year = year
        self.display = display

        # path
        self.devkit_path = data_root + 'VOC' + year
        self.annopath = os.path.join(data_root, 'VOC2007', 'Annotations', '%s.xml')
        self.imgpath = os.path.join(data_root, 'VOC2007', 'JPEGImages', '%s.jpg')
        self.imgsetpath = os.path.join(data_root, 'VOC2007', 'ImageSets', 'Main', set_type+'.txt')
        self.output_dir = self.get_output_dir('voc_eval/', self.set_type)

        # dataset
        self.dataset = VOCDetection(root=data_root, 
                                    image_sets=[('2007', set_type)],
                                    transform=transform
                                    )

    def evaluate(self, net):
        net.eval()
        num_images = len(self.dataset)
        # all detections are collected into:
        #    all_boxes[cls][image] = N x 5 array of detections in
        #    (x1, y1, x2, y2, score)
        self.all_boxes = [[[] for _ in range(num_images)]
                        for _ in range(len(self.labelmap))]

        # timers
        det_file = os.path.join(self.output_dir, 'detections.pkl')

        for i in range(num_images):
            im, gt, h, w = self.dataset.pull_item(i)

            x = Variable(im.unsqueeze(0)).to(self.device)
            t0 = time.time()
            # forward
            bboxes, scores, cls_inds = net(x)
            detect_time = time.time() - t0
            scale = np.array([[w, h, w, h]])
            bboxes *= scale

            for j in range(len(self.labelmap)):
                inds = np.where(cls_inds == j)[0]
                if len(inds) == 0:
                    self.all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                    continue
                c_bboxes = bboxes[inds]
                c_scores = scores[inds]
                c_dets = np.hstack((c_bboxes,
                                    c_scores[:, np.newaxis])).astype(np.float32,
                                                                    copy=False)
                self.all_boxes[j][i] = c_dets

            if i % 500 == 0:
                print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1, num_images, detect_time))

        with open(det_file, 'wb') as f:
            pickle.dump(self.all_boxes, f, pickle.HIGHEST_PROTOCOL)

        print('Evaluating detections')
        self.evaluate_detections(self.all_boxes)

        print('Mean AP: ', self.map)
  

    def parse_rec(self, filename):
        """ Parse a PASCAL VOC xml file """
        tree = ET.parse(filename)
        objects = []
        for obj in tree.findall('object'):
            obj_struct = {}
            obj_struct['name'] = obj.find('name').text
            obj_struct['pose'] = obj.find('pose').text
            obj_struct['truncated'] = int(obj.find('truncated').text)
            obj_struct['difficult'] = int(obj.find('difficult').text)
            bbox = obj.find('bndbox')
            obj_struct['bbox'] = [int(bbox.find('xmin').text),
                                int(bbox.find('ymin').text),
                                int(bbox.find('xmax').text),
                                int(bbox.find('ymax').text)]
            objects.append(obj_struct)

        return objects


    def get_output_dir(self, name, phase):
        """Return the directory where experimental artifacts are placed.
        If the directory does not exist, it is created.
        A canonical path is built using the name from an imdb and a network
        (if not None).
        """
        filedir = os.path.join(name, phase)
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        return filedir


    def get_voc_results_file_template(self, cls):
        # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
        filename = 'det_' + self.set_type + '_%s.txt' % (cls)
        filedir = os.path.join(self.devkit_path, 'results')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path


    def write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.labelmap):
            if self.display:
                print('Writing {:s} VOC results file'.format(cls))
            filename = self.get_voc_results_file_template(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.dataset.ids):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index[1], dets[k, -1],
                                    dets[k, 0] + 1, dets[k, 1] + 1,
                                    dets[k, 2] + 1, dets[k, 3] + 1))


    def do_python_eval(self, use_07=True):
        cachedir = os.path.join(self.devkit_path, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = use_07
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)
        for i, cls in enumerate(self.labelmap):
            filename = self.get_voc_results_file_template(cls)
            rec, prec, ap = self.voc_eval(detpath=filename, 
                                          classname=cls, 
                                          cachedir=cachedir, 
                                          ovthresh=0.5, 
                                          use_07_metric=use_07_metric
                                        )
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(self.output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        if self.display:
            self.map = np.mean(aps)
            print('Mean AP = {:.4f}'.format(np.mean(aps)))
            print('~~~~~~~~')
            print('Results:')
            for ap in aps:
                print('{:.3f}'.format(ap))
            print('{:.3f}'.format(np.mean(aps)))
            print('~~~~~~~~')
            print('')
            print('--------------------------------------------------------------')
            print('Results computed with the **unofficial** Python eval code.')
            print('Results should be very close to the official MATLAB eval code.')
            print('--------------------------------------------------------------')
        else:
            self.map = np.mean(aps)
            print('Mean AP = {:.4f}'.format(np.mean(aps)))


    def voc_ap(self, rec, prec, use_07_metric=True):
        """ ap = voc_ap(rec, prec, [use_07_metric])
        Compute VOC AP given precision and recall.
        If use_07_metric is true, uses the
        VOC 07 11 point method (default:True).
        """
        if use_07_metric:
            # 11 point metric
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap = ap + p / 11.
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))

            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap


    def voc_eval(self, detpath, classname, cachedir, ovthresh=0.5, use_07_metric=True):
        if not os.path.isdir(cachedir):
            os.mkdir(cachedir)
        cachefile = os.path.join(cachedir, 'annots.pkl')
        # read list of images
        with open(self.imgsetpath, 'r') as f:
            lines = f.readlines()
        imagenames = [x.strip() for x in lines]
        if not os.path.isfile(cachefile):
            # load annots
            recs = {}
            for i, imagename in enumerate(imagenames):
                recs[imagename] = self.parse_rec(self.annopath % (imagename))
                if i % 100 == 0 and self.display:
                    print('Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames)))
            # save
            if self.display:
                print('Saving cached annotations to {:s}'.format(cachefile))
            with open(cachefile, 'wb') as f:
                pickle.dump(recs, f)
        else:
            # load
            with open(cachefile, 'rb') as f:
                recs = pickle.load(f)

        # extract gt objects for this class
        class_recs = {}
        npos = 0
        for imagename in imagenames:
            R = [obj for obj in recs[imagename] if obj['name'] == classname]
            bbox = np.array([x['bbox'] for x in R])
            difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
            det = [False] * len(R)
            npos = npos + sum(~difficult)
            class_recs[imagename] = {'bbox': bbox,
                                    'difficult': difficult,
                                    'det': det}

        # read dets
        detfile = detpath.format(classname)
        with open(detfile, 'r') as f:
            lines = f.readlines()
        if any(lines) == 1:

            splitlines = [x.strip().split(' ') for x in lines]
            image_ids = [x[0] for x in splitlines]
            confidence = np.array([float(x[1]) for x in splitlines])
            BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

            # sort by confidence
            sorted_ind = np.argsort(-confidence)
            sorted_scores = np.sort(-confidence)
            BB = BB[sorted_ind, :]
            image_ids = [image_ids[x] for x in sorted_ind]

            # go down dets and mark TPs and FPs
            nd = len(image_ids)
            tp = np.zeros(nd)
            fp = np.zeros(nd)
            for d in range(nd):
                R = class_recs[image_ids[d]]
                bb = BB[d, :].astype(float)
                ovmax = -np.inf
                BBGT = R['bbox'].astype(float)
                if BBGT.size > 0:
                    # compute overlaps
                    # intersection
                    ixmin = np.maximum(BBGT[:, 0], bb[0])
                    iymin = np.maximum(BBGT[:, 1], bb[1])
                    ixmax = np.minimum(BBGT[:, 2], bb[2])
                    iymax = np.minimum(BBGT[:, 3], bb[3])
                    iw = np.maximum(ixmax - ixmin, 0.)
                    ih = np.maximum(iymax - iymin, 0.)
                    inters = iw * ih
                    uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                        (BBGT[:, 2] - BBGT[:, 0]) *
                        (BBGT[:, 3] - BBGT[:, 1]) - inters)
                    overlaps = inters / uni
                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)

                if ovmax > ovthresh:
                    if not R['difficult'][jmax]:
                        if not R['det'][jmax]:
                            tp[d] = 1.
                            R['det'][jmax] = 1
                        else:
                            fp[d] = 1.
                else:
                    fp[d] = 1.

            # compute precision recall
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / float(npos)
            # avoid divide by zero in case the first detection matches a difficult
            # ground truth
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            ap = self.voc_ap(rec, prec, use_07_metric)
        else:
            rec = -1.
            prec = -1.
            ap = -1.

        return rec, prec, ap


    def evaluate_detections(self, box_list):
        self.write_voc_results_file(box_list)
        self.do_python_eval()


if __name__ == '__main__':
    pass