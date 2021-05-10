import torch
import numpy as np
import pickle
import os
import cv2
from torch.autograd import Variable
import sys

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


class MAPEvaluator(object):
    """
    MAP Evaluation class.
    """
    def __init__(self, device, dataset=None, classname=[], ovthresh=0.5, name='Our', display=False):
        self.device = device
        self.dataset = dataset
        self.classnames = classname
        self.ovthresh = ovthresh
        self.devkit_path = os.path.join('mAP_results', name)
        self.set_type = 'test'
        self.mAP = 0.
        self.display = display
        self.name = name
        self.num_datas = len(self.dataset)

        # all detections are collected into:
        #    all_boxes[cls][image] = N x 5 array of detections in
        #    (x1, y1, x2, y2, score)
        self.all_boxes = [[[] for _ in range(self.num_datas)]
                for _ in range(len(self.classnames))]

        self.output_dir = self.get_output_dir('our_ap_eval/', 'test')
        self.det_file = os.path.join(self.output_dir, 'detections.pkl')


    def evaluate(self, model, use_07_metric=True):
        """
        MAP Evaluation. Iterate inference on the test dataset and the results are evaluated with mAP metric.
        Compute AP given precision and recall.
        If use_07_metric is true, uses the VOC 07 11 point method.
        """
        model.eval()

        for i in range(self.num_datas):
            if i % 500 == 0:
                print('[%d] / [%d]' % (i, self.num_datas))
            im, _, h, w = self.dataset.pull_item(i)

            x = Variable(im.unsqueeze(0)).to(self.device)
            # inference
            bboxes, scores, cls_inds = model(x)
            # scale each detection back up to the image
            scale = np.array([[w, h, w, h]])
            # map the boxes to origin image scale
            bboxes *= scale

            for j in range(len(self.classnames)):
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
                                
        with open(self.det_file, 'wb') as f:
            pickle.dump(self.all_boxes, f, pickle.HIGHEST_PROTOCOL)

        print('Evaluating detections')
        self.evaluate_detections(self.all_boxes, self.output_dir)


    def parse_rec(self, filename):
        """ Parse a PASCAL VOC-style xml file """
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
        filedir = os.path.join(name, self.name, phase)
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
        for cls_ind, cls_name in enumerate(self.classnames):
            if self.display:
                print('Writing {:s} VOC results file'.format(cls_name))
            filename = self.get_voc_results_file_template(cls_name)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.dataset.img_names_list):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:s}, {:.3f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}\n'.
                                format(index, dets[k, -1],
                                    dets[k, 0] + 1, dets[k, 1] + 1,
                                    dets[k, 2] + 1, dets[k, 3] + 1))


    def voc_eval(self, 
                detpath,
                classname,
                cachedir,
                ovthresh=0.5,
                use_07_metric=True):
        if not os.path.isdir(cachedir):
            os.mkdir(cachedir)
        cachefile = os.path.join(cachedir, 'annots.pkl')
        
        # read list of images
        imagenames = self.dataset.img_names_list
        
        if not os.path.isfile(cachefile):
            # load annots
            recs = {}
            for i, imagename in enumerate(imagenames):
                target_id = self.dataset.target_names_list[i]
                recs[imagename] = self.parse_rec(self.dataset._annopath % (target_id))
                if i % 100 == 0:
                    print('Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames)))
            # save
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
        for filename in imagenames:
            R = [obj for obj in recs[filename] if obj['name'] == classname]
            bbox = np.array([x['bbox'] for x in R])
            difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
            det = [False] * len(R)
            npos = npos + sum(~difficult)
            class_recs[filename] = {'bbox': bbox,
                                    'difficult': difficult,
                                    'det': det}

        # read dets
        detfile = detpath #.format(classname)
        with open(detfile, 'r') as f:
            lines = f.readlines()
        if any(lines) == 1:

            splitlines = [x.strip().split(',') for x in lines]
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


    def do_python_eval(self, output_dir='output', use_07=True):
        cachedir = os.path.join(self.devkit_path, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = use_07
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        for i, cls_name in enumerate(self.classnames):
            filename = self.get_voc_results_file_template(cls_name)
            rec, prec, ap = self.voc_eval(detpath=filename, 
                                          classname=cls_name, 
                                          cachedir=cachedir,
                                          ovthresh=self.ovthresh, 
                                          use_07_metric=use_07_metric
                                          )
            aps += [ap]
            
            if self.display:
                print('AP for {} = {:.4f}'.format(cls_name, ap))

            with open(os.path.join(output_dir, cls_name + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)

        Mean_AP = np.mean(aps)
        self.mAP = Mean_AP
        print('Mean AP = {:.4f}'.format(self.mAP))
        
        
        if self.display:
            print('Mean AP = {:.4f}'.format(Mean_AP))
            print('~~~~~~~~')
            print('Results:')
            for ap in aps:
                print('{:.3f}'.format(ap))
            print('{:.3f}'.format(Mean_AP))
            print('~~~~~~~~')
            print('')
            print('--------------------------------------------------------------')
            print('Results computed with the **unofficial** Python eval code.')
            print('Results should be very close to the official MATLAB eval code.')
            print('--------------------------------------------------------------')


    def voc_ap(self, rec, prec, use_07_metric=True):
        """
        MAP Evaluation. Iterate inference on the test dataset and the results are evaluated with mAP metric.
        Compute AP given precision and recall.
        If use_07_metric is true, uses the VOC 07 11 point method.
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


    def evaluate_detections(self, box_list, output_dir):
        self.write_voc_results_file(box_list)
        self.do_python_eval(output_dir)



if __name__ == "__main__":
    pass