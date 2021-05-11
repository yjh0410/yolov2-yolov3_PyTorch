# I'm ready to provide a dataset file to load our own dataset.
# Attention, I only consider about VOC-style dataset.
# Please try to use the Labelimg to build your dataset.
# 过段时间，我会提供一个用来读取我们自己数据集的文件。
# 请注意，这个dataset文件只读取VOC风格的数据集。
# 请使用Labelimg工具来建立你的数据集

import os.path as osp
import os
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import random
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


CLASSES = ('Azusa', 'Yui', 'Mio', 'Ritsu', 'Mugi') # no background label
DATA_ROOT = "/home/jxk/object-detection/dataset/KonFace/"


class AnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None):
        self.class_to_ind = class_to_ind or dict(
            zip(CLASSES, range(len(CLASSES))))

    def __call__(self, target_id, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            name = obj.find('name').text.strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)

            # check label
            try:
                label_idx = self.class_to_ind[name]
            except:
                print('The label %s is wrong, please check label file %s ...' % (name, target_id))
                continue

            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class OurDetection(data.Dataset):
    """Our Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, 
                 root, 
                 img_size=None,
                 image_sets='train',
                 transform=None, 
                 base_transform=None,
                 target_transform=AnnotationTransform(),
                 dataset_name='konface', 
                 mosaic=False):
        self.root = root
        self.img_size = img_size
        self.image_set = image_sets
        self.transform = transform
        self.base_transform = base_transform
        self.target_transform = target_transform
        self.image_sets = image_sets
        self.name = dataset_name
        self._annopath = osp.join(root, image_sets, 'Annotations', '%s.xml')
        self._imgpath = osp.join(root, image_sets, 'ImageSets', '%s')
        self.load_data_indexs()
        self.mosaic = mosaic
        

    def load_data_indexs(self):
        print('loading all indexs ...')
        filename_list = os.listdir(osp.join(self.root, self.image_sets, 'ImageSets'))
        self.img_names_list = []
        self.target_names_list = []
        for filename in filename_list:
            self.img_names_list.append(filename)
            self.target_names_list.append(filename[:-4])


    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt


    def __len__(self):
        return len(self.img_names_list)


    def pull_item(self, index):
        img_id = self.img_names_list[index]
        target_id = self.target_names_list[index]

        img = cv2.imread(self._imgpath % img_id)

        assert img is not None

        height, width, channels = img.shape

        # load target
        target = ET.parse(self._annopath % target_id).getroot()
        
        if self.target_transform is not None:
            target = self.target_transform(target_id, target, width, height)

        # mosaic augmentation
        if self.mosaic and np.random.randint(2):
            total_indexs = np.arange(0, len(self.img_names_list)).tolist()
            img_names_list_ = total_indexs[:index] + total_indexs[index+1:]
            # random sample 3 indexs
            id2, id3, id4 = random.sample(img_names_list_, 3)
            ids = [id2, id3, id4]
            img_lists = [img]
            tg_lists = [target]
            for id_ in ids:
                img_id = self.img_names_list[int(id_)]
                target_id = self.target_names_list[int(id_)]

                img_ = cv2.imread(self._imgpath % img_id)
                height_, width_, channels_ = img_.shape
                
                target_ = ET.parse(self._annopath % target_id).getroot()
                target_ = self.target_transform(target_id, target_, width_, height_)

                img_lists.append(img_)
                tg_lists.append(target_)
           
            mosaic_img = np.zeros([self.img_size*2, self.img_size*2, img.shape[2]], dtype=np.uint8)
            # mosaic center
            yc, xc = [int(random.uniform(-x, 2*self.img_size + x)) for x in [-self.img_size // 2, -self.img_size // 2]]

            mosaic_tg = []
            for i in range(4):
                img_i, target_i = img_lists[i], tg_lists[i]
                h0, w0, _ = img_i.shape

                # resize image to img_size
                img_i = cv2.resize(img_i, (self.img_size, self.img_size))
                h, w, _ = img_i.shape

                # place img in img4
                if i == 0:  # top left
                    x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                    x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
                elif i == 1:  # top right
                    x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, self.img_size * 2), yc
                    x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
                elif i == 2:  # bottom left
                    x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(self.img_size * 2, yc + h)
                    x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
                elif i == 3:  # bottom right
                    x1a, y1a, x2a, y2a = xc, yc, min(xc + w, self.img_size * 2), min(self.img_size * 2, yc + h)
                    x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

                mosaic_img[y1a:y2a, x1a:x2a] = img_i[y1b:y2b, x1b:x2b]
                padw = x1a - x1b
                padh = y1a - y1b

                # labels
                target_i = np.array(target_i)
                target_i_ = target_i.copy()
                if len(target_i) > 0:
                    # a valid target, and modify it.
                    target_i_[:, 0] = (w * (target_i[:, 0]) + padw)
                    target_i_[:, 1] = (h * (target_i[:, 1]) + padh)
                    target_i_[:, 2] = (w * (target_i[:, 2]) + padw)
                    target_i_[:, 3] = (h * (target_i[:, 3]) + padh)     
                    
                    mosaic_tg.append(target_i_)

            if len(mosaic_tg) == 0:
                mosaic_tg = np.zeros([1, 5])
            else:
                mosaic_tg = np.concatenate(mosaic_tg, axis=0)
                # Cutout/Clip targets
                np.clip(mosaic_tg[:, :4], 0, 2 * self.img_size, out=mosaic_tg[:, :4])
                # normalize
                mosaic_tg[:, :4] /= (self.img_size * 2)

            # augment
            mosaic_img, boxes, labels = self.base_transform(mosaic_img, mosaic_tg[:, :4], mosaic_tg[:, 4])
            # to rgb
            mosaic_img = mosaic_img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            mosaic_tg = np.hstack((boxes, np.expand_dims(labels, axis=1)))


            return torch.from_numpy(mosaic_img).permute(2, 0, 1).float(), mosaic_tg, self.img_size, self.img_size

        # basic augmentation(SSDAugmentation or BaseTransform)
        if self.transform is not None:
            # check labels
            if len(target) == 0:
                target = np.zeros([1, 5])
            else:
                target = np.array(target)

            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        
        return torch.from_numpy(img).permute(2, 0, 1).float(), target, height, width


    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.img_names_list[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR), img_id


    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.img_names_list[index]
        target_id = self.target_names_list[index]
        anno = ET.parse(self._annopath % target_id).getroot()
        gt = self.target_transform(target_id, anno, 1, 1)
        
        return img_id[1], gt


if __name__ == "__main__":
    def base_transform(image, size, mean):
        x = cv2.resize(image, (size, size)).astype(np.float32)
        x -= mean
        x = x.astype(np.float32)
        return x

    class BaseTransform:
        def __init__(self, size, mean):
            self.size = size
            self.mean = np.array(mean, dtype=np.float32)

        def __call__(self, image, boxes=None, labels=None):
            return base_transform(image, self.size, self.mean), boxes, labels

    img_size = 640
    # dataset
    dataset = OurDetection(root=DATA_ROOT, 
                           img_size=img_size,
                           image_sets='train',
                           transform=BaseTransform(img_size, (0, 0, 0)),
                           base_transform=BaseTransform(img_size, (0, 0, 0)),
                           target_transform=AnnotationTransform(), 
                            mosaic=True)
                            
    for i in range(1000):
        im, gt, h, w = dataset.pull_item(i)
        img = im.permute(1,2,0).numpy()[:, :, (2, 1, 0)].astype(np.uint8)
        cv2.imwrite('-1.jpg', img)
        img = cv2.imread('-1.jpg')
        for box in gt:
            xmin, ymin, xmax, ymax, _ = box
            xmin *= img_size
            ymin *= img_size
            xmax *= img_size
            ymax *= img_size
            img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,0,255), 2)
        cv2.imshow('gt', img)
        cv2.waitKey(0)
