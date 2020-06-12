from data import *
from data.cocodataset import *
import numpy as np
import random

class Box():
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

def iou(box1, box2):
    x1, y1, w1, h1 = box1.x, box1.y, box1.w, box1.h
    x2, y2, w2, h2 = box2.x, box2.y, box2.w, box2.h

    S_1 = w1 * h1
    S_2 = w2 * h2

    xmin_1, ymin_1 = x1 - w1 / 2, y1 - h1 / 2
    xmax_1, ymax_1 = x1 + w1 / 2, y1 + h1 / 2
    xmin_2, ymin_2 = x2 - w2 / 2, y2 - h2 / 2
    xmax_2, ymax_2 = x2 + w2 / 2, y2 + h2 / 2

    I_w = min(xmax_1, xmax_2) - max(xmin_1, xmin_2)
    I_h = min(ymax_1, ymax_2) - max(ymin_1, ymin_2)
    if I_w < 0 or I_h < 0:
        return 0
    I = I_w * I_h

    IoU = I / (S_1 + S_2 - I)

    return IoU

def init_centroids(boxes, n_anchors):
    """
        We use kmeans++ to initialize centroids.
    """
    centroids = []
    boxes_num = len(boxes)

    centroid_index = int(np.random.choice(boxes_num, 1)[0])
    centroids.append(boxes[centroid_index])
    print(centroids[0].w,centroids[0].h)

    for centroid_index in range(0, n_anchors-1):
        sum_distance = 0
        distance_thresh = 0
        distance_list = []
        cur_sum = 0

        for box in boxes:
            min_distance = 1
            for centroid_i, centroid in enumerate(centroids):
                distance = (1 - iou(box, centroid))
                if distance < min_distance:
                    min_distance = distance
            sum_distance += min_distance
            distance_list.append(min_distance)

        distance_thresh = sum_distance * np.random.random()

        for i in range(0, boxes_num):
            cur_sum += distance_list[i]
            if cur_sum > distance_thresh:
                centroids.append(boxes[i])
                print(boxes[i].w, boxes[i].h)
                break
    return centroids

def do_kmeans(n_anchors, boxes, centroids):
    loss = 0
    groups = []
    new_centroids = []
    # for box in centroids:
    #     print('box: ', box.x, box.y, box.w, box.h)
    # exit()
    for i in range(n_anchors):
        groups.append([])
        new_centroids.append(Box(0, 0, 0, 0))
    
    for box in boxes:
        min_distance = 1
        group_index = 0
        for centroid_index, centroid in enumerate(centroids):
            distance = (1 - iou(box, centroid))
            if distance < min_distance:
                min_distance = distance
                group_index = centroid_index
        groups[group_index].append(box)
        loss += min_distance
        new_centroids[group_index].w += box.w
        new_centroids[group_index].h += box.h

    for i in range(n_anchors):
        new_centroids[i].w /= max(len(groups[i]), 1)
        new_centroids[i].h /= max(len(groups[i]), 1)

    return new_centroids, groups, loss# / len(boxes)

def anchor_box_kmeans(total_gt_boxes, n_anchors, loss_convergence, iters, plus=True):
    """
        This function will use k-means to get appropriate anchor boxes for train dataset.
        Input:
            total_gt_boxes: 
            n_anchor : int -> the number of anchor boxes.
            loss_convergence : float -> threshold of iterating convergence.
            iters: int -> the number of iterations for training kmeans.
        Output: anchor_boxes : list -> [[w1, h1], [w2, h2], ..., [wn, hn]].
    """
    boxes = total_gt_boxes
    centroids = []
    if plus:
        centroids = init_centroids(boxes, n_anchors)
    else:
        total_indexs = range(len(boxes))
        sample_indexs = random.sample(total_indexs, n_anchors)
        for i in sample_indexs:
            centroids.append(boxes[i])

    # iterate k-means
    centroids, groups, old_loss = do_kmeans(n_anchors, boxes, centroids)
    iterations = 1
    while(True):
        centroids, groups, loss = do_kmeans(n_anchors, boxes, centroids)
        iterations += 1
        print("Loss = %f" % loss)
        if abs(old_loss - loss) < loss_convergence or iterations > iters:
            break
        old_loss = loss

        for centroid in centroids:
            print(centroid.w, centroid.h)
    
    print("k-means result : ") 
    for centroid in centroids:
        print(round(centroid.w, 2), round(centroid.h, 2), "area: ", round(centroid.w, 2) * round(centroid.h, 2))
    
    return centroids

if __name__ == "__main__":
    n_anchors = 6
    loss_convergence = 1e-6
    iters_n = 1000
    input_size = 416
    # dataset = VOCDetection(root=VOC_ROOT,
    #                     transform=BaseTransform([416, 416]))
    dataset = COCODataset(
                  data_dir='./data/COCO/',
                  img_size=input_size,
                  transform=BaseTransform(input_size),
                  debug=None)
    boxes = []
    print("The dataset size: ", len(dataset))
    print("Loading the dataset ...")
    for i in range(len(dataset)):
        if i % 5000 == 0:
            print('Loading datat [%d / %d]' % (i+1, len(dataset)))

        # For COCO
        img, _ = dataset.pull_image(i)
        w, h = img.shape[1], img.shape[0]
        annotation = dataset.pull_anno(i)

        # # For VOC
        # img = dataset.pull_image(i)
        # w, h = img.shape[1], img.shape[0]
        # _, annotation = dataset.pull_anno(i)

        for box_and_label in annotation:
            box = box_and_label[:-1]
            xmin, ymin, xmax, ymax = box
            bw = (xmax - xmin) / w * input_size
            bh = (ymax - ymin) / h * input_size
            boxes.append(Box(0, 0, bw, bh))
    # for i in range(5):
    #     w, h = 2*(i+1)+np.random.randint(5), 2*(i+1)+np.random.randint(5)
    #     print(w, h)
    #     boxes.append(Box(0, 0, w, h))
    print("Start k-means !")
    centroids = anchor_box_kmeans(boxes, n_anchors, loss_convergence, iters_n, plus=True)
