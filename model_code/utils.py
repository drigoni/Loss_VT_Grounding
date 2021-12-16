#!/usr/bin/env python
"""
Created on 26/01/21
Author: Davide Rigoni
Emails: davide.rigoni.2@phd.unipd.it - drigoni@fbk.eu
Description: this file contains some useful functions.
"""

import numpy as np
import argparse
import os
import json
import matplotlib as pl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import torch
from torch.nn.utils.rnn import pad_sequence
import random
import gzip
import pickle
import pandas as pd

pl.rcParams['figure.dpi'] = 230

def get_classes(data_folder, number=None):
    """
    Return the object detection classes. If a number is specified, return the class defined by the number.
    :param data_folder: folder of the data.
    :param number: optional. Specifies a class.
    :return: a list of classes or a precise one.
    """
    CLASSES = ['__background__']
    with open(os.path.join(data_folder, 'objects_vocab.txt')) as f:
        for object in f.readlines():
            CLASSES.append(object.lower().strip())
    if number is None:
        return CLASSES
    else:
        return CLASSES[number]


def get_attributes(data_folder, number=None):
    """
    Return the object detection attributes. If a number is specified, return the attribute defined by the number.
    :param data_folder: folder of the data.
    :param number: optional. Specifies an attribute.
    :return: a list of attributes or a precise one.
    """
    ATTRS = []
    with open(os.path.join(data_folder, 'attributes_vocab.txt')) as f:
        for attr in f.readlines():
            ATTRS.append(attr.lower().strip())
    if number is None:
        return ATTRS
    else:
        return ATTRS[number]


def one_hot(elements, classes):
    """
    This function transform a list of labels in a matrix where each line is a one-hot encoding of
    the corresponding label in the list. If the labels are not in classes, the corresponding vector has all the
    values set to 0.
    :param elements: list of labels to transform in one-hot encoding.
    :param classes: list of labels.
    :return: a matrix where each line is a one-hot encoding of the labels in input.
    """
    final = []
    for el in elements:
        if el not in classes:
            print('Error. Class {} not recognized in: {}'.format(el, elements))
        current = [1 if i == el else 0 for i in classes]
        final.append(current)
    return final


# ---------------------------------------------------------------------------------------------------
# ----------------------------------- PYTORCH PADDING FUNCTIONS -----------------
# ---------------------------------------------------------------------------------------------------
def torch_pad(data, dtype, batch_first=True, padding_value=-1):
    """
    Pads a list of list of numbers.
    :param data: list of list of numbers.
    :return: padded torch tensors.
    """
    result = torch_make_tensors(data, dtype)
    result = pad_sequence(result, batch_first=batch_first, padding_value=padding_value)
    mask = result != padding_value
    return result, mask


def torch_make_tensors(data, dtype):
    """
    Transform list of list of numbers in list of pytorch tensors. Useful in collate_fn for dataloader.
    :param data: list of list of numbers.
    :return: list of torch tensors.
    """
    # result = [torch.tensor(i, dtype=dtype).clone().detach() for i in data]
    result = [torch.tensor(i, dtype=dtype) for i in data]
    return result


# ---------------------------------------------------------------------------------------------------
# ----------------------------------- BOUNDING BOXES FUNCTIONS -----------------
# ---------------------------------------------------------------------------------------------------
def intersection_over_union(boxA, boxB):
    """
    Intersection over union of two bounding boxes.
    :param boxA: bounding box A. Format: [xmin, ymin, xmax, ymax]
    :param boxB: bounding box B.
    :return: intersection over union score.
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def bounding_boxes_xyxy2xywh(bbox_list):
    """
    Transform bounding boxes coordinates.
    :param bbox_list: list of coordinates as: [[xmin, ymin, xmax, ymax], [xmin, ymin, xmax, ymax]]
    :return: list of coordinates as: [[xmin, ymin, width, height], [xmin, ymin, width, height]]
    """
    new_coordinates = []
    for box in bbox_list:
        new_coordinates.append([box[0],
                                box[1],
                                box[2] - box[0],
                                box[3] - box[1]])
    return new_coordinates


def scale_bbox(bbox_list, width, height):
    """
    Normalize a bounding box give max_x and max_y.
    :param bbox_list: list of list of coodinates in format: [xmin, ymin, xmax, ymax]
    :param width: image max width.
    :param height: image max height
    :return: list of list of normalized coordinates.
    """
    results = []
    for i in bbox_list:
        xmin, ymin, xmax, ymax = i
        norm_cr = [xmin/width, ymin/height, xmax/width, ymax/height]
        results.append(norm_cr)
    return results

def scale_bbox_noUB(bbox_list, width, height):
    """
    Normalize a bounding box give max_x and max_y.
    :param bbox_list: list of list of coodinates in format: [xmin, ymin, xmax, ymax]
    :param width: image max width.
    :param height: image max height
    :return: list of list of normalized coordinates.
    """
    results = []
    for i in bbox_list:
        results_tmp = []
        for xmin, ymin, xmax, ymax in i:
            norm_cr = [xmin/width, ymin/height, xmax/width, ymax/height]
            results_tmp.append(norm_cr)
        results.append(results_tmp)
    return results


def union_of_rects(rects, dtype=np.int32):
    """
    Calculates union of two rectangular boxes.
    :param rects: rects as N x [xmin, ymin, xmax, ymax].
    :return: the coordinates of the new bounding box.
    """
    xA = np.min(rects[:, 0])
    yA = np.min(rects[:, 1])
    xB = np.max(rects[:, 2])
    yB = np.max(rects[:, 3])
    return np.array([xA, yA, xB, yB], dtype=dtype)

# ---------------------------------------------------------------------------------------------------
# ----------------------------------- IMAGE RELATED FUNCTIONS -----------------
# ---------------------------------------------------------------------------------------------------
def show_image(image, title, bbox_pred, bbox_gt, bbox_query, sentence):
    """
    Show an image with predicted bounding boxes and GT bounding boxes.
    Note that pyplotlib uses a different coordinate system only for figtext function (i.e. (0,0) -> bottom-left corner).
    :param image: image.
    :param bbox_pred: bounding boxes list of list [[xmin, ymin, xmax, ymax], [xmin, ymin, xmax, ymax]]
    :param bbox_gt: bounding boxes list of list [[xmin, ymin, xmax, ymax], [xmin, ymin, xmax, ymax]]
    :param bbox_query: bounding boxes list of strings
    :param title: title of the plot
    :param sentence: sentence string
    """
    # create unique colors if needed
    colors = [(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
              for i in bbox_pred if bbox_pred is not None]

    def _plot_boox_list(boxes, curr_title):
        plt.imshow(image)
        plt.title(curr_title + '\n', fontdict={'fontsize': 7})
        # transform bounding boxes coordinates
        bbox_transf = bounding_boxes_xyxy2xywh(boxes)
        # Get the current reference
        ax = plt.gca()
        for box, query, color in zip(bbox_transf, bbox_query, colors):
            frame1 = plt.gca()
            # frame1.axes.xaxis.set_ticklabels([])
            # frame1.axes.yaxis.set_ticklabels([])
            frame1.axes.xaxis.set_visible(False)
            frame1.axes.yaxis.set_visible(False)
            plt.text(box[0], box[1] -2 , query, bbox=dict(facecolor='blue', alpha=0.5), fontsize=5, color='white')
            rect = patches.Rectangle((box[0], box[1]),
                                     box[2], box[3],
                                     linewidth=1, edgecolor=color, facecolor='none')
            ax.add_patch(rect)

    # height, width, channels = image.shape
    if sentence is not None:
        plt.figtext(0.5, 0.01, '"' + sentence + '"', ha="center", fontsize=7, wrap=True)
    if bbox_pred is not None and bbox_gt is not None:
        plt.subplot(1, 2, 1)
        _plot_boox_list(bbox_pred, 'Prediction'.format(title))
        plt.subplot(1, 2, 2)
        _plot_boox_list(bbox_gt, 'Ground Truth'.format(title))
    elif bbox_pred is not None and bbox_gt is None:
        _plot_boox_list(bbox_pred, '{}'.format(title))

    # show image
    plt.show()


def show_proposals(image, title, bbox_pred, bbox_pred_cls, bbox_gt=None, sentence=None):
    """
    Show an image with proposed bounding boxes and GT bounding boxes.
    Note that pyplotlib uses a different coordinate system only for figtext function (i.e. (0,0) -> bottom-left corner).
    :param image: image.
    :param bbox_pred: bounding boxes list of list [[xmin, ymin, xmax, ymax], [xmin, ymin, xmax, ymax]]
    :param bbox_gt: bounding boxes list of list [[xmin, ymin, xmax, ymax], [xmin, ymin, xmax, ymax]]
    :param title: title of the plot
    :param sentence: sentence string
    """
    # create unique colors if needed
    colors = [(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)) for i in range(bbox_pred_cls.shape[1])]

    def _plot_boox_list(boxes, curr_title):
        plt.imshow(image)
        plt.title(curr_title, fontdict={'fontsize': 7})
        # transform bounding boxes coordinates
        bbox_transf = bounding_boxes_xyxy2xywh(boxes)
        # Get the current reference
        ax = plt.gca()
        for box, classes in zip(bbox_transf, bbox_pred_cls):
            rect = patches.Rectangle((box[0], box[1]),
                                     box[2], box[3],
                                     linewidth=1, edgecolor=colors[np.argmax(classes)],
                                     facecolor='none')
            ax.add_patch(rect)

    # height, width, channels = image.shape
    if sentence is not None:
        plt.figtext(0.5, 0.01, sentence, ha="center", fontsize=7, wrap=True)
    if bbox_pred is not None and bbox_gt is not None:
        plt.subplot(1, 2, 1)
        _plot_boox_list(bbox_pred, 'Predictions img: {} '.format(title))
        plt.subplot(1, 2, 2)
        _plot_boox_list(bbox_gt, 'Ground Truth img: {}'.format(title))
    elif bbox_pred is not None and bbox_gt is None:
        _plot_boox_list(bbox_pred, '{}'.format(title))

    # show image
    plt.show()

# ---------------------------------------------------------------------------------------------------
# ----------------------------------- LOADING AND SAVING FUNCTIONS -----------------
# ---------------------------------------------------------------------------------------------------
def load_image(file):
    """
    Read an image.
    :param file: image path.
    :return: the image.
    """
    loaded_img = cv2.imread(file)
    # change color
    im = cv2.cvtColor(loaded_img, cv2.COLOR_BGR2RGB)
    return im


def save_gzip(file, data):
    """
    Save a dictionary in a gzip file.
    :param file: name of the file
    :param data: data to save as dictionary
    :param compression: compression to use
    """
    str = json.dumps(data)
    encoded = str.encode('utf-8')
    with gzip.open(file, 'wb') as f:
        f.write(encoded)


def load_gzip(file):
    """
    Read a gzip file.
    :param file: file to read
    :return: the dictionary contained in the gzip file
    """
    with gzip.open(file, "rb") as f:
        data = f.read()
    d = json.loads(data)
    return d


def load_json(file):
    """
    Load a .json file.
    :param file: file .json to load.
    :return: loaded data.
    """
    with open(file, "r") as json_file:
        data = json.load(json_file)
    return data


def save_json(file, data, indent=None):
    """
    Save a .json file.
    :param file: file.json name.
    :param data: data to save.
    :param indent: indent of the file.
    """
    with open(file, 'w') as f:
        json.dump(data, f, indent=indent)


def load_pickle(file, decompress=True):
    """
    Load a .pickle file.
    :param file: file .pickle to load.
    :param decompress: the compress or not the file
    :return: loaded data.
    """
    with open(file, "rb") as f:
        if decompress:
            data = pickle.load(f)
        else:
            data = f.read()
    return data


def save_pickle(file, data):
    """
    Save a .pickle file.
    :param file: file .pickle name.
    :param data: data to save.
    """
    with open(file, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_glove(file):
    """
    This function load the .csv glove embedding file.
    wget http://nlp.stanford.edu/data/wordvecs/glove.6B.zip
    unzip glove.6B.zip
    :param file: file to load
    :return: a dictionary of words and their embeddings
    """
    glove = pd.read_csv(file, sep=" ", quoting=3, header=None, index_col=0)
    glove_embedding = {key: val.values for key, val in glove.T.items()}
    return glove_embedding


def parse_args():
    """
    Parse input arguments.
    """
    parser = argparse.ArgumentParser(description='Inputs')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    print('SMALL TEST')
    # take input image from Flickr30k
    args = parse_args()
    bbox_a = [0, 0, 50, 50]
    bbox_b = [0, 0, 25, 25]
    intersection = intersection_over_union(bbox_a, bbox_b)
    print('intersection', intersection)
    one_hot_result = one_hot(['a', 'a', 'c', 'd'], ['a', 'b', 'c'])
    print('one_hot_result', one_hot_result)
    img = load_image('/home/drigoni/repository/Loss_VT_Grounding/data/flickr30k/flickr30k_images/22143604.jpg')
    show_image(img, [[253, 122, 285, 134], [220, 125, 244, 177], [209, 128, 224, 179], [305, 123, 341, 180]])
    print('END')
