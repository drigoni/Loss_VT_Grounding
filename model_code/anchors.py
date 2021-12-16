#!/usr/bin/env python
"""
Created on 19/02/21
Author: Davide Rigoni
Emails: davide.rigoni.2@phd.unipd.it - drigoni@fbk.eu
Description: This file contains all the function related to bounding boxes manipulation in pytorch.
Based on code from https://github.com/fastai/fastai_docs/blob/master/dev_nb/102a_coco.ipynb
"""

import torch


def torch_generate_spatial_feature(bounding_box, W, H):
    """
    This function generate spatial features.
    :param bounding_box: set of bounding boxes in format [xmin, ymin, xmax, ymax]
    :param W: images width.
    :param H: images height.
    :return: set of spatial features.
    """
    res_1 = bounding_box[:, 0] / W
    res_2 = bounding_box[:, 1] / H
    res_3 = bounding_box[:, 2] / W
    res_4 = bounding_box[:, 3] / H
    width = torch.clamp(bounding_box[:, 2] - bounding_box[:, 0], min=0)
    heigth = torch.clamp(bounding_box[:, 3] - bounding_box[:, 1], min=0)
    res_5 = (width * heigth) / (W * H)

    results = torch.cat([res_1, res_2, res_3, res_4, res_5], dim=-1)
    return results


def generalized_box_iou(bbox_1, bbox_2):
    """
    Return generalized intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Args:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])
    Returns:
        generalized_iou (Tensor[N, M]): the NxM matrix containing the pairwise generalized_IoU values
        for every element in boxes1 and boxes2
    """
    # check degenerative bounding boxes which give NAN and Inf errors.
    assert (bbox_1[:, 2:] >= bbox_1[:, :2]).all()
    assert (bbox_2[:, 2:] >= bbox_2[:, :2]).all()
    # clamp values
    bbox_1 = torch.clamp(bbox_1, min=0)
    bbox_2 = torch.clamp(bbox_2, min=0)

    inter, union = _box_inter_union(bbox_1, bbox_2)
    iou = inter / union

    # determine the (x, y)-coordinates of the intersection rectangle
    xAi = torch.min(bbox_1[:, 0], bbox_2[:, 0])
    yAi = torch.min(bbox_1[:, 1], bbox_2[:, 1])
    xBi = torch.max(bbox_1[:, 2], bbox_2[:, 2])
    yBi = torch.max(bbox_1[:, 3], bbox_2[:, 3])
    # compute the area of intersection rectangle
    x_diffi = torch.clamp(xBi - xAi, min=0)
    y_diffi = torch.clamp(yBi - yAi, min=0)
    areai = x_diffi * y_diffi

    return iou - (areai - union) / (areai + 1e-9)

def iou(bbox_1, bbox_2):
    """
    Given two bounding boxes in format [x1, y1, x2, y2] returns intersection over union.
    :param bbox_1: bounding box 1.
    :param bbox_2: bounding box 2.
    :return:bounding boxes iou.
    """
    # check degenerative bounding boxes
    assert (bbox_1[:, 2:] >= bbox_1[:, :2]).all()
    assert (bbox_2[:, 2:] >= bbox_2[:, :2]).all()
    bbox_1 = torch.clamp(bbox_1, min=0)
    bbox_2 = torch.clamp(bbox_2, min=0)
    inter, union = _box_inter_union(bbox_1, bbox_2)
    return inter / (union + 1e-9)

def _box_inter_union(bbox_1, bbox_2):
    """
    Given two bounding boxes in format [x1, y1, x2, y2] returns the intersection and union of them.
    :param bbox_1: bounding box 1.
    :param bbox_2: bounding box 2.
    :return: intersection and union of bounding boxes.
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = torch.max(bbox_1[:, 0], bbox_2[:, 0])
    yA = torch.max(bbox_1[:, 1], bbox_2[:, 1])
    xB = torch.min(bbox_1[:, 2], bbox_2[:, 2])
    yB = torch.min(bbox_1[:, 3], bbox_2[:, 3])
    # compute the area of intersection rectangle
    x_diff = torch.clamp(xB - xA, min=0)
    y_diff = torch.clamp(yB - yA, min=0)
    inter = x_diff * y_diff
    # compute union
    area_bbox1 = _box_area(bbox_1)
    area_bbox2 = _box_area(bbox_2)
    union = area_bbox1 + area_bbox2 - inter
    return inter, union

def _box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.
    Args:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format
    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


# ---------------------------------------------------------------------------------------------------
# ----------------------------------- COORDINATE TRANSFORMATION FUNCTIONS -----------------
# ---------------------------------------------------------------------------------------------------
def cthw2tlbr(boxes):
    """
    Convert center/size format `boxes` to top/left bottom/right corners.
    :param boxes: bounding boxes
    :return: bounding boxes
    """
    top_left = boxes[..., :2] - boxes[..., 2:]/2
    bot_right = boxes[..., :2] + boxes[..., 2:]/2
    return torch.cat([top_left, bot_right], dim=-1)


def tlbr2cthw(boxes):
    """
    Convert top/left bottom/right format `boxes` to center/size corners."
    :param boxes: bounding boxes
    :return: bounding boxes
    """
    center = (boxes[..., :2] + boxes[..., 2:])/2
    sizes = boxes[..., 2:] - boxes[..., :2]
    return torch.cat([center, sizes], dim=-1)


def tlbr2tlhw(boxes):
    """
    Convert tl br format `boxes` to tl hw format"
    :param boxes: bounding boxes
    :return: bounding boxes
    """
    top_left = boxes[..., :2]
    height_width = boxes[..., 2:] - boxes[..., :2]
    return torch.cat([top_left, height_width], dim=-1)


def tlhw2tlbr(boxes):
    """
    Convert tl br format `boxes` to tl hw format"
    :param boxes: bounding boxes
    :return: bounding boxes
    """
    top_left = boxes[..., :2]
    bottom_right = boxes[..., 2:] + boxes[..., :2]
    return torch.cat([top_left, bottom_right], dim=-1)


def x1y1x2y2_to_y1x1y2x2(boxes):
    """
    "Convert xy boxes to yx boxes and vice versa"
    :param boxes: bounding boxes
    :return: bounding boxes
    """
    box_tmp = boxes.clone()
    box_tmp[..., 0], box_tmp[..., 1] = boxes[..., 1], boxes[..., 0]
    box_tmp[..., 2], box_tmp[..., 3] = boxes[..., 3], boxes[..., 2]
    return box_tmp


def simple_inter(box1, box2):
    """
    Simple intersection among bounding boxes.
    :param box1: bounding boxes coordinates.
    :param box2: bounding boxes coordinates.
    :return: intersection among pair of bounding boxes.
    """
    top_left_i = torch.max(box1[..., :2], box2[..., :2])
    bot_right_i = torch.min(box1[..., 2:], box2[..., 2:])
    sizes = torch.clamp(bot_right_i - top_left_i, min=0)
    return sizes[..., 0] * sizes[..., 1]


def simple_iou(box1, box2):
    """
    Simple intersection among bounding boxes.
    :param box1: bounding boxes coordinates.
    :param box2: bounding boxes coordinates.
    :return: intersection over union among pair of bounding boxes.
    """
    inter = simple_inter(box1, box2)
    ancs, tgts = tlbr2tlhw(box1), tlbr2tlhw(box2)
    anc_sz, tgt_sz = ancs[..., 2] * ancs[..., 3], tgts[..., 2] * tgts[..., 3]
    union = anc_sz + tgt_sz - inter
    return inter / (union + 1e-9)


def bbox_final_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9, clamp=True):
    """
    Get bounding boxes iou among pairs.
    :param box1: bounding boxes 1
    :param box2: bounding boxes 2
    :param x1y1x2y2: if in this format the coordinates.
    :param GIoU: generalize iou. Default false
    :param DIoU: Distance iou. Default false
    :param CIoU: Complete iou. Default false
    :param eps: epsilon value in order to avoid division by 0
    :return: bounding boxes iou.
    """

    # sett all coordinates to min values 0.
    if clamp:
        box1 = torch.clamp(box1, min=0, max=1)
        box2 = torch.clamp(box2, min=0, max=1)

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[..., 0] - box1[..., 2] / 2, box1[..., 0] + box1[..., 2] / 2
        b1_y1, b1_y2 = box1[..., 1] - box1[..., 3] / 2, box1[..., 1] + box1[..., 3] / 2
        b2_x1, b2_x2 = box2[..., 0] - box2[..., 2] / 2, box2[..., 0] + box2[..., 2] / 2
        b2_y1, b2_y2 = box2[..., 1] - box2[..., 3] / 2, box2[..., 1] + box2[..., 3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / 3.1415927410125732 ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    # drigoni: sometimes was throwing a division by 0 error.
                    denominator = ((1 + eps) - iou + v)
                    denominator_clear = denominator.masked_fill(denominator == 0, eps)
                    alpha = v / denominator_clear

                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU
