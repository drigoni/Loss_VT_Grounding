#!/usr/bin/env python
"""
Created on 31/01/21
Author: Davide Rigoni
Emails: davide.rigoni.2@phd.unipd.it - drigoni@fbk.eu
Description:  this file contains the functions needed to extract the Bottom UP faster RCNN information.
"""

import argparse
import os
import numpy as np


def read_img(file):
    """
    Read img information from bottom-up .npz file.
    :param file: file of bottom-up .npz file.
    :return: a dictionary as shown in the code.
    bbox in format: [xmin, ymin, xmax, ymax]??
    """
    data = np.load(file)
    # need to remove all the ndarray type of numpy
    data_dict = {
        'num_bbox': data['num_bbox'].tolist(),
        'image_w': data['image_w'].tolist(),
        'image_h': data['image_h'].tolist(),
        'bbox': data['bbox'].tolist(),
        'cls_prob': data['cls_prob'].tolist(),
        'attr_prob': data['attr_prob'].tolist(),
        'x': data['x'].tolist(),
    }
    return data_dict


def parse_args():
    """
    Parse input arguments.
    """
    parser = argparse.ArgumentParser(description='Inputs')
    parser.add_argument('--folder', dest='folder',
                        help='Bottom up extracted features.',
                        default='/home/drigoni/repository/Loss_VT_Grounding/data/out_bu/',
                        type=str)

    parser.add_argument('--id', dest='id',
                        help='A given image id.',
                        default='8849890',
                        type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    print('SMALL TEST')
    args = parse_args()
    folder = args.folder
    file = os.path.join(folder, args.id + '.jpg.npz')
    data = read_img(file)
    print('Data: ', data)
    print('END')
