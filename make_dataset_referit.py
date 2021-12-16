#!/usr/bin/env python
"""
Created on 26/01/21
Author: Davide Rigoni
Emails: davide.rigoni.2@phd.unipd.it - drigoni@fbk.eu
Description: this file contains all the function needed to prepare the pre-processed dataset.
"""

import os
import numpy as np
import argparse
from model_code.parser import BU_parser, referit_parser
from multiprocessing import Process
import random
import model_code.utils as utils
from torchtext.data.utils import get_tokenizer
from collections import Counter


def extract_info(id, data_bu, data_referit):
    """
    Saving a preprocessing data example in json.
    :param id: resource id.
    :param data_ewiser: ewiser data.
    :param data_bu: bottom up data.
    :param data_referit: referit data.
    :return: a dictionary ready for the pytorch model_code input.
    """
    #  ---------------- sentences data from flickr30k
    out_sentences = {
        'id': id,
        'sentences': [i for i in data_referit['query']],
        'phrases': [[i] for i in data_referit['query']],
        'n_phrases': [1 for i in data_referit['query']],
    }
    #  ---------------- annotations data from flickr30k
    out_annotations = {
        'image_w': data_bu['image_w'],  # [1]
        'image_h': data_bu['image_h'],  # [1]
        'image_d': 3,  # [1]
        # Note: the union of the bounding boxes is performed in the pre-processing step.
        'image_boxes_coordinates': [data_referit['bbox'] for i in data_referit['query']],  # [nbox]
    }
    #  ---------------- bottom-up faster rcnn data.
    out_bu = {
        'pred_n_boxes': data_bu['num_bbox'],  # [1]
        # same data contained in in flickr30k annotations
        # 'image_w': data_bu['image_w'],                                                # [1]
        # 'image_h': data_bu['image_h'],                                                # [1]
        'pred_boxes': data_bu['bbox'],  # [1]
        'pred_cls_prob': data_bu['cls_prob'],  # [np_box, n_pcls]
        'pred_attr_prob': data_bu['attr_prob'],  # [np_box, n_patt]
        'pred_boxes_features': np.transpose(data_bu['x']).tolist(),  # [np_box, n_pfeatures]
    }
    out_dict = {}
    out_dict.update(out_sentences)
    out_dict.update(out_annotations)
    out_dict.update(out_bu)
    return out_dict


def save(info, out_folder, file, img_name):
    """
    This function saves all the preprocessed data of an image with all its sentences.
    In particular, this function save a file for each example represented by an image with only one sentence.
    :param info: data to save.
    :param out_folder: output folder.
    :param file: name of the file.
    """
    # dividing in 2 in order to save HDD space
    example_dict_img = {
        'id': int(img_name),
        'image_w': info['image_w'],
        'image_h': info['image_h'],
        'image_d': info['image_d'],
        # 'image_boxes_id': int(info['image_boxes_id']),
        'image_boxes_coordinates': info['image_boxes_coordinates'],
        'pred_n_boxes': info['pred_n_boxes'],
        'pred_boxes': info['pred_boxes'],
        'pred_cls_prob': info['pred_cls_prob'],
        'pred_attr_prob': info['pred_attr_prob'],
        'pred_boxes_features': info['pred_boxes_features'],
    }
    final_file_img = os.path.join(out_folder, img_name + '_img.pickle')
    # saving in this way seems that does not loose precision
    if not os.path.isfile(final_file_img):
        utils.save_pickle(final_file_img, example_dict_img)
    # create a file for each sentence
    n_example_out = 0
    for line in range(0, len(info['sentences'])):
        example_dict_text = {
            'id': int(info['id']),
            'sentence': info['sentences'][line],
            'phrases': info['phrases'][line],
            'n_phrases': info['n_phrases'][line],
        }

        # as the scientific community, we remove all the queries without bounding boxes.
        # moreover, we create a new useful variable "phrases_2_crd".
        example_dict_text['phrases_2_crd'] = []
        example_dict_text['phrases_2_crd'].append(example_dict_img['image_boxes_coordinates'][line])

        # if there are no phrases aligned with bounding boxes, we don't save the example.
        if example_dict_text['n_phrases'] > 0:
            final_file_text = os.path.join(out_folder, file + '_' + str(line) + '.pickle')
            # saving in this way seems that does not loose precision
            utils.save_pickle(final_file_text, example_dict_text)
        else:
            n_example_out += 1
    return n_example_out


def make_dataset(args):
    """
    Make dataset representation.
    :param args:  command line information.
    """

    def _make_dataset(files, out_folder, img_folder, annotations):
        for i, file in enumerate(files):
            img_name = file.split('_')[0]
            # load and pre-processing of ewiser data
            # check if the image was preprocessed correctly
            current_img_file = os.path.join(img_folder, img_name + '.jpg.npz')
            if os.path.isfile(current_img_file):
                # load data from bottom-up faster rcnn
                data_img = BU_parser.read_img(current_img_file)
                # load ground truth data from flickr30k annotation file
                data_referit = [i for i in annotations if i['ann_id'] == file][0]
                # generate the last data representation
                final_data = extract_info(file, data_img, data_referit)
                # save the data
                save(final_data, out_folder, file, img_name)
            else:
                print('Proposals for image {} not found. Phrase removed: {} .'.format(img_name, file))


    root = args.root
    refer_data_folder = os.path.join(root, 'data/refer/data/')
    img_folder = os.path.join(root, 'data/referit_raw/out_bu/')
    out_folder = os.path.join(root, 'data/referit_raw/preprocessed/')

    ref_ann, ref_inst_ann = referit_parser.referit_load_data(refer_data_folder)
    annotations = referit_parser.get_annotations(ref_ann, ref_inst_ann)

    # get list of annotations_id, there are no repetitions.
    list_idx = []
    for i in annotations:
        list_idx.append(i['ann_id'])

    if args.id is not None:
        id = args.id
        print('SMALL TEST')
        _make_dataset([id], out_folder, img_folder, annotations)
    elif args.id is None and args.n_proc == 1:
        print('Preparing the dataset.')
        for i, file in enumerate(list_idx):
            print('Processing {current}/{total}: {id}'.format(current=i, total=len(list_idx), id=str(file)), end='\r')
            _make_dataset([file], out_folder, img_folder, annotations)
    else:
        print('Preparing the dataset with {n_proc} process.'.format(n_proc=args.n_proc))
        procs = []
        process_id = [random.randint(0, args.n_proc-1) for i in list_idx]
        for i in range(args.n_proc):
            p = Process(target=_make_dataset,
                        args=([f for p, f in zip(process_id, list_idx) if p == i],
                              out_folder, img_folder, annotations))
            p.daemon = True
            p.start()
            procs.append(p)
        for p in procs:
            p.join()


def make_split_idx(args):
    """
    Make split files for training, validation and test set.
    :param args: arguments.
    """
    print("Make idx list. ")
    root = args.root
    data_root = root + 'data/refer/data/'
    output_folder = root + 'data/referit_raw/'
    ref_ann, ref_inst_ann = referit_parser.referit_load_data(data_root)
    annotations = referit_parser.get_annotations(ref_ann, ref_inst_ann)
    splits = referit_parser.referit_get_idx_split(annotations)
    tmp = {0: [], 1:[], 2:[]}
    for i, name in enumerate(['train', 'val', 'test']):
        split_name = output_folder + name + '.txt'
        with open(split_name, 'w') as file:
            curr_indexes = list(set([i.split('_')[0] for i in splits[i]]))
            file.writelines('\n'.join(curr_indexes))
            tmp[i] = curr_indexes
        print('Idx {} saved: {} '.format(name, split_name))
    print('End.')


def make_vocabulary_torch(args):
    """
    Make vocabulary of words.
    :param args: input parameters.
    """
    print('Make vocabulary.')
    root = args.root
    out_folder = os.path.join(root, 'data/referit_raw/preprocessed')
    en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
    counter = Counter()
    # loading all sentences
    list_examples = os.listdir(out_folder)
    for i, id in enumerate(list_examples):
        if 'vocab' not in id and 'img' not in id:
            print('Processed {}/{}: {}'.format(i, len(list_examples), id), end='\r')
            cur_data = utils.load_pickle(os.path.join(out_folder, id))
            # chek in order to exclude vocab.json
            if 'sentence' in cur_data:
                counter.update(en_tokenizer(cur_data['sentence']))
            else:
                print('Excluded file: {}'.format(id))
    tmp_dict = dict(counter)
    vocab_file_name = os.path.join(out_folder, 'vocab.json')
    utils.save_json(vocab_file_name, tmp_dict, indent=2)
    print('Vocabulary saved in {} with {} words.'.format(vocab_file_name, len(tmp_dict)))


def parse_args():
    """
    Parse input arguments.
    """
    parser = argparse.ArgumentParser(description='Inputs')
    parser.add_argument('--root', dest='root',
                        help='Loss_VT_Grounding root folder.',
                        default='/home/drigoni/repository/Loss_VT_Grounding/',
                        type=str)
    parser.add_argument('--id', dest='id',
                        help='Example id.',
                        default=None,
                        # default='23041_8',
                        type=str)
    parser.add_argument('--n_proc', dest='n_proc',
                        help='Number of process to use',
                        default=1,
                        type=int)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    make_dataset(args)
    make_split_idx(args)
    make_vocabulary_torch(args)
