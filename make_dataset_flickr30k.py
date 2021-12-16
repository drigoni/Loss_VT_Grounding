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
from model_code.parser import BU_parser, flickr_parser
from multiprocessing import Process
import random
import model_code.utils as utils
from torchtext.data.utils import get_tokenizer
from collections import Counter

def extract_info(id, data_bu, data_flickr_ann, data_flickr_sent):
    """
    Saving a preprocessing data example in json.
    :param id: resource id.
    :param data_bu: bottom up data.
    :param data_flickr_ann: Flickr annotations ground truth data.
    :param data_flickr_sent: Flickr sentences ground truth data.
    :return: a dictionary ready for the pytorch model_code input.
    """
    #  ---------------- sentences data from flickr30k
    out_sentences = {
        'id': id,
        'sentences': [i['sentence'] for i in data_flickr_sent],
        'phrases': [[phrase['phrase'] for phrase in line['phrases']] for line in data_flickr_sent],
        'n_phrases': [len(line['phrases']) for line in data_flickr_sent],
        'phrases_index': [[phrase['first_word_index'] for phrase in line['phrases']] for line in data_flickr_sent],
        'phrases_id': [[phrase['phrase_id'] for phrase in line['phrases']] for line in data_flickr_sent],
        # Note: the problem of having a list of phrase_type is solved in the pre-processing step.
        'phrases_type': [[phrase['phrase_type'] for phrase in line['phrases']] for line in data_flickr_sent],
    }
    #  ---------------- annotations data from flickr30k
    out_annotations = {
        'image_w': data_flickr_ann['width'],  # [1]
        'image_h': data_flickr_ann['height'],  # [1]
        'image_d': data_flickr_ann['depth'],  # [1]
        'image_boxes_id': [i for i in data_flickr_ann['boxes'].keys()],  # [nbox]
        # Note: the union of the bounding boxes is performed in the pre-processing step.
        'image_boxes_coordinates': [i for i in data_flickr_ann['boxes'].values()],  # [nbox]
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


def save(info, out_folder, file):
    """
    This function saves all the preprocessed data of an image with all its sentences.
    In particular, this function save a file for each example represented by an image with only one sentence.
    :param info: data to save.
    :param out_folder: output folder.
    :param file: name of the file.
    """
    # dividing in 2 in order to save HDD space
    example_dict_img = {
        'id': info['id'],
        'image_w': info['image_w'],
        'image_h': info['image_h'],
        'image_d': info['image_d'],
        'image_boxes_id': info['image_boxes_id'],
        'image_boxes_coordinates': info['image_boxes_coordinates'],
        'pred_n_boxes': info['pred_n_boxes'],
        'pred_boxes': info['pred_boxes'],
        'pred_cls_prob': info['pred_cls_prob'],
        'pred_attr_prob': info['pred_attr_prob'],
        'pred_boxes_features': info['pred_boxes_features'],
    }
    # Code needed to select only a subset of bounding boxes proposals
    # n_boxes_to_keep = 50
    # example_dict_img['pred_boxes'] = example_dict_img['pred_boxes'][:n_boxes_to_keep]
    # example_dict_img['pred_n_boxes'] = n_boxes_to_keep
    # example_dict_img['pred_boxes_features'] = example_dict_img['pred_boxes_features'][:n_boxes_to_keep]
    final_file_img = os.path.join(out_folder, file + '_img.pickle')
    # saving in this way seems that does not loose precision
    utils.save_pickle(final_file_img, example_dict_img)
    # create a file for each sentence
    n_example_out = 0
    for line in range(0, 5):
        example_dict_text = {
            'id': int(info['id']),
            'sentence': info['sentences'][line],
            'phrases': info['phrases'][line],
            'n_phrases': info['n_phrases'][line],
            'phrases_index': info['phrases_index'][line],
            'phrases_id': info['phrases_id'][line],
            'phrases_type': info['phrases_type'][line],
        }

        # as the scientific community, we remove all the queries without bounding boxes.
        # moreover, we create a new useful variable "phrases_2_crd".
        example_dict_text['phrases_2_crd'] = []
        new_phrases = []
        for idx, phrase in zip(example_dict_text['phrases_id'], example_dict_text['phrases']):
            # check link
            if idx in example_dict_img['image_boxes_id']:
                val = example_dict_img['image_boxes_id'].index(idx)
                example_dict_text['phrases_2_crd'].append(example_dict_img['image_boxes_coordinates'][val])
                new_phrases.append(phrase)
        # update dictionary values
        example_dict_text['phrases'] = new_phrases
        example_dict_text['n_phrases'] = len(new_phrases)
        # NOTA: filtered in wrong way, but this information is not used in the model. So it does not influence
        # the model results
        example_dict_text['phrases_index'] = example_dict_text['phrases_index'][: example_dict_text['n_phrases']]
        example_dict_text['phrases_id'] = example_dict_text['phrases_id'][: example_dict_text['n_phrases']]
        example_dict_text['phrases_type'] = example_dict_text['phrases_type'][: example_dict_text['n_phrases']]

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

    def _make_dataset(files, out_folder, img_folder, ann_folder, sent_folder):
        for i, file in enumerate(files):
            # load data from bottom-up faster rcnn
            data_img = BU_parser.read_img(os.path.join(img_folder, file + '.jpg.npz'))
            # load ground truth data from flickr30k annotation file
            data_ann = flickr_parser.Flickr30k_annotations(os.path.join(ann_folder, file + '.xml'))
            data_ann_clear = flickr_parser.bbox_prepocessing(data_ann)  # union boxes
            # load ground truth data from flickr30k sentences file
            data_sent = flickr_parser.Flickr30k_sentence_data(os.path.join(sent_folder, file + '.txt'))
            data_sent_clear = flickr_parser.sentence_prepocessing(data_sent)  # unifies sentence_types
            # generate the last data representation
            final_data = extract_info(file, data_img, data_ann_clear, data_sent_clear)
            # save the data
            save(final_data, out_folder, file)
            # exit(0)

    root = args.root
    img_folder = os.path.join(root, 'data/flickr30k_raw/out_bu/')
    out_folder = os.path.join(root, 'data/flickr30k_raw/preprocessed/')
    ann_folder = os.path.join(root, 'data/flickr30k/flickr30k_entities/Annotations/')
    sent_folder = os.path.join(root, 'data/flickr30k/flickr30k_entities/Sentences/')

    if args.id is not None:
        id = args.id
        print('SMALL TEST')
        _make_dataset([id], out_folder, img_folder, ann_folder, sent_folder)
    elif args.id is None and args.n_proc == 1:
        print('Preparing the dataset.')
        files = [i.split('.')[0] for i in os.listdir(img_folder)]  # only files name without extension
        for i, file in enumerate(files):
            print('Processing {current}/{total}: {id}'.format(current=i, total=len(files), id=str(file)), end='\r')
            _make_dataset([file], out_folder, img_folder, ann_folder, sent_folder)
    else:
        print('Preparing the dataset with {n_proc} process.'.format(n_proc=args.n_proc))
        procs = []
        files = [i.split('.')[0] for i in os.listdir(img_folder)]  # only files name without extension
        process_id = [random.randint(0, args.n_proc-1) for i in files]
        for i in range(args.n_proc):
            p = Process(target=_make_dataset,
                        args=([f for p, f in zip(process_id, files) if p == i],
                              out_folder, img_folder, ann_folder, sent_folder))
            p.daemon = True
            p.start()
            procs.append(p)
        for p in procs:
            p.join()


def make_vocabulary_torch(args):
    """
    Make vocabulary of words.
    :param args: input parameters.
    """
    print('Make vocabulary.')
    root = args.root
    out_folder = os.path.join(root, 'data/flickr30k_raw/preprocessed')
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
                        # default='10002456',
                        type=str)
    parser.add_argument('--vktel', dest='vktel',
                        help='VTKEL dataset.',
                        default='VTKEL_dataset_100_documents.ttl',
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
    make_vocabulary_torch(args)

