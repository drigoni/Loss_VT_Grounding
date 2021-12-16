#!/usr/bin/env python
"""
Created on 26/01/21
Author: Davide Rigoni
Emails: davide.rigoni.2@phd.unipd.it - drigoni@fbk.eu
Description: this file extracts image(s) annotations (both visual and textual) from Flickr30k.
"""
import xml.etree.ElementTree as ET
import argparse
import model_code.utils as utils
import numpy as np


def Flickr30k_annotations(flickr_image):
    """
    Parses the xml files in the Flickr30K Entities dataset.
    :param flickr_image:
    :return: a dictionary with the following fields:
                scene - list of identifiers which were annotated as pertaining to the whole scene
                nobox - list of identifiers which were annotated as not being visible in the image
                boxes - a dictionary where the fields are identifiers and the values are its list of boxes in the
                [xmin ymin xmax ymax] format
                width - image width.
                height - image height.
                depth - image depth.
    """

    tree = ET.parse(flickr_image)
    root = tree.getroot()
    size_container = root.findall('size')[0]
    anno_info = {'boxes' : {}, 'scene' : [], 'nobox' : []}
    for size_element in size_container:
        anno_info[size_element.tag] = int(size_element.text)

    for object_container in root.findall('object'):
        for names in object_container.findall('name'):
            box_id = names.text
            box_container = object_container.findall('bndbox')
            if len(box_container) > 0:
                if box_id not in anno_info['boxes']:
                    anno_info['boxes'][box_id] = []
                xmin = int(box_container[0].findall('xmin')[0].text) - 1
                ymin = int(box_container[0].findall('ymin')[0].text) - 1
                xmax = int(box_container[0].findall('xmax')[0].text) - 1
                ymax = int(box_container[0].findall('ymax')[0].text) - 1
                anno_info['boxes'][box_id].append([xmin, ymin, xmax, ymax])
            else:
                nobndbox = int(object_container.findall('nobndbox')[0].text)
                if nobndbox > 0:
                    anno_info['nobox'].append(box_id)

                scene = int(object_container.findall('scene')[0].text)
                if scene > 0:
                    anno_info['scene'].append(box_id)

    return anno_info


def Flickr30k_sentence_data(fn):
    """
    Parses a sentence file from the Flickr30K Entities dataset.
    :param fn: full file path to the sentence file to parse.
    :return: a list of dictionaries for each sentence with the following fields:
                sentence - the original sentence
                phrases - a list of dictionaries for each phrase with the following fields:
                    phrase - the text of the annotated phrase
                    first_word_index - the position of the first word of the phrase in the sentence
                    phrase_id - an identifier for this phrase
                    phrase_type - a list of the coarse categories this phrase belongs to
    """
    with open(fn, 'r') as f:
        sentences = f.read().split('\n')

    annotations = []
    for sentence in sentences:
        if not sentence:
            continue

        first_word = []
        phrases = []
        phrase_id = []
        phrase_type = []
        words = []
        current_phrase = []
        add_to_phrase = False
        for token in sentence.split():
            if add_to_phrase:
                if token[-1] == ']':
                    add_to_phrase = False
                    token = token[:-1]
                    current_phrase.append(token)
                    phrases.append(' '.join(current_phrase))
                    current_phrase = []
                else:
                    current_phrase.append(token)

                words.append(token)
            else:
                if token[0] == '[':
                    add_to_phrase = True
                    first_word.append(len(words))
                    parts = token.split('/')
                    phrase_id.append(parts[1][3:])
                    phrase_type.append(parts[2:])
                else:
                    words.append(token)

        sentence_data = {'sentence': ' '.join(words), 'phrases' : []}
        for index, phrase, p_id, p_type in zip(first_word, phrases, phrase_id, phrase_type):
            sentence_data['phrases'].append({'first_word_index': index,
                                             'phrase': phrase,
                                             'phrase_id': p_id,
                                             'phrase_type': p_type})

        annotations.append(sentence_data)

    return annotations


def bbox_prepocessing(flickr_annotations):
    """
    This pre-processing function make union boxes of the Flickr30k annotations.
    Note that in order to increase the performance, we apply this function "inplace".
    :param flickr_annotations: Flickr30k annotations.
    :return: reference to the same data in input which is modified "inplace" due to performance reasons.
    """
    boxes = flickr_annotations['boxes']
    new_boxes = {}
    for box, box_list in boxes.items():
        if len(box_list) == 0:
            continue
        elif len(box_list) > 1:
            new_boxes[box] = utils.union_of_rects(np.array(box_list)).tolist()
        else:
            new_boxes[box] = box_list[0]
    flickr_annotations['boxes'] = new_boxes
    return flickr_annotations  # not really necessary. It is just a reference.


def sentence_prepocessing(flickr_sentences, sep='-'):
    """
    This pre-processing function unifies 'phrase_type' when they are more than one.
    Note that in order to increase the performance, we apply this function "inplace".
    An example is given by sentence 48296285.txt line 0: phrase 'a lab' has 'phrase_type': ['animals', 'scene'].
    In this case this function returns: 'phrase_type': 'animals-scene'.
    :param flickr_sentences: Flickr30k sentences.
    :param sep: separator to use in merging types.
    :return: reference to the same data in input which is modified "inplace" due to performance reasons.
    """
    for sentence in flickr_sentences:
        for phrase in sentence['phrases']:
            if len(phrase['phrase_type']) == 1:
                current = phrase['phrase_type'][0]
            else:
                current = sep.join(phrase['phrase_type'])
            phrase['phrase_type'] = current
    return flickr_sentences  # not really necessary. It is just a reference.


def parse_args():
    """
    Parse input arguments.
    """
    parser = argparse.ArgumentParser(description='Inputs')
    parser.add_argument('--root', dest='root',
                        help='Flickr30k root folder.',
                        default='/home/drigoni/repository/Loss_VT_Grounding/data/flickr30k/',
                        type=str)
    parser.add_argument('--id', dest='id',
                        help='Flickr30k id.',
                        default='48296285',
                        type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    print('SMALL TEST')
    # take input image from Flickr30k
    args = parse_args()
    root = args.root
    annotation_folder = root + 'flickr30k_entities/Annotations/'
    sentence_folder = root + 'flickr30k_entities/Sentences/'
    image_id = args.id
    out_visual_annotations = Flickr30k_annotations(annotation_folder + image_id + '.xml')
    print('Annotation: ', out_visual_annotations)
    out_visual_annotations_clean = bbox_prepocessing(out_visual_annotations)
    print('Annotation cleaned: ', out_visual_annotations_clean)
    out_textual_annotation = Flickr30k_sentence_data(sentence_folder + image_id + '.txt')
    print('Sentence: ', out_textual_annotation)
    out_textual_annotation_clean = sentence_prepocessing(out_textual_annotation)
    print('Sentence cleaned: ', out_textual_annotation_clean)
    print('END')