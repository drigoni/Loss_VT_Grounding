#!/usr/bin/env python
"""
Created on 26/01/21
Author: Davide Rigoni
Emails: davide.rigoni.2@phd.unipd.it - drigoni@fbk.eu
Description: this file contains the dataset file needed by the pytorch model_code to load the pre-processed dataset.
"""

import os
import argparse
import torch
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer
import torchtext
from collections import Counter, defaultdict
import model_code.utils as utils
import random
import pickle


class VTKELReferitDataset(Dataset):
    """
    Dataset class which load the examples and makes batch of them.
    """

    def __init__(self, folder, file_idx, transform=None, load_first=False, load_first_img=False, load_subset=None):
        """
        Class init.
        :param folder: preprocessed data folder.
        :param file_idx: file containing the indexes of the examples (train, valid, test).
        :param transform: a generic transformation function.
        :param load_first: load first all the data in RAM
        :param load_first_img: if 'load_first'=True, load only images in RAM
        :param load_subset: load a subset of each dataset. It can be a float in [0, 1] or a int number <= 1000.
        """
        self.folder = folder
        self.file_idx = file_idx
        self.load_first = load_first
        self.load_first_img = load_first_img
        self.load_subset = load_subset
        self.already_loaded_data_img = []
        self.already_loaded_data_ph = []

        # load vocabulary of all the dataset
        self.vocab_dict = utils.load_json(os.path.join(self.folder, 'vocab.json'))
        self.vocab = torchtext.vocab.Vocab(Counter(self.vocab_dict), specials=['<pad>'], specials_first=True)
        self.tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

        # reading idx
        with open(self.file_idx, 'r') as file:
            self.idx = [line.strip('\n') for line in file.readlines()]
        self.real_n = len(self.idx)

        self.transform = transform
        all_examples = sorted(os.listdir(self.folder))  # only same orexample_imgder
        # filtering files which belongs to vocabulary or id_img.pickle files.
        self.examples = []
        for i in all_examples:
            if 'vocab' not in i and 'img' not in i and i.split('_')[0] in self.idx:
                self.examples.append(i)
        self.n = len(self.examples)

        # consider only a data subset
        if self.load_subset is not None:
            # consider only a subset of the data
            if isinstance(self.load_subset, int):
                selected_indexes = random.sample(range(0, self.n), self.load_subset)
                self.examples = [self.examples[i] for i in selected_indexes]
                self.n = self.load_subset
            elif isinstance(self.load_subset, float) and 0 < self.load_subset < 1:
                selected_indexes = random.sample(range(0, self.n), round(self.n*self.load_subset))
                self.examples = [self.examples[i] for i in selected_indexes]
                self.n = len(self.examples)
            else:
                print("Error. load_subset not recognized.")
                exit(1)

        # load all the data in memory before starting
        if self.load_first is True:
            # Loading all examples before starting
            idx_done = []
            img_data = []
            if self.load_first_img is True:
                # loading only the images
                for es in self.examples:
                    example_name_img = "{}_img.pickle".format(es.split('_')[0])
                    if example_name_img not in idx_done:
                        example_img = utils.load_pickle(os.path.join(self.folder, example_name_img), decompress=False)
                        img_data.append(example_img)
                        idx_done.append(example_name_img)
                    # build fake list of references
                    idx = idx_done.index(example_name_img)
                    self.already_loaded_data_img.append(img_data[idx])
            else:
                # loading all data
                for es in self.examples:
                    # loading first text
                    example_name = "{}".format(es)
                    example = utils.load_pickle(os.path.join(self.folder, example_name), decompress=False)
                    self.already_loaded_data_ph.append(example)
                    # loading images
                    example_name_img = "{}_img.pickle".format(es.split('_')[0])
                    if example_name_img not in idx_done:
                        example_img = utils.load_pickle(os.path.join(self.folder, example_name_img), decompress=False)
                        img_data.append(example_img)
                        idx_done.append(example_name_img)
                    # build fake list of references
                    idx = idx_done.index(example_name_img)
                    self.already_loaded_data_img.append(img_data[idx])

    def __len__(self):
        """
        Number of elements in the dataset.
        :return: the number of elements.
        """
        return self.n

    def __getitem__(self, idx):
        """
        Returns an example.
        :param idx: idx of the example.
        :return: data as a dictionary.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # loading id_line.gz file and id_img.gz file
        if self.load_first is False:
            example_name = "{}".format(self.examples[idx])
            example_name_img = "{}_img.pickle".format(self.examples[idx].split('_')[0])
            example = utils.load_pickle(os.path.join(self.folder, example_name))
            # print(os.path.join(self.folder, example_name_img))
            # example_img = utils.load_pickle(os.path.join(self.folder, example_name_img))
            try:
                example_img = utils.load_pickle(os.path.join(self.folder, example_name_img))
            except:
                print("Error during the loading of: ", os.path.join(self.folder, example_name_img))
            example.update(example_img)
        else:
            if self.load_first_img is True:
                # data images already loaded
                example_img = pickle.loads(self.already_loaded_data_img[idx])
                example_name = "{}".format(self.examples[idx])
                example = utils.load_pickle(os.path.join(self.folder, example_name))
                example.update(example_img)
            else:
                # all data already loaded
                example_img = pickle.loads(self.already_loaded_data_img[idx])
                example = pickle.loads(self.already_loaded_data_ph[idx])
                example.update(example_img)

        if self.transform is not None:
            example = self.transform(example)

        example['id'] = int(example['id'])  # string to integer
        example['phrases_2_crd'] = utils.scale_bbox(example['phrases_2_crd'], example['image_w'], example['image_h'])
        example['pred_boxes'] = utils.scale_bbox(example['pred_boxes'], example['image_w'], example['image_h'])

        # sometimes the bounding boxes are not as much as required, so we need to generate them as [0, 0, 0, 0]
        n_boxes_to_keep = 100  # in order to be sure
        pred_n_boxes = example['pred_n_boxes']
        n_boxes_class = len(example['pred_attr_prob'][0])
        n_boxes_attr = len(example['pred_cls_prob'][0])
        n_boxes_features = len(example['pred_boxes_features'][0])
        n_boxes_to_gen = n_boxes_to_keep - pred_n_boxes
        example['pred_n_boxes'] = n_boxes_to_keep
        example['pred_boxes'] = example['pred_boxes'] + [[0] * 4 for i in range(n_boxes_to_gen)]
        example['pred_boxes_features'] = example['pred_boxes_features'] + [[0] * n_boxes_features for i in range(n_boxes_to_gen)]
        example['pred_attr_prob'] = example['pred_attr_prob'] + [[0] * n_boxes_class for i in range(n_boxes_to_gen)]
        example['pred_cls_prob'] = example['pred_cls_prob'] + [[0] * n_boxes_attr for i in range(n_boxes_to_gen)]

        return example

    def tokenization(self, data):
        """
        Use the vocabulary to retrieve words index.
        :param data: list of strings.
        :return: a list of list of indexes.
        """
        results = []
        for raw_str in data:
            tmp = [self.vocab[token] for token in self.tokenizer(raw_str)]
            results.append(tmp)
        return results

    def pad_phrases(self, phrases, padding_value=0):
        """
        This function pad phrases.
        :param phrases: the batched phrases as list of list of strings
        :param padding_value: value used for padding. Default 0
        :return: the padded matrix and the boolean mask
        """
        batch_size = len(phrases)
        max_phrases_for_example = 0
        max_length_phrases = 0
        # tokenize
        phrases_tokenized = [self.tokenization(el) for el in phrases]

        # get info
        for i in phrases_tokenized:
            max_phrases_for_example = max(max_phrases_for_example, len(i))
            for ph in i:
                max_length_phrases = max(max_length_phrases, len(ph))

        # creating the padding tensor
        results = torch.zeros([batch_size, max_phrases_for_example, max_length_phrases], dtype=torch.int64)
        results += padding_value
        for es, es_data in enumerate(phrases_tokenized):
            for ph, ph_data in enumerate(es_data):
                for idx, idx_data in enumerate(ph_data):
                    results[es, ph, idx] = idx_data

        mask = results != padding_value
        return results, mask

    def collate_fn(self, data_batch):
        """
        This function aggregate examples in batch and applies padding functions when needed.
        :param data_batch: data as batch.
        :return: batch ready for pytorch model.
        """
        # smart way to change format in a dictionary of list instead of a list of dictionaries
        dict_batched = defaultdict(list)
        for el in data_batch:
            for key, val in el.items():
                dict_batched[key].append(val)

        # padding text sequence with position 0 which is the pad value (see Vocabulary)
        sentence, sentence_mask = utils.torch_pad(self.tokenization(dict_batched['sentence']),
                                                  dtype=torch.int64, padding_value=0)
        phrases, phrases_mask = self.pad_phrases(dict_batched['phrases'], padding_value=0)
        # NOTA: problema con il padding e variabile phrases_2_crd_mask? No.
        obj_p2c = utils.torch_pad(dict_batched['phrases_2_crd'], dtype=torch.float32, padding_value=0)
        phrases_2_crd, phrases_2_crd_mask = obj_p2c
        # not needed for now
        # obj_image = utils.torch_pad(dict_batched['image_boxes_coordinates'], dtype=torch.float32, padding_value=0)
        # image_boxes_coordinates, image_boxes_coordinates_mask = obj_image

        final_batch = {
            'id': torch.tensor(dict_batched['id'], dtype=torch.int64),  # int64 because 34 is too small
            'sentence': sentence,
            'sentence_mask': sentence_mask,
            'n_phrases': torch.tensor(dict_batched['n_phrases'], dtype=torch.long),
            'phrases': phrases,  # list of torch tensors
            'phrases_mask': phrases_mask,  # list of torch tensors
            'phrases_2_crd': phrases_2_crd,
            'phrases_2_crd_mask': phrases_2_crd_mask,
            # 'phrases_index': utils.torch_pad(dict_batched['phrases_index'], dtype=torch.long, padding_value=-1),
            # 'phrases_id': utils.torch_pad(dict_batched['phrases_id'], dtype=torch.long),
            # 'phrases_type': dict_batched['phrases_type'],
            # 'ewiser_chunks': [self.vocab_conv(el, pad=True) for el in dict_batched['ewiser_chunks']],  # [packed tensors]
            # 'ewiser_heads': [self.vocab_conv(el, pad=True) for el in dict_batched['ewiser_heads']],  # [packed tensors]
            # 'ewiser_begin': utils.torch_pad(dict_batched['ewiser_begin'], dtype=torch.long, padding_value=-1),
            # 'ewiser_end': utils.torch_pad(dict_batched['ewiser_end'], dtype=torch.long, padding_value=-1),
            # 'ewiser_n': utils.torch_pad(dict_batched['ewiser_n'], dtype=torch.long),
            # 'ewiser_yago_entities': dict_batched['ewiser_yago_entities'],
            'image_w': torch.tensor(dict_batched['image_w'], dtype=torch.long),
            'image_h': torch.tensor(dict_batched['image_h'], dtype=torch.long),
            # 'image_d': torch.tensor(dict_batched['image_d'], dtype=torch.long),
            # 'image_boxes_id': utils.torch_pad(dict_batched['image_boxes_id'], dtype=torch.long),
            # 'image_boxes_coordinates': image_boxes_coordinates,
            # 'image_boxes_coordinates_mask': image_boxes_coordinates_mask,
            'pred_n_boxes': torch.tensor(dict_batched['pred_n_boxes'], dtype=torch.long),
            'pred_boxes': torch.tensor(dict_batched['pred_boxes'], dtype=torch.float32),
            'pred_cls_prob': torch.tensor(dict_batched['pred_cls_prob'], dtype=torch.float32),
            # 'pred_attr_prob': torch.tensor(dict_batched['pred_attr_prob'], dtype=torch.float32),
            'pred_boxes_features': torch.tensor(dict_batched['pred_boxes_features'], dtype=torch.float32),
        }
        # print(final_batch['n_phrases'], final_batch['n_phrases'].size())
        # exit(0)
        return final_batch

    def make_string_from_idx(self, idx_list):
        """
        This function build a string starting given a list of vocabulary indexes.
        :param idx_list: list of indexes
        :return: string
        """
        s_list = [self.vocab.itos[i] for i in idx_list]
        # removing padding symbol
        s_list_filtered = []
        for i in s_list:
            if i != '<pad>':
                s_list_filtered.append(i)
        string = ' '.join(s_list_filtered)
        return string

    def load_dataset_image(self, img_id, folder=None):
        str_id = str(img_id)
        curr_img = str_id.zfill(5)
        init_idx = curr_img[:2]
        final_address = os.path.join('{}/data/images/saiapr_tc-12/{}/images/{}.jpg'.format(folder, init_idx, img_id))
        print(final_address, init_idx)
        img = utils.load_image(final_address)
        return img


def get_statistics(dataset):
    """
    Extract some data from the raw dataset.
    :param dataset: dataset.
    """
    import statistics
    queries = []  # queries
    queries_words = []
    n_queries = []  # number of queries
    sentences = []  # sentences
    sentences_words = []
    no_boxes = 0
    boxes = []
    pred_boxes = []
    for n, i in enumerate(dataset):
        n_queries.append(i['n_phrases'])
        for ph in i['phrases']:
            queries.append(len(ph))
            queries_words.append(len(ph.split(' ')))
        sentences.append(len(i['sentence']))
        sentences_words.append(len(i['sentence'].split(' ')))
        if len(i['phrases']) == 0:
            no_boxes += 1
        boxes.append(len(i['image_boxes_coordinates']))
        pred_boxes.append(len(i['pred_boxes']))
    print('Max number of queries: ', max(n_queries))
    print('Mean number of queries: ', statistics.mean(n_queries))
    print('Max length of queries: ', max(queries))
    print('Mean length of queries: ', statistics.mean(queries))
    print('Max number of words in queries: ', max(queries_words))
    print('Mean number of words in queries: ', statistics.mean(queries_words))
    print('Max length of sentences: ', max(sentences))
    print('Mean length of sentences: ', statistics.mean(sentences))
    print('Max number of words in sentences: ', max(sentences_words))
    print('Mean number of words in sentences: ', statistics.mean(sentences_words))
    print('Example without bounding boxes: ', no_boxes)
    print('Max number of boxes: ', max(boxes))
    print('Mean number of boxes: ', statistics.mean(boxes))
    print('Max number of pred boxes: ', max(pred_boxes))
    print('Min number of pred boxes: ', min(pred_boxes))
    # Save some results. Note that queries without bounding boxes are removed in make_Dataset.py
    # """ TRAINING
    # Max number of queries:  1
    # Mean number of queries:  1
    # Max length of queries:  151
    # Mean length of queries:  17.46261324811749
    # Max number of words in queries:  30
    # Mean number of words in queries:  3.4595536366881405
    # Max length of sentences:  151
    # Mean length of sentences:  17.46261324811749
    # Max number of words in sentences:  30
    # Mean number of words in sentences:  3.4595536366881405
    # Example without bounding boxes:  0
    # Max number of boxes:  67*
    # Mean number of boxes:  1.8596998181230984*
    # Max number of pred boxes:  100
    # Min number of pred boxes:  100
    #
    # """ TEST
    # Max number of queries:  1
    # Mean number of queries:  1
    # Max length of queries:  171
    # Mean length of queries:  17.393766010645642
    # Max number of words in queries:  36
    # Mean number of words in queries:  3.4447699835866916
    # Max length of sentences:  171
    # Mean length of sentences:  17.393766010645642
    # Max number of words in sentences:  36
    # Mean number of words in sentences:  3.4447699835866916
    # Example without bounding boxes:  0
    # Max number of boxes:  47*
    # Mean number of boxes:  2.228959418916785*
    # Max number of pred boxes:  100
    # Min number of pred boxes:  100
    # VALID
    # Max number of queries:  1
    # Mean number of queries:  1
    # Max length of queries:  108
    # Mean length of queries:  16.712300647402493
    # Max number of words in queries:  22
    # Mean number of words in queries:  3.3161219011526923
    # Max length of sentences:  108
    # Mean length of sentences:  16.712300647402493
    # Max number of words in sentences:  22
    # Mean number of words in sentences:  3.3161219011526923
    # Example without bounding boxes:  0
    # Max number of boxes:  47*
    # Mean number of boxes:  2.228959418916785*
    # Max number of pred boxes:  100
    # Min number of pred boxes:  100


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Inputs')
    parser.add_argument('--folder', dest='folder',
                        help='Folder including the preprocessed data.',
                        default='/home/drigoni/repository/Loss_VT_Grounding/data/referit_raw/preprocessed/',
                        type=str)
    parser.add_argument('--idx_train', dest='idx_train',
                        help='File .txt including the train indexes.',
                        default='/home/drigoni/repository/Loss_VT_Grounding/data/referit_raw/train.txt',
                        type=str)
    parser.add_argument('--idx_val', dest='idx_val',
                        help='File .txt including the valid indexes.',
                        default='/home/drigoni/repository/Loss_VT_Grounding/data/referit_raw/val.txt',
                        type=str)
    parser.add_argument('--idx_test', dest='idx_test',
                        help='File .txt including the test indexes.',
                        default='/home/drigoni/repository/Loss_VT_Grounding/data/referit_raw/test.txt',
                        type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print('SMALL TEST')
    dataset = VTKELReferitDataset(args.folder, args.idx_val)
    dataset.load_dataset_image("40031")
    # print('Len: ', len(dataset))
    # example = dataset[1000]
    # print('Example: ', example, len(example['pred_boxes_features']))  # shape: (10, 2048)
    # batch = [dataset[0], dataset[1], dataset[2], dataset[3]]
    # print('Batch: ', batch)
    # batched = dataset.collate_fn(batch)
    # print('Batched: ', batched['image_w'])
    # dataset.check_example(26)
    # get_statistics(dataset)
    print('END')
