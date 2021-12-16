#!/usr/bin/env python
"""
Created on 26/01/21
Author: Davide Rigoni
Emails: davide.rigoni.2@phd.unipd.it - drigoni@fbk.eu
Description: this file extracts the finromation about the referit dataset.
"""

import argparse
import os.path as osp
import json
import pickle as pickle
from collections import defaultdict


def get_annotations(ref_ann, ref_inst_ann):
	instance_dict_by_ann_id = {
		v['id']: ind for ind, v in enumerate(ref_inst_ann)}
	out_dict_list = []
	for rj in ref_ann:
		spl = rj['split']
		sents = rj['sentences']
		ann_id = rj['ann_id']
		inst_bbox = ref_inst_ann[instance_dict_by_ann_id[ann_id]]['bbox']
		# Saving in [x0, y0, x1, y1] format
		inst_bbox = [inst_bbox[0], inst_bbox[1],
					 inst_bbox[2] + inst_bbox[0], inst_bbox[3] + inst_bbox[1]]

		sents = [s['raw'] for s in sents]
		sents = [t.strip().lower() for t in sents]
		out_dict = {}
		out_dict['img_id'] = f"{rj['image_id']}.jpg"
		out_dict['ann_id'] = ann_id
		out_dict['bbox'] = inst_bbox
		out_dict['split'] = spl
		out_dict['query'] = sents
		out_dict_list.append(out_dict)
	return out_dict_list


def referit_load_data(data_root):
	splitBy = 'berkeley'
	data_dir = osp.join(data_root, 'refclef')
	ref_ann_file = osp.join(data_dir, f'refs({splitBy}).p')
	ref_instance_file = osp.join(data_dir, 'instances.json')
	ref_ann = pickle.load(open(ref_ann_file, 'rb'))
	ref_inst = json.load(open(ref_instance_file, 'r'))
	ref_inst_ann = ref_inst['annotations']
	return ref_ann, ref_inst_ann


def referit_get_trn_val_test_ids(output_annot):
	final = defaultdict(list)
	for el in output_annot:
		final[el['split']].append(el)
	return final['train'], final['val'], final['test']

def referit_get_idx_split(output_annot):
	final = defaultdict(list)
	for el in output_annot:
		final[el['split']].append(el['ann_id'])
	return final['train'], final['val'], final['test']


def parse_args():
	"""
	Parse input arguments.
	"""
	parser = argparse.ArgumentParser(description='Inputs')
	parser.add_argument('--root', dest='root',
						help='Flickr30k root folder.',
						default='/home/drigoni/repository/Loss_VT_Grounding/data/',
						type=str)
	parser.add_argument('--id', dest='id',
						help='Flickr30k id.',
						default='48296285',
						type=str)

	args = parser.parse_args()
	return args


if __name__ == "__main__":
	print('SMALL TEST')
	args = parse_args()
	root = args.root
	data_root = root + 'refer/data/'
	ref_ann, ref_inst_ann = referit_load_data(data_root)
	annotations = get_annotations(ref_ann, ref_inst_ann)
	for i in annotations:
		if len(i['query']) > 50:
			print('block')
	trn_ids_mask, val_ids_mask, test_ids_mask = referit_get_trn_val_test_ids(annotations)
	trn_ids_mask, val_ids_mask, test_ids_mask = referit_get_idx_split(annotations)
	print('END')
