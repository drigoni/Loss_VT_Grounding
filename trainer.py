#!/usr/bin/env python
"""
Created on 26/01/21
Author: Davide Rigoni
Emails: davide.rigoni.2@phd.unipd.it - drigoni@fbk.eu
Description: this file contains the trainer model.
"""
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torchtext
from torchtext.data import Field, TabularDataset, BucketIterator
from torchtext.datasets import text_classification
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from model_code.dataset_flickr30k import VTKELFlickr30kDataset
from model_code.dataset_referit import VTKELReferitDataset
from model_code.losses import VTKELSolverLoss
from model_code.utils import torch_pad
from model_code.model import VTKELModel
import torchvision
import random
import json
from model_code.utils import load_image, show_image
import os

# temporary fix for the runtime error in cluster
torch.multiprocessing.set_sharing_strategy('file_system')


class Trainer:
    params = {
        "mode": 0,
        "dataset": "flickr",
        "restore": None,
        "suffix": 'default',
        "develop": True,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),

        # dataloader
        "batch_size": 64,
        "num_workers": 0,
        "prefetch_factor": 2,
        "load_subset": None,
        "load_first": False,
        "load_first_img": False,

        # learning
        "learning_rate": 0.001,
        "grad_clipping": 1,
        "scheduler_gamma": 0.9,
        "n_epochs": 30,
        "align_loss": "kl-sem",
        "align_loss_kl_threshold": 0.5,
        "regression_loss": "iou_c-sem",
        "dropout_ratio": 0.3,
        'loss_weight_pred': 1,
        'loss_weight_reg': 1,

        # network size
        "embeddings_text": "glove",
        "embeddings_freeze": True,
        "lstm_dim": 500,
        "lstm_num_layers": 1,
        "fusion_dim": 2053,
        "text_emb_size": 300,
    }
    params_flickr ={
        "folder_img": "/home/drigoni/repository/Loss_VT_Grounding/data/flickr30k/flickr30k_images",
        "folder_results": "/home/drigoni/repository/Loss_VT_Grounding/results/flickr30k",
        "folder_data": "/home/drigoni/repository/Loss_VT_Grounding/data/flickr30k_raw/preprocessed",
        "folder_idx_train": "/home/drigoni/repository/Loss_VT_Grounding/data/flickr30k/flickr30k_entities/train.txt",
        "folder_idx_valid": "/home/drigoni/repository/Loss_VT_Grounding/data/flickr30k/flickr30k_entities/val.txt",
        "folder_idx_test": "/home/drigoni/repository/Loss_VT_Grounding/data/flickr30k/flickr30k_entities/test.txt",
    }
    params_referit = {
        "folder_img": "/home/drigoni/repository/Loss_VT_Grounding/data/refer",
        "folder_results": "/home/drigoni/repository/Loss_VT_Grounding/results/referit",
        "folder_data": "/home/drigoni/repository/Loss_VT_Grounding/data/referit_raw/preprocessed",
        "folder_idx_train": "/home/drigoni/repository/Loss_VT_Grounding/data/referit_raw/train.txt",
        "folder_idx_valid": "/home/drigoni/repository/Loss_VT_Grounding/data/referit_raw/val.txt",
        "folder_idx_test": "/home/drigoni/repository/Loss_VT_Grounding/data/referit_raw/test.txt",
    }

    def __init__(self, configs=None):
        super().__init__()
        # set params
        self.params.update(self.params_flickr)
        if configs:
            loaded_dict = json.loads(configs)
            if "dataset" in loaded_dict.keys() and loaded_dict['dataset'] == 'referit':
                self.params.update(self.params_referit)
            self.params.update(loaded_dict)

        print("Model started with the following parameters: ")
        print(self.params)
        # prepare tensorboard, default folder_results/suffix/
        if self.params['develop']:
            self.writer = SummaryWriter("{folder}/{suffix}/".format(folder=self.params['folder_results'],
                                                                    suffix=self.params['suffix']))
        else:
            self.writer = None

        # load datasets
        self.train_dataset, self.valid_dataset, self.test_dataset = self.load_datasets()
        self.vocab_size = len(self.train_dataset.vocab)

        # preparing model
        VTKELModel.set_seeds()
        self.start_epoch = 0
        self.device = self.params['device']
        self.model = VTKELModel(self.params, self.train_dataset.vocab)
        self.model.to(self.device)

        # define optimization
        self.min_valid_loss = float('inf')
        self.criterion = VTKELSolverLoss(self.params)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.params['learning_rate'])
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, self.params['scheduler_gamma'])

        # restore a checkpoint
        if self.params['restore'] is not None:
            self.load_model(self.params['restore'])

        # check model status
        if self.params['mode'] == 0:
            # train
            self.train_model()
        elif self.params['mode'] == 1:
            # test
            self.test_model()
        elif self.params['mode'] == 2:
            # test
            self.test_example()
        elif self.params['mode'] == 3:
            # test
            self.test_model()
            self.valid_model()
        else:
            print('Error. Mode not recognized.')
            exit(1)

    def execute_epoch(self, dataset, train=True, develop=False):
        epoch_mode = 'Train' if train else 'Valid'
        # change mode
        if train:
            self.model.train(True)
        else:
            self.model.train(False)
        # Train the model
        cum_n_proc_element = 0
        cum_loss = 0
        cum_pred = 0
        cum_reg = 0
        cum_acc = 0
        cum_pacc = 0
        data = DataLoader(dataset, batch_size=self.params['batch_size'], shuffle=True,
                          collate_fn=dataset.collate_fn,
                          num_workers=self.params['num_workers'], prefetch_factor=self.params['prefetch_factor'])
        n_batch = len(data)
        for i, curr_batch in enumerate(data):
            # if i == 395:
            #     print('block)')
            # move data on the selected device
            for key, value in curr_batch.items():
                curr_batch[key] = value.to(self.device)
            # reset gradient
            self.optimizer.zero_grad()
            # model forward
            output = self.model(curr_batch)
            loss, bcel, msel, accuracy, iou, paccuracy = self.criterion(output, curr_batch)
            curr_batch_examples = np.sum(curr_batch['n_phrases'].tolist())
            cum_n_proc_element += curr_batch_examples
            cum_loss += loss.item() * curr_batch_examples
            cum_reg += msel.item() * curr_batch_examples
            cum_pred += bcel.item() * curr_batch_examples
            cum_acc += accuracy.item() * curr_batch_examples
            cum_pacc += paccuracy.item() * curr_batch_examples
            if train:
                loss.backward()
                if self.params['grad_clipping'] is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.params['grad_clipping'])
                self.optimizer.step()
            # print current results
            tmp_loss = cum_loss / cum_n_proc_element
            tmp_reg = cum_reg / cum_n_proc_element
            tmp_pred = cum_pred / cum_n_proc_element
            tmp_acc = (cum_acc / cum_n_proc_element) * 100
            tmp_pacc = (cum_pacc / cum_n_proc_element) * 100
            print("{mode} {batch}/{total}, loss: {loss:2.6f}, reg_loss: {reg:2.6f}, pred_loss: {pred:2.6f}, "
                  "acc: {acc:3.4f}, pacc: {pacc:3.4f} ."
                  .format(batch=str(i + 1), total=n_batch, mode=epoch_mode, loss=tmp_loss, reg=tmp_reg, pred=tmp_pred,
                          acc=tmp_acc, pacc=tmp_pacc),
                  end='\r')
        epoch_loss = cum_loss / cum_n_proc_element
        epoch_reg = cum_reg / cum_n_proc_element
        epoch_pred = cum_pred / cum_n_proc_element
        epoch_acc = (cum_acc / cum_n_proc_element) * 100
        epoch_pacc = (cum_pacc / cum_n_proc_element) * 100
        # print tensorboard data
        if self.params['develop'] and develop:
            self.writer.add_scalar('{mode}/Loss'.format(mode=epoch_mode), epoch_loss)
            self.writer.add_scalar('{mode}/Regression'.format(mode=epoch_mode), epoch_reg)
            self.writer.add_scalar('{mode}/Prediction'.format(mode=epoch_mode), epoch_pred)
            self.writer.add_scalar('{mode}/Accuracy'.format(mode=epoch_mode), epoch_acc)
            self.writer.add_scalar('{mode}/PAccuracy'.format(mode=epoch_mode), epoch_pacc)
            self.writer.flush()

        # if training, adjust the learning rate at the end of each epoch
        if train:
            self.scheduler.step()

        return epoch_loss, epoch_reg, epoch_pred, epoch_acc, epoch_pacc

    def train_model(self):
        print('------------- START MODEL TRAINING')
        for epoch in range(self.start_epoch, self.params['n_epochs']):
            print("----- Epoch: {i}".format(i=epoch+1))
            # take time
            start_time = time.time()
            # training and validation
            train_loss, train_reg, train_pred, train_acc, train_pacc = self.execute_epoch(self.train_dataset, train=True)
            print("--- Training completed.   Loss: {loss:2.6f}, Reg_loss: {reg:2.6f}, Pred_loss: {pred:2.6f}, "
                  "Accuracy: {acc:3.4f}, PAccuracy: {pacc:3.4f} ."
                  .format(loss=train_loss, reg=train_reg, pred=train_pred, acc=train_acc, pacc=train_pacc))
            valid_loss, valid_reg, valid_pred, valid_acc, valid_pacc= self.execute_epoch(self.valid_dataset, train=False)
            print("--- Validation completed.   Loss: {loss:2.6f}, Reg_loss: {reg:2.6f}, Pred_loss: {pred:2.6f}, "
                  "Accuracy: {acc:3.4f}, PAccuracy: {pacc:3.4f} ."
                  .format(loss=valid_loss, reg=valid_reg, pred=valid_pred, acc=valid_acc, pacc=valid_pacc))

            # save model
            self.save_model(epoch + 1)
            # print results
            secs = int(time.time() - start_time)
            hours = secs // 3600
            mins = (secs - hours * 3600) // 60
            secs = secs % 60
            print('Epoch {i} completed in {h} hours, {m} minute and {s} seconds .'.format(i=epoch+1, h=hours,
                                                                                          m=mins, s=secs))

        # close writer if it is needed
        if self.writer is not None:
            self.writer.close()
        print('Model training end.')

    def test_model(self):
        print('------------- START MODEL TEST')
        # take time
        start_time = time.time()
        test_loss, test_reg, test_pred, test_acc, test_pacc = self.execute_epoch(self.test_dataset, False, False)
        print("--- Test completed.   Loss: {loss:2.6f}, Reg_loss: {reg:2.6f}, Pred_loss: {pred:2.6f}, "
              "Accuracy: {acc:3.4f}, PAccuracy: {pacc:3.4f} ."
              .format(loss=test_loss, reg=test_reg, pred=test_pred, acc=test_acc, pacc=test_pacc))

        # print results
        secs = int(time.time() - start_time)
        hours = secs // 3600
        mins = (secs - hours * 3600) // 60
        secs = secs % 60
        print('Test completed in {h} hours, {m} minute and {s} seconds .'.format(h=hours, m=mins, s=secs))
        print('Model testing end.')

    def valid_model(self):
        print('------------- START MODEL VALID')
        # take time
        start_time = time.time()
        test_loss, test_reg, test_pred, test_acc, test_pacc = self.execute_epoch(self.valid_dataset, False, False)
        print("--- Valid completed.   Loss: {loss:2.6f}, Reg_loss: {reg:2.6f}, Pred_loss: {pred:2.6f}, "
              "Accuracy: {acc:3.4f}, PAccuracy: {pacc:3.4f} ."
              .format(loss=test_loss, reg=test_reg, pred=test_pred, acc=test_acc, pacc=test_pacc))

        # print results
        secs = int(time.time() - start_time)
        hours = secs // 3600
        mins = (secs - hours * 3600) // 60
        secs = secs % 60
        print('Valid completed in {h} hours, {m} minute and {s} seconds .'.format(h=hours, m=mins, s=secs))
        print('Model validation end.')

    def test_example(self):
        print('------------- START EXAMPLE TEST ')
        data = DataLoader(self.test_dataset, batch_size=1, shuffle=True,
                          collate_fn=self.test_dataset.collate_fn,
                          num_workers=self.params['num_workers'], prefetch_factor=self.params['prefetch_factor'])
        for i, curr_batch in enumerate(data):
            sentence = self.valid_dataset.make_string_from_idx(curr_batch['sentence'].numpy()[0])
            phrases = [self.valid_dataset.make_string_from_idx(i) for i in curr_batch['phrases'].numpy()[0]]
            real_boxes = curr_batch['phrases_2_crd'].numpy()[0]
            id = curr_batch['id'].numpy()[0]
            height = curr_batch['image_h'].numpy()[0]
            width = curr_batch['image_w'].numpy()[0]
            print("Img: {} ({}x{})".format(id, width, height))
            # move data on the selected device
            for key, value in curr_batch.items():
                curr_batch[key] = value.to(self.device)
            # reset gradient
            self.optimizer.zero_grad()
            # model forward
            pred_logits, reg = self.model(curr_batch)
            loss, bcel, msel, accuracy, iou, paccuracy = self.criterion((pred_logits, reg), curr_batch)
            pred_logits = pred_logits.detach()[0]                                             # [n_ph, n_boxes]
            pred_prob = F.softmax(pred_logits, -1).numpy()
            reg = reg.detach().numpy()[0]                                               # [n_ph, n_boxes, 4]
            iou = iou.detach().numpy()[0]                                               # [n_ph, n_boxes, 1]
            # scale bounding boxes
            reg[:, :, 0] = np.round(reg[:, :, 0] * width)
            reg[:, :, 2] = np.round(reg[:, :, 2] * width)
            reg[:, :, 1] = np.round(reg[:, :, 1] * height)
            reg[:, :, 3] = np.round(reg[:, :, 3] * height)
            # scale bounding boxes
            real_boxes[:, 0] = np.round(real_boxes[:, 0] * width)
            real_boxes[:, 2] = np.round(real_boxes[:, 2] * width)
            real_boxes[:, 1] = np.round(real_boxes[:, 1] * height)
            real_boxes[:, 3] = np.round(real_boxes[:, 3] * height)
            # concatenation
            res = np.concatenate([reg, pred_prob[:, :, np.newaxis], iou], axis=-1)
            idx_max = np.argmax(res[:, :, 4], axis=-1)
            idx_val = res[np.arange(np.size(idx_max)), idx_max, :]

            print("Sentence: ", sentence)
            print("GT bounding boxes: ", real_boxes)
            # some print
            for pred_values, phrase in zip(idx_val, phrases):
                print("Phrase: ", phrase)
                print("Pred: ", pred_values[:-2])
                print("Prob: ", pred_values[-2])
                print("iou: ", pred_values[-1])
            # loading img
            # print(self.params['folder_img'], id)
            img = self.test_dataset.load_dataset_image(id, folder=self.params['folder_img'])
            show_image(img, bbox_pred=idx_val[:, :4], bbox_gt=real_boxes, bbox_query=phrases, title=id, sentence=sentence)

        print('Model testing end.')

    def save_model(self, epoch):
        file = "{folder}/model_{suffix}_{epoch}.pth".format(folder=self.params['folder_results'],
                                                            suffix=self.params['suffix'],
                                                            epoch=epoch)
        print("Saving model: {file} .".format(file=file), end='\r')
        tmp = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }
        torch.save(tmp, file)
        print("Saved model: {file} .".format(file=file))

    def load_model(self, file):
        print("Loading model: {file} .".format(file=file), end='\r')
        checkpoint = torch.load(file, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch']
        print("Loaded model: {file} .".format(file=file))

    def load_datasets(self):
        if self.params['dataset'] == "flickr":
            # if "flickr" in self.params['dataset']:
            selected_dataset = VTKELFlickr30kDataset
        else:
            selected_dataset = VTKELReferitDataset
        print("Loading training dataset.")
        train_dataset = selected_dataset(self.params['folder_data'], self.params['folder_idx_train'],
                                         load_first=self.params['load_first'],
                                         load_first_img=self.params['load_first_img'],
                                         load_subset=self.params['load_subset'])
        print("Loading validation dataset.")
        valid_dataset = selected_dataset(self.params['folder_data'], self.params['folder_idx_valid'],
                                         load_first=self.params['load_first'],
                                         load_first_img=self.params['load_first_img'],
                                         load_subset=self.params['load_subset'])
        print("Loading test dataset.")
        test_dataset = selected_dataset(self.params['folder_data'], self.params['folder_idx_test'],
                                        load_first=self.params['load_first'],
                                        load_first_img=self.params['load_first_img'],
                                        load_subset=self.params['load_subset'])
        return train_dataset, valid_dataset, test_dataset


def parse_args():
    """
    Parse input arguments.
    """
    parser = argparse.ArgumentParser(description='Inputs')
    parser.add_argument('--configs', dest='configs',
                        help='Parameters to use passed as json dictionary.',
                        default=None,
                        type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(args.configs)
