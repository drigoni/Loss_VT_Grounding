#!/usr/bin/env python
"""
Created on 26/01/21
Author: Davide Rigoni
Emails: davide.rigoni.2@phd.unipd.it - drigoni@fbk.eu
Description: this file contains the pytorch model.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
import model_code.anchors as anchors
import torchtext


class VTKELModel(nn.Module):
    """
    Model class
    """

    def __init__(self, configs, text_vocab):
        super().__init__()
        # set params
        self.params = configs
        self.text_vocab = text_vocab
        self.device = self.params['device']

        # set initial seeds
        VTKELModel.set_seeds()

        # sentence branch
        self.text_emb = VTKELModel.create_embeddings_network(self.params['embeddings_text'],
                                                             self.text_vocab,
                                                             self.params['text_emb_size'],
                                                             self.params['embeddings_freeze'])

        self.sentence_rnn = nn.LSTM(self.params['text_emb_size'], self.params['lstm_dim'],
                                    num_layers=self.params['lstm_num_layers'], bidirectional=True,
                                    batch_first=True)
        for name, param in self.sentence_rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

        # phrases branch
        self.phrases_rnn = nn.LSTM(self.params['text_emb_size'], self.params['lstm_dim'],
                                   num_layers=self.params['lstm_num_layers'], bidirectional=False,
                                   batch_first=False)
        for name, param in self.phrases_rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        self.phrases_dropout = nn.Dropout(self.params['dropout_ratio'])

        # fusion
        self.fusion_linear_1 = nn.Linear(2053 + self.params['lstm_dim'], self.params['fusion_dim'])
        nn.init.xavier_normal_(self.fusion_linear_1.weight)
        nn.init.zeros_(self.fusion_linear_1.bias)
        self.fusion_linear_1_act = nn.LeakyReLU()

        # prediction
        self.linear_1 = nn.Linear(self.params['fusion_dim'], 1)
        nn.init.xavier_normal_(self.linear_1.weight)
        nn.init.zeros_(self.linear_1.bias)

        # regression
        self.linear_2 = nn.Linear(self.params['fusion_dim'], 4)
        nn.init.xavier_normal_(self.linear_2.weight)
        nn.init.zeros_(self.linear_2.bias)
        self.linear_act_2 = nn.ReLU()

    def forward(self, curr_batch):
        # get input data
        phrases = curr_batch['phrases']                                                 # [b, max_n_ph, max_len_ph]
        phrases_mask = curr_batch['phrases_mask']                                       # [b, max_n_ph, max_len_ph]
        phrases_length = torch.sum(phrases_mask.type(torch.long), dim=-1)               # [b, max_n_ph]
        # create a mask in order to eliminate phrases added only for padding
        mask = torch.any(phrases_mask, dim=-1, keepdim=True).type(torch.long)           # [b, max_n_ph, 1]
        pred_boxes = curr_batch['pred_boxes']                                           # [b, n_pred_box, 4]
        pred_boxes_features = curr_batch['pred_boxes_features']                         # [b, n_pred_box, 2048]
        pred_n_boxes = curr_batch['pred_n_boxes']                                       # [b]

        # set some dimensions
        max_n_ph = phrases.size()[1]
        n_pred_box = pred_n_boxes[0]

        # --- image branch
        img_x = self.get_img_features(pred_boxes_features, pred_boxes)

        # --- phrases branch
        phrases_emb = self.text_emb(phrases)                                            # [b, max_n_ph, max_len_ph, fp]
        phrases_x = self.apply_lstm_phrases(phrases_emb, phrases_length, mask)


        # --- fusion
        pred_text_rep = phrases_x.unsqueeze(2).repeat(1, 1, n_pred_box, 1)              # [b, max_n_ph, n_pred_box, fp]
        pred_img_rep = img_x.unsqueeze(1).repeat(1, max_n_ph, 1, 1)                     # [b, max_n_ph, n_pred_box, fi]
        pred_rep = torch.cat([pred_text_rep, pred_img_rep], dim=-1)                 # [b, max_n_ph, n_pred_box, fp+fi]
        fused_x = self.fusion_linear_1_act(self.fusion_linear_1(pred_rep))          # [b, max_n_ph, n_pred_box, 512]

        # --- prediction
        pred_logits = self.linear_1(fused_x).squeeze(dim=-1)                        # [b, max_n_ph, n_pred_box]

        # --- regression
        reg_offsets = self.linear_2(fused_x)                                                # [b, max_n_ph, n_pred_box, 4]
        pred_boxes_rep = pred_boxes.unsqueeze(1).repeat(1, max_n_ph, 1, 1)          # [b, max_n_ph, n_pred_box, 4]
        reg = anchors.tlbr2cthw(pred_boxes_rep) + reg_offsets
        reg = anchors.cthw2tlbr(reg)
        reg = reg * mask.unsqueeze(-1)
        reg = torch.clamp(reg, min=0, max=1)

        return pred_logits, reg

    def apply_lstm_phrases(self, phrases_emb, phrases_length, mask):
        """
        This function applies the transformation to the phrases branch.
        :param phrases_emb: embedding of each phrase
        :param phrases_length: length of each phrase
        :param mask: phrase mask
        :return: tensor of embedding features for each phrase
        """
        batch_size = phrases_emb.size()[0]
        max_n_ph = phrases_emb.size()[1]
        max_ph_len = phrases_emb.size()[2]
        # out_feats_dim = self.params['lstm_dim'] * 2
        out_feats_dim = self.params['lstm_dim']
        # [max_ph_len, b*max_n_ph, 300]
        phrases_emb = phrases_emb.view(-1, phrases_emb.size()[-2], phrases_emb.size()[-1])
        phrases_emb = phrases_emb.permute(1, 0, 2).contiguous()
        # note: we need to fix the bug about phrases with lengths 0. On cpu required by torch
        phrases_length_clamp = phrases_length.view(-1).clamp(min=1).cpu()
        phrases_pack_emb = rnn.pack_padded_sequence(phrases_emb, phrases_length_clamp, enforce_sorted=False)
        phrases_x_o, (phrases_x_h, phrases_x_c) = self.phrases_rnn(phrases_pack_emb)
        phrases_x_o = rnn.pad_packed_sequence(phrases_x_o, batch_first=False)                    # (values, length)
        # due to padding we need to get indexes in this way. On device now.
        idx = (phrases_x_o[1] - 1).unsqueeze(0).unsqueeze(-1).repeat(1, 1, phrases_x_o[0].size()[2]).to(self.device)
        phrases_x = torch.gather(phrases_x_o[0], 0, idx)                                        # [1, b*max_n_ph, 2048]
        phrases_x = phrases_x.permute(1, 0, 2).contiguous().unsqueeze(0)                        # [b*max_n_ph, 2048]
        # back to batch size dimension
        phrases_x = phrases_x.squeeze(1).view(batch_size, max_n_ph, out_feats_dim)              # [b, max_n_ph, out_feats_dim]
        phrases_x = torch.masked_fill(phrases_x, mask == 0, 0)                                  # boolean mask required
        # normalize features
        phrases_x_norm = F.normalize(phrases_x, p=1, dim=-1)
        return phrases_x_norm

    def get_img_features(self, pred_boxes_features, pred_boxes):
        """
        This function applies the transformation to the image branch
        :param pred_boxes_features: bounding boxes predicted features
        :param pred_boxes: bounding boxes predicted coordinates as [xmin, ymin, xmax, ymax]
        :return: new images features with spatial information
        """
        # l2 norm
        pred_boxes_features_l2 = F.normalize(pred_boxes_features, p=1, dim=-1)
        # concatenate spatial features. Note that we have already have the first 4 features, only
        img_x_area = anchors.tlbr2tlhw(pred_boxes)
        img_x_area = (img_x_area[..., 2] * img_x_area[..., 3]).unsqueeze(-1)                # [b, n_pred_box, 1]
        img_x = torch.cat([pred_boxes_features_l2, pred_boxes, img_x_area], dim=-1)         # [b, n_pred_box, 2048 + 5]
        return img_x

    @staticmethod
    def create_embeddings_network(embedding, vocab, text_emb_size, freeze=False):
        """
        This function creates an embedding layer.
        :param embedding: embedding type as: glove
        :param vocab: vocabulary of words
        :param text_emb_size: size of the embedding
        :param freeze: if the embedding is trained or not.
        :return: the embedding layer.
        """
        vocab_size = len(vocab)
        out_of_vocabulary = 0
        if embedding == 'glove':
            #  +1 for special char
            embedding_matrix_values = torch.zeros((vocab_size + 1, text_emb_size), requires_grad=(freeze==False))
            glove_embeddings = torchtext.vocab.GloVe('840B', dim=300)
            glove_words = glove_embeddings.stoi.keys()
            for idx in range(vocab_size):
                word = vocab.itos[idx]
                if word in glove_words:
                    glove_idx = glove_embeddings.stoi[word]
                    embedding_matrix_values[idx, :] = glove_embeddings.vectors[glove_idx]
                else:
                    out_of_vocabulary += 1
                    nn.init.normal_(embedding_matrix_values[idx, :])
            embedding_matrix = nn.Embedding(vocab_size, text_emb_size)
            embedding_matrix.weight = torch.nn.Parameter(embedding_matrix_values)
            embedding_matrix.weight.requires_grad = (freeze==False)
            return embedding_matrix

        else:
            return nn.Embedding(vocab_size, text_emb_size)

    @staticmethod
    def set_seeds(seed=42):
        """
        This function set the seeds.
        :param seed:
        :return:
        """
        np.random.seed(seed)
        torch.manual_seed(seed)

