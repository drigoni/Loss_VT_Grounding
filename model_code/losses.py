#!/usr/bin/env python
"""
Created on 09/02/21
Author: Davide Rigoni
Emails: davide.rigoni.2@phd.unipd.it - drigoni@fbk.eu
Description: this file contain the loss of the model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import model_code.anchors as anchors


class VTKELSolverLoss(nn.Module):
    """
    This class computes the model loss.
    """

    def __init__(self, params):
        super(VTKELSolverLoss, self).__init__()
        self.params = params
        self.BCEL = torch.nn.BCELoss(reduction='none')
        self.CEL = torch.nn.CrossEntropyLoss(reduction='none')
        self.MSE = torch.nn.MSELoss(reduction='none')
        self.smoothL1 = torch.nn.SmoothL1Loss(reduction='none')
        self.KL = torch.nn.KLDivLoss(reduction='none')

    def forward(self, output, curr_batch):
        """
        This function calculate the loss.
        :param output: model results.
        :param curr_batch: batch data.
        :return: loss.
        """
        # inputs
        phrases_2_crd = curr_batch['phrases_2_crd']                                 # [b, max_n_ph, 4]
        sentence_mask = curr_batch['sentence_mask']                                 # [b, len_sent]
        phrases_mask = curr_batch['phrases_mask']                                   # [b, max_n_ph, max_len_ph]
        # create a mask in order to eliminate phrases added only for padding
        mask = torch.any(phrases_mask, dim=-1, keepdim=True).type(torch.int32)      # [b, max_n_ph, 1]
        pred_boxes = curr_batch['pred_boxes']                                       # [b, n_pred_box, 4]
        pred_cls_prob = curr_batch['pred_cls_prob']                                 # [b, n_pred_box, n_classes]
        pred_score = output[0]                                                      # [b, max_n_ph, n_pred_box]
        pred_reg = output[1]                                                        # [b, max_n_ph, n_pred_box, 4]
        # set some dimensions
        batch_size = pred_score.size()[0]
        max_n_ph = pred_score.size()[1]
        n_pred_box = pred_score.size()[2]

        with torch.no_grad():
            # --- IOU between refined bounding boxes and GT
            phrases_2_crd_ext = torch.unsqueeze(phrases_2_crd, 2).repeat(1, 1, n_pred_box, 1)       # [b, max_n_ph, n_pred_box, 4]
            iou_scores_ref, align_ref2gt = self.get_iou_scores(pred_reg, phrases_2_crd_ext, mask)

            # --- IOU between object detector proposal bounding boxes and GT
            pred_boxes = pred_boxes.unsqueeze(1).repeat(1, max_n_ph, 1, 1)                          # [b, max_n_ph, n_pred_box, 4]
            iou_scores_prop, align_prop2gt = self.get_iou_scores(pred_boxes, phrases_2_crd_ext, mask)

            if self.params['align_loss'] == 'kl-sem' or self.params['regression_loss'] == 'iou_c-sem':
                iousem_scores_prop, align_iousem2gt = self.get_iouSem_scores(iou_scores_prop, align_prop2gt,
                                                                             pred_cls_prob,
                                                                             threshold=self.params['align_loss_kl_threshold'])
            # --- METRICS
            accuracy = self.get_accuracy(pred_score, iou_scores_ref, mask)
            point_accuracy = self.get_point_accuracy(pred_score, pred_reg, phrases_2_crd, mask)

        # --- Classification LOSS
        if self.params['align_loss'] == 'ce':
            classification_loss = self.get_classification_loss_ce(pred_score, mask, align_prop2gt)
        elif self.params['align_loss'] == 'kl':
            classification_loss = self.get_classification_loss_kl(pred_score, mask, iou_scores_prop,
                                                                  threshold=self.params['align_loss_kl_threshold'])
        elif self.params['align_loss'] == 'kl-sem':
            classification_loss = self.get_classification_loss_kl_sem(pred_score, mask, iousem_scores_prop)
        else:
            print('Error, loss not recognized.')
            exit(1)

        # --- REG LOSS
        if self.params['regression_loss'] == 'reg':
            reg_loss = self.get_regression_loss(pred_reg, mask, align_prop2gt, phrases_2_crd)
        elif self.params['regression_loss'] == 'iou_c-sem':
            reg_loss = self.get_regression_loss_iou_sem(pred_reg, mask, iousem_scores_prop, phrases_2_crd)
        elif self.params['regression_loss'] == 'iou_c':
                reg_loss = self.get_regression_loss_iou(pred_reg, mask, align_prop2gt, phrases_2_crd)
        else:
            print('Error, loss not recognized.')
            exit(1)

        final_loss = self.params['loss_weight_pred'] * classification_loss + self.params['loss_weight_reg'] * reg_loss

        return final_loss, classification_loss.detach(), reg_loss.detach(), accuracy, \
               iou_scores_ref, point_accuracy

    def get_accuracy(self, pred_score, iou_scores, mask):
        """
        Calculate accuracy as state of the art. This mean that we take the best sphrases_2_crdcore for each
        pair phrase-proposal and we calculate the iou among the refined coordinates and the GT. If the iou
        is higher than 0.5 than it is 1, else a 0.
        :param pred_score: predicted phrase-proposal score
        :param iou_scores: iou score among refined bounding boxes and gt
        :param mask: phrases mask.
        :return: accuracy
        """
        pred_best_score_idx = torch.argmax(pred_score, dim=2, keepdim=True)             # [b, max_n_ph, 1]
        pred_best_score_idx = pred_best_score_idx.unsqueeze(-1).repeat(1, 1, 1, 1)      # [b, max_n_ph, 1, 1]
        accuracy_tmp = torch.gather(iou_scores, 2, pred_best_score_idx).squeeze(2)      # [b, max_n_ph, 1]
        # iou higher than 0.5
        accuracy = accuracy_tmp[:, :, 0] > 0.5                                         # [b, max_n_ph]
        accuracy = accuracy.type(torch.int32) * mask.squeeze(dim=-1)
        accuracy = torch.sum(accuracy) / mask.sum()
        return accuracy

    def get_point_accuracy(self, pred_score, pred_reg, phrases_2_crd, mask):
        """
        Calculate the point game accuracy as some paper usually does. Usually the works which use this metric do
        an unfair comparison. We report this just to show the difference.
        We take the best score for each pair phrase-proposal and we calculate its central point.
        Then if the central point is inside the ground truth bounding box, than it is a 1, else a 0.
        Of course this metric give better results thant the real accuracy.
        :param pred_score: predicted phrase-proposal score
        :param pred_reg: predicted phrase-proposal refined bounding boxes
        :param phrases_2_crd: phrases ground truth
        :param mask: phrases mask
        :return: point game accuracy
        """
        pred_best_score_idx = torch.argmax(pred_score, dim=2, keepdim=True)             # [b, max_n_ph, 1]
        pred_best_score_idx = pred_best_score_idx.unsqueeze(-1).repeat(1, 1, 1, 4)      # [b, max_n_ph, 1, 4]
        pred_reg_tmp = torch.gather(pred_reg, 2, pred_best_score_idx).squeeze(2)        # [b, max_n_ph, 4]
        # coordinates conversion
        pred_reg_tmp_conv = anchors.tlbr2cthw(pred_reg_tmp)
        # calculate if the predicted center of the bounding boxes are inside the gt bounding boxes
        x_point = pred_reg_tmp_conv[..., 0]
        y_point = pred_reg_tmp_conv[..., 1]
        accuracy_x = torch.logical_and(phrases_2_crd[..., 0] <= x_point, x_point <= phrases_2_crd[..., 2])
        accuracy_y = torch.logical_and(phrases_2_crd[..., 1] <= y_point, y_point <= phrases_2_crd[..., 3])
        # final accuracy
        accuracy = torch.logical_and(accuracy_x, accuracy_y)
        accuracy = accuracy.type(torch.int32) * mask.squeeze(dim=-1)
        accuracy = torch.sum(accuracy) / mask.sum()
        return accuracy

    def get_regression_loss(self, pred_reg, mask, align_prop2gt, phrases_2_crd):
        """
        Standard regression loss with SmoothL1.
        :param pred_reg: predicted coordinates.
        :param mask: mask.
        :param align_prop2gt: best proposals with ground truth.
        :param phrases_2_crd: ground truth coordinates.
        :return: loss.
        """
        # take the predicted bounding boxes which overlap most with the ground truth
        reg_idx = align_prop2gt.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 4)  # [b, max_n_ph, 1, 4]
        pred_align_coordinates = torch.gather(pred_reg, 2, reg_idx).squeeze(2)  # [b, max_n_ph, 4]
        # coordinates conversion
        pred_crds = anchors.tlbr2cthw(pred_align_coordinates)
        pred_crds_gt = anchors.tlbr2cthw(phrases_2_crd)
        reg_loss = self.smoothL1(pred_crds, pred_crds_gt) * mask
        reg_loss = torch.sum(reg_loss) / mask.sum()
        return reg_loss

    def get_regression_loss_iou(self, pred_reg, mask, align_prop2gt, phrases_2_crd):
        """
        Complete IoU loss.
        :param pred_reg: predicted coordinates.
        :param mask: mask.
        :param align_prop2gt: best proposals with ground truth.
        :param phrases_2_crd: ground truth coordinates.
        :return: loss.
        """
        # takes boxes with higher probabilities
        reg_idx = align_prop2gt.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 4)  # [b, max_n_ph, 1, 4]
        pred_align_coordinates = torch.gather(pred_reg, 2, reg_idx).squeeze(2)  # [b, max_n_ph, 4]
        # calculates iou
        iou_scores = anchors.bbox_final_iou(pred_align_coordinates, phrases_2_crd, CIoU=True).unsqueeze(-1)
        iou_scores = (1.0 - iou_scores) * mask
        iou_scores = torch.sum(iou_scores) / mask.sum()
        return iou_scores

    def get_regression_loss_iou_sem(self, pred_reg, mask, iousem_scores_prop, phrases_2_crd, eps=1e-09):
        """
        Our loss.
        :param pred_reg: predicted coordinates.
        :param mask: mask.
        :param iousem_scores_prop: semantic score used to weight the loss.
        :param phrases_2_crd: ground truth coordinates.
        :param eps: small epsilon.
        :return: loss.
        """
        n_proposal = iousem_scores_prop.size()[2]
        phrases_2_crd_ext = phrases_2_crd.unsqueeze(2).repeat(1, 1, n_proposal, 1)
        iousem_scores_prop_norm = iousem_scores_prop / (torch.max(iousem_scores_prop, dim=-1, keepdim=True)[0] + eps)
        # calculates iou
        iou_scores = anchors.bbox_final_iou(pred_reg, phrases_2_crd_ext, CIoU=True)
        iou_scores = (1.0 - iou_scores) * iousem_scores_prop_norm
        iou_scores = iou_scores * mask
        iou_scores = torch.sum(iou_scores) / mask.sum()
        return iou_scores

    def get_classification_loss_ce(self, pred_score, mask, align_prop2gt):
        """
        Cross Entropy implementation for loss.
        :param pred_score: predicted scores for each proposal
        :param mask: phrases mask
        :param align_prop2gt: ground truth for each phrase
        :return: masked loss.
        """
        # create gt with one hot encoder -> prob=1 for only one bounding box
        bce_loss = self.CEL(pred_score.permute(0, 2, 1), align_prop2gt)                     # [b, max_n_ph]
        bce_loss = bce_loss * mask.squeeze(dim=-1)                                          # [b, max_n_ph]
        bce_loss = torch.sum(bce_loss) / mask.sum()                                         # []
        return bce_loss

    def get_classification_loss_kl(self, pred_score, mask, iou_scores_prop, threshold=0.5):
        """
        Implementation of kl loss for classification
        :param pred_score: predicted score for each bounding box
        :param mask: phrases mask
        :param iou_scores_prop: iou score between the proposed bounding boxes coordinates and the GT
        :param threshold: iou threshold
        :return: classification loss
        """
        # calculate distribution with a threshold
        iou_scores = iou_scores_prop.squeeze(3)                                               # [b, max_n_ph, n_pred_box]
        iou_scores_cl = iou_scores.masked_fill(iou_scores < threshold, 0)                   # [b, max_n_ph, n_pred_box]
        if torch.sum(iou_scores_cl) > 0:
            iou_scores_l1 = F.normalize(iou_scores_cl, p=1, dim=2)                          # [b, max_n_ph, n_pred_box]
        else:
            _, align_prop2gt_ciou =torch.max(iou_scores_prop.squeeze(-1), dim=-1)
            iou_scores_l1 = F.one_hot(align_prop2gt_ciou, iou_scores.size()[2])
        # get predicted probabilities from logits
        pred_prob = F.log_softmax(pred_score, dim=2)                                        # [b, max_n_ph, n_pred_box]
        kl_loss = self.KL(pred_prob, iou_scores_l1)                                         # [b, max_n_ph, n_pred_box]
        kl_loss = kl_loss * mask                                                            # [b, max_n_ph, n_pred_box]
        kl_loss = torch.sum(kl_loss) / mask.sum()                                           # []
        return kl_loss

    def get_classification_loss_kl_sem(self, pred_score, mask, scores_masked):
        """
        Implementation of ours kl loss for classification
        :param pred_score: predicted score for each bounding box
        :param mask: phrases mask
        :param scores_masked: iou with semantic score.
        :return: classification loss
        """
        target_prob = F.normalize(scores_masked, p=1, dim=2)
        # get predicted probabilities from logits
        pred_prob = F.log_softmax(pred_score, dim=2)                                        # [b, max_n_ph, n_pred_box]
        kl_loss = self.KL(pred_prob, target_prob)                                         # [b, max_n_ph, n_pred_box]
        kl_loss = kl_loss * mask                                                            # [b, max_n_ph, n_pred_box]
        kl_loss = torch.sum(kl_loss) / mask.sum()                                           # []
        return kl_loss

    def get_iou_scores(self, boxes, gt, mask, GIoU=False, DIoU=False, CIoU=False):
        """
        Return the IoU score.
        :param boxes: predicted boxes.
        :param gt: ground truth boxes.
        :param mask: mask.
        :param GIoU: Generic IoU.
        :param DIoU: Distance IoU.
        :param CIoU: Complete IoU.
        :return: the IoU score and the argmax of it for each phrase.
        """
        iou_scores_ref = anchors.bbox_final_iou(boxes, gt, GIoU=GIoU, DIoU=DIoU, CIoU=CIoU).unsqueeze(-1)  # [b, max_n_ph, n_pred_box, 1]
        iou_scores_ref = iou_scores_ref * mask.unsqueeze(-1)
        # find predicted bounding boxes with highest iou
        _, align = torch.max(iou_scores_ref.squeeze(-1), dim=-1)  # [b, max_n_ph]
        return iou_scores_ref, align

    def get_iouSem_scores(self, iou_scores_prop, align_prop2gt, pred_cls_prob, threshold=0.5):
        """
        Our semantic IoU score.
        :param iou_scores_prop: iou score among proposals and ground truth.
        :param align_prop2gt: argmax of the iou_scores_prop scores for each phrase.
        :param pred_cls_prob: predicted class probabilities for each proposal.
        :param threshold: threshold to consider.
        :return: semantic score
        """
        # get right size
        n_cls = pred_cls_prob.size()[-1]
        max_n_ph = align_prop2gt.size()[1]
        n_pred_box = pred_cls_prob.size()[1]
        pred_cls_prob_exp = pred_cls_prob.unsqueeze(1).repeat(1, max_n_ph, 1, 1)        # [b, max_n_ph, n_pred_box, n_cls]

        # get best proposals classes
        cls_idx = align_prop2gt.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, n_cls)      # [b, max_n_ph, 1, n_cls]
        pred_align_classes = torch.gather(pred_cls_prob_exp, 2, cls_idx)                # [b, max_n_ph, 1, n_cls]
        pred_align_classes = torch.transpose(pred_align_classes, 3, 2)                  # [b, max_n_ph, n_cls, 1]

        # get iou mask, calculate distribution with a threshold
        iou_scores = iou_scores_prop.squeeze(3)                                             # [b, max_n_ph, n_pred_box]
        iou_scores_cl = iou_scores.masked_fill(iou_scores < threshold, 0)                   # [b, max_n_ph, n_pred_box]
        if torch.sum(iou_scores_cl) == 0:
            _, align_prop2gt_ciou = torch.max(iou_scores_prop.squeeze(-1), dim=-1)
            iou_scores_cl = F.one_hot(align_prop2gt_ciou, iou_scores.size()[2])

        pred_align_classes = torch.transpose(pred_align_classes, 3, 2).repeat(1, 1, n_pred_box, 1)  # [b, max_n_ph, n_pred_box, n_cls]
        cosine_sim = F.cosine_similarity(pred_cls_prob_exp, pred_align_classes, dim=-1)
        scores_masked = cosine_sim * iou_scores_cl
        # scores_masked = scores.squeeze(-1) * iou_scores_mask
        return scores_masked, torch.max(scores_masked, dim=-1)[1]
