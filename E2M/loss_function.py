# -*- coding: utf-8 -*-

import torch.nn as nn
import torch


class MultiLabelLoss(nn.Module):

    def __init__(self, type_loss, reduction='sum'):
        super().__init__()
        self.type_loss = type_loss
        assert self.type_loss in ['log_sum',
                                  'sum_log',
                                  'sum_log_nce',
                                  'max_min']
        self.reduction = reduction

    def forward(self, logits, label_marks):
        if self.type_loss == 'log_sum':
            return log_sum_loss(logits, label_marks, self.reduction)
        elif self.type_loss == 'sum_log':
            return sum_log_loss(logits, label_marks, self.reduction)
        elif self.type_loss == 'sum_log_nce':
            return sum_log_nce_loss(logits, label_marks, self.reduction)
        elif self.type_loss == 'max_min':
            return max_min_loss(logits, label_marks, self.reduction)
        else:
            raise ValueError('wrong type of multi-label loss')


def log_sum_loss(logits, mask, reduction='sum'):
    """
    :param logits: reranking logits(E2M x TC) or span loss(E2M x TC x L)
    :param mask: reranking mask(E2M x TC) or span mask(E2M x TC x L)
    :return: log sum p_positive
    """
    #  log marginal likelihood
    gold_logits = logits.masked_fill(~(mask.bool()), -10000)
    gold_log_sum_exp = torch.logsumexp(gold_logits, -1)
    all_log_sum_exp = torch.logsumexp(logits, -1)
    gold_log_probs = gold_log_sum_exp - all_log_sum_exp
    loss = -gold_log_probs.sum()
    if reduction == 'mean':
        loss /= logits.size(0)
    return loss


def sum_log_nce_loss(logits, mask, reduction='sum'):
    """
        :param logits: reranking logits(E2M x TC) or span loss(E2M x TC x L)
        :param mask: reranking mask(E2M x TC) or span mask(E2M x TC x L)
        :return: sum log p_positive i  over (positive i, negatives)
    """
    gold_scores = logits.masked_fill(~(mask.bool()), 0)
    gold_scores_sum = gold_scores.sum(-1)  # E2M x TC
    neg_logits = logits.masked_fill(mask.bool(), float('-inf'))  # E2M x TC x L
    neg_log_sum_exp = torch.logsumexp(neg_logits, -1, keepdim=True)  # E2M x TC x 1
    norm_term = torch.logaddexp(logits, neg_log_sum_exp).masked_fill(~(
        mask.bool()), 0).sum(-1)
    gold_log_probs = gold_scores_sum - norm_term
    loss = -gold_log_probs.sum()
    if reduction == 'mean':
        print('mean reduction')
        loss /= logits.size(0)
    return loss


def sum_log_loss(logits, mask, reduction='sum'):
    """
            :param logits: reranking logits(E2M x TC) or span loss(E2M x TC x L)
            :param mask: reranking mask(E2M x TC) or span mask(E2M x TC x L)
            :return: sum log p_positive i  over all candidates
    """
    num_pos = mask.sum(-1)  # E2M x TC
    gold_scores = logits.masked_fill(~(mask.bool()), 0)
    gold_scores_sum = gold_scores.sum(-1)  # BxC
    all_log_sum_exp = torch.logsumexp(logits, -1)  # E2M x TC
    # gold_log_probs = gold_scores_sum - all_log_sum_exp * num_pos
    gold_log_probs = gold_scores_sum/num_pos - all_log_sum_exp
    loss = -gold_log_probs.sum()
    if reduction == 'mean':
        loss /= logits.size(0)
    return loss


def max_min_loss(logits, mask, reduction='sum'):
    """
            :param logits: reranking logits(E2M x TC) or span loss(E2M x TC x L)
            :param mask: reranking mask(E2M x TC) or span mask(E2M x TC x L)
            :return: min log p_positive i  over all positives
    """
    gold_scores = logits.masked_fill(~(mask.bool()), 10000)
    min_gold_scores = gold_scores.min(-1)[0]
    all_log_sum_exp = torch.logsumexp(logits, -1)
    min_gold_probs = min_gold_scores - all_log_sum_exp
    loss = -min_gold_probs.sum()
    if reduction == 'mean':
        loss /= logits.size(0)
    return loss
