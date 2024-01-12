# -*- coding: utf-8 -*-
# @FileName: e2m_module.py
# @Software: PyCharm


import numpy as np
import torch
import torch.nn as nn
from loss_function import MultiLabelLoss

# Extractor model

class BiEncoder(nn.Module):
    def __init__(self, mention_encoder,
                 entity_encoder,
                 type_loss):
        super(BiEncoder, self).__init__()
        self.mention_encoder = mention_encoder
        self.entity_encoder = entity_encoder
        self.loss_fct = MultiLabelLoss(type_loss)

    def encode(self, mention_token_ids=None,
               mention_masks=None,
               candidate_token_ids=None,
               candidate_masks=None,
               entity_token_ids=None,
               entity_masks=None):
        candidates_embeds = None
        mention_embeds = None
        entity_embeds = None
        # candidate_token_ids and mention_token_ids not None during training
        # mention_token_ids not None for embedding mentions during inference
        # entity_token_ids not None for embedding entities during inference
        if candidate_token_ids is not None:
            B, C, L = candidate_token_ids.size()
            candidate_token_ids = candidate_token_ids.view(-1, L)
            candidate_masks = candidate_masks.view(-1, L)
            # E2M X TC X L --> BC X L
            candidates_embeds = self.entity_encoder(
                input_ids=candidate_token_ids,
                attention_mask=candidate_masks
            )[0][:, 0, :].view(B, C, -1)
        if mention_token_ids is not None:
            mention_embeds = self.mention_encoder(
                input_ids=mention_token_ids,
                attention_mask=mention_masks
            )[0][:, 0, :]
        if entity_token_ids is not None:
            # for getting all the entity embeddings
            entity_embeds = self.entity_encoder(input_ids=entity_token_ids,
                                                attention_mask=entity_masks)[
                                0][:, 0, :]
        return mention_embeds, candidates_embeds, entity_embeds

    def forward(self,
                mention_token_ids=None,
                mention_masks=None,
                candidate_token_ids=None,
                candidate_masks=None,
                passages_labels=None,
                entity_token_ids=None,
                entity_masks=None
                ):
        """

        :param inputs: [
                        mention_token_ids,mention_masks,  size: E2M X L
                        candidate_token_ids,candidate_masks, size: E2M X TC X L
                        passages_labels, size: E2M X TC
                        ]
        :return: loss, logits

        """
        if not self.training:
            return self.encode(mention_token_ids, mention_masks,
                               candidate_token_ids, candidate_masks,
                               entity_token_ids, entity_masks)
        B, C, L = candidate_token_ids.size()
        mention_embeds, candidates_embeds, _ = self.encode(
            mention_token_ids,
            mention_masks,
            candidate_token_ids,
            candidate_masks)
        mention_embeds = mention_embeds.unsqueeze(1)
        logits = torch.matmul(mention_embeds,
                              candidates_embeds.transpose(1, 2)).view(B, -1)
        loss = self.loss_fct(logits, passages_labels)

        return loss, logits


# Matcher model

class Matcher(nn.Module):

    def __init__(self, encoder,
                 type_span_loss,
                 do_rerank,
                 type_rank_loss,
                 max_answer_len,
                 max_passage_len):
        super(Matcher, self).__init__()
        self.encoder = encoder
        self.span_loss_fct = MultiLabelLoss(type_span_loss)
        self.rank_loss_fct = MultiLabelLoss(type_rank_loss)
        self.do_rerank = do_rerank
        # maximum answer(mention span) length
        self.max_answer_len = max_answer_len
        self.max_passage_len = max_passage_len
        self.dim_hidden = self.encoder.config.hidden_size
        self.qa_outputs = nn.Linear(self.dim_hidden, 2)
        self.qa_classifier = nn.Linear(self.dim_hidden, 1)
        self.init_weights()

    def init_weights(self):
        self.qa_outputs.weight.data.normal_(mean=0.0,
                                            std=self.encoder.config.initializer_range)
        self.qa_classifier.weight.data.normal_(mean=0.0,
                                               std=self.encoder.config.initializer_range)
        self.qa_outputs.bias.data.zero_()
        self.qa_classifier.bias.data.zero_()

    def get_batch_probs(self,
                        start_logits,
                        end_logits,
                        rank_logits=None):

        # E2M x TC x max_passage_len
        start_probs = start_logits.log_softmax(-1)
        # E2M x TC x max_passage_len
        end_probs = end_logits.log_softmax(-1)
        # E2M x TC x L x 1 + E2M x TC x 1 x L + E2M x TC x 1 x 1 --> E2M x TC x L x L
        mention_probs = start_probs.unsqueeze(-1) + end_probs.unsqueeze(
            -2)
        if self.do_rerank:
            # E2M x TC
            rank_probs = rank_logits.log_softmax(-1)
            mention_probs = mention_probs + rank_probs.unsqueeze(-1).unsqueeze(
                -1)
        # E2M x TC x max_passage_len x max_passage_len
        mention_probs = mention_probs.exp().triu(0).tril(self.max_answer_len
                                                         - 1)[:, :,
                        :self.max_passage_len, :self.max_passage_len]
        if self.do_rerank:
            return mention_probs, rank_logits
        return mention_probs

    def forward(self, input_ids,
                attention_mask,
                token_type_ids,
                answer_mask,
                passage_labels=None,
                start_labels=None,
                end_labels=None):
        # batchsize, number of candidates per question, length
        B, C, L = input_ids.size()
        input_ids = input_ids.view(-1, L)
        attention_mask = attention_mask.view(-1, L)
        token_type_ids = token_type_ids.view(-1, L)
        # BC x L x d
        last_hiddens = self.encoder(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids)[0]
        span_logits = self.qa_outputs(last_hiddens)
        start_logits, end_logits = span_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).view(B, C, L)
        end_logits = end_logits.squeeze(-1).view(B, C, L)
        rank_logits = None
        if self.do_rerank:
            rank_logits = self.qa_classifier(last_hiddens[:, 0, :]).view(B, C)
        start_logits = start_logits.masked_fill(~(answer_mask.bool()),
                                                -10000)
        end_logits = end_logits.masked_fill(~(answer_mask.bool()), -10000)
        if self.training:
            start_loss = self.span_loss_fct(start_logits, start_labels)
            end_loss = self.span_loss_fct(end_logits, end_labels)
            if self.do_rerank:
                rank_loss = self.rank_loss_fct(rank_logits, passage_labels)
                loss = (start_loss + end_loss + rank_loss) / 3
            else:
                loss = start_loss + end_loss
            return loss
        return self.get_batch_probs(start_logits, end_logits, rank_logits)


def get_top_spans(_mention_probs,
                  k=10, filter_spans=True):
    """
    :param _mention_probs: max_passage_len x max_passage_len
    :param k: top k spans
    :param filter_spans:  prevent nested mention spans
    :return:
    """
    spans = _mention_probs.nonzero(as_tuple=False)
    scores = _mention_probs[_mention_probs.nonzero(as_tuple=True)]
    spans_scores = torch.cat((spans, scores.unsqueeze(-1)), -1)
    sorted_spans_scores = spans_scores[spans_scores[:, -1].argsort(0, True)]
    selected_spans = []
    for start, end, s in sorted_spans_scores:
        start = start.long()
        end = end.long()
        if start.item() == 0 and end.item() == 0:
            break
        if filter_spans and any(start.item() <= selected_start <=
                                selected_end <= end.item()
                                or selected_start <= start.item() <= end.item() <= selected_end
                                for selected_start, selected_end, _ in
                                selected_spans):
            continue
        selected_spans.append([start.item(), end.item(), s.item()])
        if len(selected_spans) == k:
            break
    selected_spans = torch.tensor(selected_spans)
    return selected_spans


def get_predicts(mention_probs,
                 k=10,
                 filter_span=True,
                 no_multi_ents=False):
    """
    :param mention_probs:  N x TC x max_passage_len x max_passage_len
    :param k: top k spans for each candidate
    :param filter_span: prevent nested mention spans?
    :param no_multi_ents:  prevent multiple entities for a single mention span?
    :return: batch predictions before thresholding
    """
    B, C, max_passage_len, _ = mention_probs.size()
    print(mention_probs.size())
    results = []
    for i in range(B):
        candidate_predicts = []
        for j in range(C):
            # k x 3 : start, end, score
            spans = get_top_spans(mention_probs[i, j], k, filter_span)
            num_spans = spans.size(0)
            if num_spans != 0:
                # k x 1
                candidate_idx = torch.tensor([[j]] * num_spans)
                # kx4: entity,start,end,score
                result = torch.cat((candidate_idx, spans), 1)
                assert result.size() == (num_spans, 4)
                candidate_predicts.append(result)
        if len(candidate_predicts) > 0:
            candidate_predicts = torch.cat(candidate_predicts, 0)  # Ck x 4
            #  prevent multiple entities for the same mention span
            if no_multi_ents:
                candidate_predicts = candidate_predicts[
                    candidate_predicts[:, -1].argsort(0, True)].numpy()
                unique_ids = np.unique(candidate_predicts[:, 1:3],
                                       axis=0,
                                       return_index=True)[1]
                candidate_predicts = torch.tensor(candidate_predicts[
                                                      unique_ids])
            # entity, start, end,score
            results.append(candidate_predicts)
        else:
            results.append([])

    return results


def prune_predicts(predicts, threshold):
    assert len(predicts) > 0
    results = []
    for cand_predicts in predicts:
        if len(cand_predicts) == 0:
            results.append([])
        else:
            cand_probs = cand_predicts[:, -1]
            selection = (cand_probs > threshold)
            cand_results = cand_predicts[:, :-1].long()[selection].tolist()
            results.append(cand_results)
    return results
