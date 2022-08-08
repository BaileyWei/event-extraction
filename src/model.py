import torch
import torch.nn as nn
from copy import deepcopy
from allennlp.models import Model
from allennlp.nn.util import sequence_cross_entropy_with_logits

from metrics import ArgumentMetric
from cfg import args


class ArgumentExtractor(Model):
    def __init__(self, vocab, embedder, role_num, event_roles, prob_threshold, use_loss_weight=False):
        super().__init__(vocab)
        emb_dim = embedder.get_output_dim()
        self.role_num = role_num
        self.event_roles = event_roles

        self.embedder = embedder
        self.start_tagger = nn.ModuleList([nn.Linear(emb_dim, 2) for i in range(role_num)])
        self.end_tagger = nn.ModuleList([nn.Linear(emb_dim, 2) for i in range(role_num)])
        self.metric = ArgumentMetric(event_roles, prob_threshold)

        self.use_loss_weight = use_loss_weight

    def forward(self, sentence, text_id, event_type=0, trigger=None, roles=None, ori_text=None):
        tokens = sentence['tokens']
        offset = sentence['tokens-offsets']
        mask = sentence['mask']


        embedding = self.embedder(input_ids=tokens, offsets=offset)
        B, L, E = embedding.size()

        if roles is not None:  # 训练过程
            start_role_label = torch.zeros(self.role_num, B, L).long()
            end_role_label = torch.zeros(self.role_num, B, L).long()
            batch_role_num = roles.size(1)
            if embedding.is_cuda:
                start_role_label = start_role_label.cuda(args.extractor_cuda_device)
                end_role_label = end_role_label.cuda(args.extractor_cuda_device)
            for idxb in range(B):
                batch_span = roles[idxb]
                for idxs in range(batch_role_num):
                    span = batch_span[idxs]
                    span_start, span_end, span_label = span[0], span[1], span[2]
                    if span_start == -1:
                        break
                    assert span_label >= 0 and span_label < self.role_num
                    if span_start < L and span_end < L:  # DuEE-Fin中存在超标的
                        start_role_label[span_label, idxb, span_start] = 1
                        end_role_label[span_label, idxb, span_end] = 1

            start_logits = []
            end_logits = []
            sum_loss = 0.
            for idc in range(self.role_num):
                s_tagger = self.start_tagger[idc]
                e_tagger = self.end_tagger[idc]

                new_mask = deepcopy(mask)
                for idb in range(B):

                    if idc not in self.event_roles[0]:
                        new_mask[idb, :] = 0

                batch_weight = torch.Tensor([1. / B for i in range(B)])
                if embedding.is_cuda:
                    batch_weight = batch_weight.cuda(args.extractor_cuda_device)

                # ==========AF/IEF==============
                if self.training:
                    # tagger_ief = self.ief[idc]
                    # batch_af = self.af[idc, event_type]
                    # if embedding.is_cuda:
                    #     tagger_ief = tagger_ief.cuda(args.extractor_cuda_device)
                    #     batch_af = batch_af.cuda(args.extractor_cuda_device)
                    # af_ief = tagger_ief * batch_af
                    # batch_weight = torch.exp(af_ief)
                    # batch_weight_sum = torch.sum(batch_weight)
                    # batch_weight = batch_weight / batch_weight_sum
                    #
                    # batch_af = batch_af.view(-1, 1)
                    # if self.use_loss_weight:
                    #     new_mask_start = new_mask.float() + batch_af * start_role_label[idc].float()
                    #     new_mask_end = new_mask.float() + batch_af * end_role_label[idc].float()
                    # else:
                    new_mask_start = new_mask.float()
                    new_mask_end = new_mask.float()
                s_logit = s_tagger(embedding)
                e_logit = e_tagger(embedding)
                s_label = start_role_label[idc]
                e_label = end_role_label[idc]
                if self.training:
                    s_loss = self.per_batch_loss(s_logit, s_label, new_mask_start, batch_weight)
                    e_loss = self.per_batch_loss(e_logit, e_label, new_mask_end, batch_weight)
                    # s_loss = sequence_cross_entropy_with_logits(s_logit, s_label, new_mask)
                    # e_loss = sequence_cross_entropy_with_logits(e_logit, e_label, new_mask)
                else:
                    # if torch.sum(mask) != 0:
                    s_loss = sequence_cross_entropy_with_logits(s_logit, s_label, new_mask, average='token')
                    e_loss = sequence_cross_entropy_with_logits(e_logit, e_label, new_mask, average='token')
                sum_loss = sum_loss + (s_loss + e_loss) / 2.
                start_logits.append(s_logit)
                end_logits.append(e_logit)
            if not self.training:
                self.metric(start_logits, end_logits, event_type, roles)
            mean_loss = sum_loss / self.role_num
            output = {'start_logits': start_logits, 'end_logits': end_logits, 'loss': mean_loss}
            return output
        else:  # 预测过程
            start_logits = []
            end_logits = []

            for idc in range(self.role_num):
                s_tagger = self.start_tagger[idc]
                e_tagger = self.end_tagger[idc]

                s_logit = s_tagger(embedding)
                e_logit = e_tagger(embedding)

                start_logits.append(s_logit)
                end_logits.append(e_logit)

            output = {'start_logits': start_logits, 'end_logits': end_logits}
            return output

    def get_metrics(self, reset=False):
        if not self.training:
            evaluation = self.metric.get_metric(reset)
            return evaluation
        else:
            return {}

    def per_batch_loss(self,
                       logits: torch.FloatTensor,
                       targets: torch.LongTensor,
                       weights: torch.FloatTensor,
                       batch_weights):
        # shape : (batch * sequence_length, num_classes)
        logits_flat = logits.view(-1, logits.size(-1))
        # shape : (batch * sequence_length, num_classes)
        log_probs_flat = torch.nn.functional.log_softmax(logits_flat, dim=-1)
        # shape : (batch * max_len, 1)
        targets_flat = targets.view(-1, 1).long()

        negative_log_likelihood_flat = - torch.gather(log_probs_flat, dim=1, index=targets_flat)
        # shape : (batch, sequence_length)
        negative_log_likelihood = negative_log_likelihood_flat.view(*targets.size())
        # shape : (batch, sequence_length)
        negative_log_likelihood = negative_log_likelihood * weights.float()

        # shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(1) / (weights.sum(1).float() + 1e-13)
        per_batch_loss = per_batch_loss * batch_weights
        return per_batch_loss.sum()