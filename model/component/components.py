# -*- coding:utf-8 -*-
"""
-------------------------------------------------
Project Name: dl_train_form_torch_demo
File Name: components.py
Author: gaoyw
Create Date: 2020/12/8
-------------------------------------------------
"""
import torch
import torch.nn.functional as F

from config.hyper import hyper
from utils.utils import find_entity


def selection_decode(text_list, sequence_tags, selection_tags, relation2index, bio2index):
    import torch
    reversed_relation_vocab = {v: k for k, v in relation2index.items()}

    reversed_bio_vocab = {v: k for k, v in bio2index.items()}

    text_list = list(map(list, text_list))

    batch_num = len(sequence_tags)
    result = [[] for _ in range(batch_num)]
    idx = torch.nonzero(selection_tags.cpu())
    for i in range(idx.size(0)):
        b, s, p, o = idx[i].tolist()
        predicate = reversed_relation_vocab[p]
        if predicate == 'N':
            continue
        tags = list(map(lambda x: reversed_bio_vocab[x], sequence_tags[b]))
        object = find_entity(o, text_list[b], tags)
        subject = find_entity(s, text_list[b], tags)
        assert object != '' and subject != ''
        triplet = {
            'object': object,
            'predicate': predicate,
            'subject': subject
        }
        result[b].append(triplet)
    return result


def masked_BCEloss(mask, selection_logits, selection_gold, relation_len):
    selection_mask = (mask.unsqueeze(2) * mask.unsqueeze(1)) \
        .unsqueeze(2) \
        .expand(-1, -1, relation_len, -1)  # batch x seq x rel x seq
    selection_loss = F.binary_cross_entropy_with_logits(selection_logits,
                                                        selection_gold,
                                                        reduction='none')
    selection_loss = selection_loss.masked_select(selection_mask).sum()
    selection_loss /= mask.sum()
    return selection_loss


def inference(mask, text_list, decoded_tag, selection_logits, relation2index, bio2index):
    selection_mask = (mask.unsqueeze(2) * mask.unsqueeze(1)).unsqueeze(2).expand(
        -1, -1, len(relation2index), -1)  # batch x seq x rel x seq
    selection_tags = (torch.sigmoid(selection_logits) *
                      selection_mask.float()) > hyper.threshold

    selection_triplets = selection_decode(text_list, decoded_tag, selection_tags,
                                          relation2index, bio2index)
    return selection_triplets


def description(epoch, epoch_num, output):
    return "L: {:.2f}, L_crf: {:.2f}, L_selection: {:.2f}, epoch: {}/{}:".format(
        output['loss'].item(), output['crf_loss'].item(),
        output['selection_loss'].item(), epoch, epoch_num)
