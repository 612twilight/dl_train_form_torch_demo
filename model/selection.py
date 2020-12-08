# -*- coding:utf-8 -*-
"""
-------------------------------------------------
Project Name: dl_train_form_torch_demo
File Name: selection.py
Author: gaoyw
Create Date: 2020/12/8
-------------------------------------------------
"""
import copy
from functools import partial
from typing import Dict

import torch
import torch.nn as nn
from torchcrf import CRF

from config.hyper import hyper
from model.component.components import masked_BCEloss, inference, description
from utils.utils import read_vocab


class MultiHeadSelection(nn.Module):
    def __init__(self) -> None:
        super(MultiHeadSelection, self).__init__()
        self.data_root = hyper.data_root
        self.gpu = hyper.gpu

        self.word2index = read_vocab(hyper.vocab_path)
        self.relation2index = read_vocab(hyper.label_path)
        self.bio2index = read_vocab(hyper.bio_label_path)

        self.word_embeddings = nn.Embedding(num_embeddings=len(
            self.word_vocab),
            embedding_dim=hyper.emb_size)

        self.relation_emb = nn.Embedding(num_embeddings=len(
            self.relation_vocab),
            embedding_dim=hyper.rel_emb_size)
        # bio + pad
        self.bio_emb = nn.Embedding(num_embeddings=len(self.bio_vocab),
                                    embedding_dim=hyper.bio_emb_size)

        self.encoder = nn.LSTM(hyper.emb_size,
                               hyper.hidden_size,
                               bidirectional=True,
                               batch_first=True)

        if hyper.activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif hyper.activation.lower() == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError('unexpected activation!')

        self.tagger = CRF(len(self.bio_vocab) - 1, batch_first=True)

        self.selection_u = nn.Linear(hyper.hidden_size + hyper.bio_emb_size,
                                     hyper.rel_emb_size)
        self.selection_v = nn.Linear(hyper.hidden_size + hyper.bio_emb_size,
                                     hyper.rel_emb_size)
        self.selection_uv = nn.Linear(2 * hyper.rel_emb_size,
                                      hyper.rel_emb_size)
        self.emission = nn.Linear(hyper.hidden_size, len(self.bio_vocab) - 1)

    def forward(self, sample, is_train: bool) -> Dict[str, torch.Tensor]:

        tokens = sample.tokens_id.cuda(self.gpu)
        selection_gold = sample.selection_id.cuda(self.gpu)
        bio_gold = sample.bio_id.cuda(self.gpu)

        text_list = sample.text
        spo_gold = sample.spo_gold

        bio_text = sample.bio

        mask = tokens != self.word_vocab['<pad>']  # batch x seq
        bio_mask = mask

        embedded = self.word_embeddings(tokens)
        o, h = self.encoder(embedded)

        o = (lambda a: sum(a) / 2)(torch.split(o, self.hyper.hidden_size, dim=2))
        emi = self.emission(o)

        output = {}

        crf_loss = 0

        if is_train:
            crf_loss = -self.tagger(emi, bio_gold,
                                    mask=bio_mask, reduction='mean')
        else:
            decoded_tag = self.tagger.decode(emissions=emi, mask=bio_mask)

            output['decoded_tag'] = [list(map(lambda x: self.id2bio[x], tags)) for tags in decoded_tag]
            output['gold_tags'] = bio_text

            temp_tag = copy.deepcopy(decoded_tag)
            for line in temp_tag:
                line.extend([self.bio_vocab['<pad>']] *
                            (self.hyper.max_text_len - len(line)))
            bio_gold = torch.tensor(temp_tag).cuda(self.gpu)

        tag_emb = self.bio_emb(bio_gold)

        o = torch.cat((o, tag_emb), dim=2)

        # forward multi head selection
        B, L, H = o.size()
        u = self.activation(self.selection_u(o)).unsqueeze(1).expand(B, L, L, -1)
        v = self.activation(self.selection_v(o)).unsqueeze(2).expand(B, L, L, -1)
        uv = self.activation(self.selection_uv(torch.cat((u, v), dim=-1)))

        # correct one
        selection_logits = torch.einsum('bijh,rh->birj', uv,
                                        self.relation_emb.weight)

        # use loop instead of matrix
        # selection_logits_list = []
        # for i in range(self.hyper.max_text_len):
        #     uvi = uv[:, i, :, :]
        #     sigmoid_input = uvi
        #     selection_logits_i = torch.einsum('bjh,rh->brj', sigmoid_input,
        #                                         self.relation_emb.weight).unsqueeze(1)
        #     selection_logits_list.append(selection_logits_i)
        # selection_logits = torch.cat(selection_logits_list,dim=1)

        if not is_train:
            output['selection_triplets'] = inference(mask, text_list,
                                                     decoded_tag, selection_logits,
                                                     self.relation2index, self.bio2index)
            output['spo_gold'] = spo_gold

        selection_loss = 0
        if is_train:
            selection_loss = masked_BCEloss(mask, selection_logits,
                                            selection_gold, len(self.relation_vocab))

        loss = crf_loss + selection_loss
        output['crf_loss'] = crf_loss
        output['selection_loss'] = selection_loss
        output['loss'] = loss

        output['description'] = partial(description, output=output)
        return output
