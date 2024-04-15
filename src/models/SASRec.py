# coding=utf-8
import torch
import torch.nn as nn
import logging
import torch.nn.functional as F
import os
import numpy as np
from utils.constants import *
import math
from typing import Optional, Tuple
from models.BaseRecModel import BaseRecModel

class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_dim, ff_dim, dropout_rate):
        super(PositionwiseFeedForward, self).__init__()
        self.conv1 = torch.nn.Conv1d(hidden_dim, ff_dim, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(ff_dim, hidden_dim, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

class SASRec(BaseRecModel):
    @staticmethod
    def parse_model_args(parser, model_name='SASRec'):
        parser.add_argument('--num_heads', default=1, type=int)
        parser.add_argument('--num_blocks', default=1, type=int)
        parser.add_argument('--maxlen', type=int, default=100, help='maxlen')
        return BaseRecModel.parse_model_args(parser, model_name)

    def __init__(self, data_processor_dict, user_num, item_num, num_heads, num_blocks, u_vector_size, i_vector_size, maxlen,
                 random_seed=2020, dropout=0.2, model_path='../model/Model/Model.pt'):
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.hidden_dim = i_vector_size
        self.ff_dim = self.hidden_dim
        self.maxlen = maxlen
        BaseRecModel.__init__(self, data_processor_dict, user_num, item_num, u_vector_size, i_vector_size,
                              random_seed=random_seed, dropout=dropout, model_path=model_path)

    def _init_nn(self):
        self.iid_embeddings = torch.nn.Embedding(self.item_num + 1, self.i_vector_size, padding_idx=0)
        self.pos_embedding = torch.nn.Embedding(self.maxlen, self.hidden_dim)
        self.emb_dropout = torch.nn.Dropout(p=self.dropout)

        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(self.hidden_dim, eps=1e-8)

        for _ in range(self.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(self.hidden_dim, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(self.hidden_dim,
                                                         self.num_heads,
                                                         self.dropout)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(self.hidden_dim, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PositionwiseFeedForward(self.hidden_dim, self.ff_dim, self.dropout)
            self.forward_layers.append(new_fwd_layer)

    def log2feats(self, log_seqs):
        seqs = self.iid_embeddings(log_seqs)
        seqs *= self.iid_embeddings.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_embedding(torch.LongTensor(positions).to(device=log_seqs.device))
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.cuda.BoolTensor(log_seqs == 0)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=log_seqs.device))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1).to(device=log_seqs.device)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs,
                                            attn_mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    def predict(self, feed_dict):
        check_list = []
        seq = feed_dict['seq']
        pos_ids = feed_dict['pos']
        neg_ids = feed_dict['neg']

        log_feats = self.log2feats(seq)
        u_vectors = log_feats[:, -1]

        pos_vectors = self.iid_embeddings(pos_ids)
        neg_vectors = self.iid_embeddings(neg_ids)

        pos_prediction = (u_vectors.unsqueeze(1) * pos_vectors).sum(dim=-1)
        neg_prediction = (u_vectors.unsqueeze(1) * neg_vectors).sum(dim=-1)
        prediction = torch.cat((pos_prediction, neg_prediction), -1)

        out_dict = {'pos_prediction': pos_prediction,
                    'neg_prediction': neg_prediction,
                    'prediction' : prediction,
                    'check': check_list,
                    'u_vectors': u_vectors}
        return out_dict

    def predict_vectors(self, vectors, feed_dict):
        check_list = []
        pos_ids = feed_dict['pos']
        neg_ids = feed_dict['neg']

        u_vectors = vectors

        pos_vectors = self.iid_embeddings(pos_ids)
        neg_vectors = self.iid_embeddings(neg_ids)

        pos_prediction = (u_vectors.unsqueeze(1) * pos_vectors).sum(dim=-1)
        neg_prediction = (u_vectors.unsqueeze(1) * neg_vectors).sum(dim=-1)
        prediction = torch.cat((pos_prediction, neg_prediction), -1)

        out_dict = {'pos_prediction': pos_prediction,
                    'neg_prediction': neg_prediction,
                    'prediction' : prediction,
                    'check': check_list,
                    'u_vectors': u_vectors}
        return out_dict
