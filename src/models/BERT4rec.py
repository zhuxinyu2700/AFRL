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

class BERTEmbeddings(nn.Module):

    def __init__(self, vocab_size: int, embed_size: int, maxlen: int, dropout_rate: float = 0.1):
        """
        :param vocab_size: total_vocab_size
        :param embed_size: embedding size of token embedding
        :param maxlen : maxlen of sequence
        :param dropout_rate: dropout rate
        """
        super(BERTEmbeddings, self).__init__()
        self.iid_embeddings = nn.Embedding(vocab_size + 1, embed_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(maxlen, embed_size)
        self.segment_embeddings = nn.Embedding(3, embed_size, padding_idx=0)
        # layer_norm + dropout
        self.layer_norm = nn.LayerNorm(embed_size, eps=1e-6)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, seq: torch.Tensor, segment_label: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_length = seq.size(0), seq.size(1)  # seq : (batch, seq_len)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=seq.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)  # position_ids : (batch_size, seq_length)
        # token , position embeddings
        iid_embeddings = self.iid_embeddings(seq)
        position_embeddings = self.position_embeddings(position_ids)
        # bert_embedddings
        embeddings = iid_embeddings + position_embeddings
        # segment embeddings
        if segment_label is not None:
            segment_embeddings = self.segment_embeddings(segment_label)
            embeddings += segment_embeddings
        # layer-norm + drop out
        embeddings = self.dropout(self.layer_norm(embeddings))

        return embeddings

class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention
    """
    def __init__(self, num_heads, hidden_dim, dropout_rate_attn=0.1):
        """
        :param head_num: attention head num
        :param hidden_dim : hidden dim
        :param dropout_rate: dropout rate
        """
        super(MultiHeadedAttention, self).__init__()

        assert hidden_dim % num_heads == 0, "Wrong hidden_dim, head_num"

        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        self.head_num = num_heads
        # Q,K,V linear layer
        self.query_linear = nn.Linear(hidden_dim, hidden_dim)
        self.key_linear = nn.Linear(hidden_dim, hidden_dim)
        self.value_linear = nn.Linear(hidden_dim, hidden_dim)
        self.scale = math.sqrt(self.head_dim)
        # dropout
        self.dropout = nn.Dropout(p=dropout_rate_attn)
        # Output linear layer
        self.output_linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = q.size(0)
        # Q, K, V
        query = self.query_linear(q)  # query,key, value : [batch_size, seq_length, hidden_dim]
        key = self.key_linear(k)
        value = self.value_linear(v)
        # [batch, len, head 수, head 차원] -> [batch, head_num, len, head_dim]
        query = query.view(batch_size, -1, self.head_num, self.head_dim).permute(0, 2, 1, 3)
        key = key.view(batch_size, -1, self.head_num, self.head_dim).permute(0, 2, 1, 3)
        value = value.view(batch_size, -1, self.head_num, self.head_dim).permute(0, 2, 1, 3)
        # scores : [batch, head_num, query_len, key_len]
        scores = torch.matmul(query, key.permute(0, 1, 3, 2)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        # attention : [batch, head_num, query_len, key_len]
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        # attention_seq : [batch, head_num, query_len, head_dim]
        attention_seq = torch.matmul(attention, value).contiguous()
        # attention_seq : [batch, query_len, hidden_dim]
        attention_seq = attention_seq.view(batch_size, -1, self.hidden_dim)
        attention_seq = self.output_linear(attention_seq)

        return attention_seq, attention


class SublayerConnection(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.1):
        super(SublayerConnection, self).__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, layer: torch.Tensor, sublayer: torch.Tensor) -> torch.Tensor:
        "Apply residual connection to any sublayer with the same size."
        return layer + self.dropout(self.layer_norm(sublayer))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_dim: int, ff_dim: int, dropout_rate: float = 0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.feed_forward_1 = nn.Linear(hidden_dim, ff_dim)
        self.feed_forward_2 = nn.Linear(ff_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feed_forward_2(
            self.dropout(self.activation(self.feed_forward_1(x)))
        )


class TransformerEncoder(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(
        self, hidden_dim: int, num_heads: int, ff_dim: int, dropout_rate: float = 0.1, dropout_rate_attn: float = 0.1
    ):
        """
        :param hidden_dim: hidden dim of transformer
        :param head_num: head sizes of multi-head attention
        :param ff_dim: feed_forward_hidden, usually 4*hidden_dim
        :param dropout_rate: dropout rate
        :param dropout_rate_attn : attention layer의 dropout rate
        """
        super(TransformerEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.dropout_rate_attn = dropout_rate_attn
        # multi-head attn
        self.attention = MultiHeadedAttention(
            num_heads=self.num_heads,
            hidden_dim=self.hidden_dim,
            dropout_rate_attn=self.dropout_rate_attn,
        )
        # sublayer connection - 1 (input embeddings + input embeddings attn)
        self.input_sublayer = SublayerConnection(
            hidden_dim=self.hidden_dim, dropout_rate=self.dropout_rate
        )
        # FFN
        self.feed_forward = PositionwiseFeedForward(
            hidden_dim=self.hidden_dim,
            ff_dim=self.ff_dim,
            dropout_rate=self.dropout_rate,
        )
        self.output_sublayer = SublayerConnection(
            hidden_dim=self.hidden_dim, dropout_rate=self.dropout_rate
        )
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        attention_seq, _ = self.attention(q=seq, k=seq, v=seq, mask=mask)
        connected_layer = self.input_sublayer(seq, attention_seq)
        connected_layer = self.output_sublayer(connected_layer, self.feed_forward(connected_layer))

        return self.dropout(connected_layer)

class BERT4rec(BaseRecModel):
    @staticmethod
    def parse_model_args(parser, model_name='SASRec'):
        parser.add_argument('--num_heads', default=1, type=int)
        parser.add_argument('--num_blocks', default=1, type=int)
        parser.add_argument('--maxlen', type=int, default=100, help='maxlen')
        return BaseRecModel.parse_model_args(parser, model_name)

    def __init__(self, data_processor_dict, user_num, item_num, num_heads, num_blocks, u_vector_size, i_vector_size,
                 maxlen, random_seed=2020, dropout=0.2, model_path='../model/Model/Model.pt'):
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.hidden_dim = i_vector_size
        self.ff_dim = self.hidden_dim
        self.maxlen = maxlen
        BaseRecModel.__init__(self, data_processor_dict, user_num, item_num, u_vector_size, i_vector_size,
                              random_seed=random_seed, dropout=dropout, model_path=model_path)

    def _init_nn(self):
        # embedding
        self.embedding = BERTEmbeddings(vocab_size=self.item_num, embed_size=self.hidden_dim, maxlen=self.maxlen)
        self.transformer_encoders = nn.ModuleList(
            [
                TransformerEncoder(
                    hidden_dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    ff_dim=self.ff_dim,
                    dropout_rate=self.dropout,
                    dropout_rate_attn=self.dropout,
                )
                for _ in range(self.num_blocks)
            ]
        )

    def log2feats(self, log_seqs):
        mask = (log_seqs > 0).unsqueeze(1).unsqueeze(1)
        log_feats = self.embedding(log_seqs)
        for transformer in self.transformer_encoders:
            log_feats = transformer(log_feats, mask)

        return log_feats

    def predict(self,feed_dict):
        check_list = []
        seq = feed_dict['seq']
        pos_ids = feed_dict['pos']
        neg_ids = feed_dict['neg']

        log_feats = self.log2feats(seq)
        u_vectors = log_feats[:, -1]
        pos_vectors = self.embedding.iid_embeddings(pos_ids)
        neg_vectors = self.embedding.iid_embeddings(neg_ids)

        pos_prediction = (u_vectors.unsqueeze(1) * pos_vectors).sum(dim=-1)
        neg_prediction = (u_vectors.unsqueeze(1) * neg_vectors).sum(dim=-1)
        prediction = torch.cat((pos_prediction, neg_prediction), -1)

        out_dict = {'pos_prediction': pos_prediction,
                    'neg_prediction': neg_prediction,
                    'prediction': prediction,
                    'check': check_list,
                    'u_vectors': u_vectors}
        return out_dict

    def predict_vectors(self,vectors, feed_dict):
        check_list = []
        pos_ids = feed_dict['pos']
        neg_ids = feed_dict['neg']


        u_vectors = vectors
        pos_vectors = self.embedding.iid_embeddings(pos_ids)
        neg_vectors = self.embedding.iid_embeddings(neg_ids)

        pos_prediction = (u_vectors.unsqueeze(1) * pos_vectors).sum(dim=-1)
        neg_prediction = (u_vectors.unsqueeze(1) * neg_vectors).sum(dim=-1)
        prediction = torch.cat((pos_prediction, neg_prediction), -1)

        out_dict = {'pos_prediction': pos_prediction,
                    'neg_prediction': neg_prediction,
                    'prediction': prediction,
                    'check': check_list,
                    'u_vectors': u_vectors}
        return out_dict

