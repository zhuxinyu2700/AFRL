# coding=utf-8

import torch
import torch.nn as nn
from models.BaseRecModel import BaseRecModel


class MLP(BaseRecModel):
    @staticmethod
    def parse_model_args(parser, model_name='MLP'):
        parser.add_argument('--num_layers', type=int, default=3,
                            help="Number of mlp layers.")
        return BaseRecModel.parse_model_args(parser, model_name)

    def __init__(self, data_processor_dict, user_num, item_num, u_vector_size, i_vector_size,
                 num_layers=3, random_seed=2020, dropout=0, model_path='../model/Model/Model.pt'):
        self.num_layers = num_layers
        self.factor_size = u_vector_size // (2 ** (self.num_layers - 1))
        BaseRecModel.__init__(self, data_processor_dict, user_num, item_num, u_vector_size, i_vector_size,
                              random_seed=random_seed, dropout=dropout, model_path=model_path)

    @staticmethod
    def init_weights(m):
        """
        initialize nn weights，called in main.py
        :param m: parameter or the nn
        :return:
        """
        if type(m) == torch.nn.Linear:
            torch.nn.init.kaiming_uniform_(m.weight, a=1, nonlinearity='sigmoid')
            if m.bias is not None:
                m.bias.data.zero_()
        elif type(m) == torch.nn.Embedding:
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def _init_nn(self):
        # Init embeddings
        self.uid_embeddings = torch.nn.Embedding(self.user_num + 1, self.u_vector_size)
        self.iid_embeddings = torch.nn.Embedding(self.item_num + 1, self.u_vector_size)

        # Init MLP
        self.mlp = nn.ModuleList([])
        pre_size = self.factor_size * (2 ** self.num_layers)
        for i in range(self.num_layers):
            self.mlp.append(nn.Dropout(p=self.dropout))
            self.mlp.append(nn.Linear(pre_size, pre_size // 2))
            self.mlp.append(nn.ReLU())
            pre_size = pre_size // 2
        self.mlp = nn.Sequential(*self.mlp)

        # Init predictive layer
        self.p_layer = nn.ModuleList([])
        assert pre_size == self.factor_size
        # pre_size = pre_size * 2
        self.prediction = torch.nn.Linear(pre_size, 1)

    def predict(self, feed_dict):
        check_list = []
        u_ids = feed_dict['uid']
        pos_ids = feed_dict['pos']
        neg_ids = feed_dict['neg']

        mlp_u_vectors = self.uid_embeddings(u_ids)
        mlp_pos_vectors = self.iid_embeddings(pos_ids)
        mlp_neg_vectors = self.iid_embeddings(neg_ids)

        mlp_u_vectors_pos = torch.repeat_interleave(mlp_u_vectors.unsqueeze(1), pos_ids.size(-1), 1)
        mlp_pos = torch.cat((mlp_u_vectors_pos, mlp_pos_vectors), dim=-1)
        mlp_pos = self.mlp(mlp_pos)
        pos_prediction = self.prediction(mlp_pos).squeeze(-1)

        mlp_u_vectors_neg = torch.repeat_interleave(mlp_u_vectors.unsqueeze(1), neg_ids.size(-1), 1)
        mlp_neg = torch.cat((mlp_u_vectors_neg, mlp_neg_vectors), dim=-1)
        mlp_neg = self.mlp(mlp_neg)
        neg_prediction = self.prediction(mlp_neg).squeeze(-1)
        prediction = torch.cat((pos_prediction, neg_prediction), -1)

        out_dict = {'pos_prediction': pos_prediction,
                    'neg_prediction': neg_prediction,
                    'prediction': prediction,
                    'check': check_list,
                    'u_vectors': mlp_u_vectors}
        return out_dict

    def predict_vectors(self, vectors, feed_dict):
        check_list = []
        pos_ids = feed_dict['pos']
        neg_ids = feed_dict['neg']

        mlp_u_vectors = vectors
        mlp_pos_vectors = self.iid_embeddings(pos_ids)
        mlp_neg_vectors = self.iid_embeddings(neg_ids)

        mlp_u_vectors_pos = torch.repeat_interleave(mlp_u_vectors.unsqueeze(1), pos_ids.size(-1), 1)
        mlp_pos = torch.cat((mlp_u_vectors_pos, mlp_pos_vectors), dim=-1)
        mlp_pos = self.mlp(mlp_pos)
        pos_prediction = self.prediction(mlp_pos).squeeze(-1)

        mlp_u_vectors_neg = torch.repeat_interleave(mlp_u_vectors.unsqueeze(1), neg_ids.size(-1), 1)
        mlp_neg = torch.cat((mlp_u_vectors_neg, mlp_neg_vectors), dim=-1)
        mlp_neg = self.mlp(mlp_neg)
        neg_prediction = self.prediction(mlp_neg).squeeze(-1)
        prediction = torch.cat((pos_prediction, neg_prediction), -1)

        out_dict = {'pos_prediction': pos_prediction,
                    'neg_prediction': neg_prediction,
                    'prediction': prediction,
                    'check': check_list,
                    'u_vectors': mlp_u_vectors}
        return out_dict