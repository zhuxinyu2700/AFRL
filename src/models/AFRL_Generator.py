import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

class AFRL_Generator(nn.Module):
    def __init__(self, embed_dim, feature_info,
                 model_dir_path='../model/Model/', model_name='', type_id=False):
        super(AFRL_Generator, self).__init__()

        self.embed_dim = int(embed_dim)
        self.neg_slope = 0.2
        self.feature_info = feature_info
        self.name = feature_info.name
        self.model_file_name = \
            self.name + '_' + model_name + '_generator.pt' if model_name != '' else self.name + '_generator.pt'
        self.model_path = os.path.join(model_dir_path, self.model_file_name)
        self.optimizer = None
        self.type_id = type_id

        # best
        self.generator_network = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim * 2, bias=True),
            nn.ReLU(),
            nn.Linear(self.embed_dim * 2, self.embed_dim * 4, bias=True),
            nn.ReLU(),
            nn.Linear(self.embed_dim * 4, self.embed_dim * 8, bias=True),
            nn.ReLU(),
            nn.Linear(self.embed_dim * 8, self.embed_dim * 4, bias=True),
            nn.ReLU(),
            nn.Linear(self.embed_dim * 4, self.embed_dim * 2, bias=True),
            nn.ReLU(),
            nn.Linear(self.embed_dim * 2, self.embed_dim, bias=True),
        )


    @staticmethod
    def init_weights(m):
        """
        initialize nn weights，called in main.py
        :param m: parameter or the nn
        :return:
        """
        if type(m) == torch.nn.Linear:
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.1)
            if m.bias is not None:
                torch.nn.init.normal_(m.bias, mean=0.0, std=0.1)
        elif type(m) == torch.nn.Embedding:
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.1)

    def forward(self, x):
        feature_emb = self.generator_network(x)
        return feature_emb

    def save_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        dir_path = os.path.dirname(model_path)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        torch.save(self.state_dict(), model_path)
        logging.info('Save ' + self.name + ' Generator to ' + model_path)

    def load_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        self.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.eval()
        logging.info('Load ' + self.name + ' generator model from ' + model_path)

