import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

class AFRL_CombineMLP(nn.Module):

    def __init__(self, embed_dim,
                 model_dir_path='../model/Model/', model_name=''):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.name = 'CombineMLP'
        self.model_file_name = \
            self.name + '.pt'
        self.model_path = os.path.join(model_dir_path, self.model_file_name)
        self.optimizer = None

        self.network = nn.Sequential(
            nn.Linear(self.embed_dim * 4, self.embed_dim * 8),
            nn.ReLU(),
            nn.Linear(self.embed_dim * 8, self.embed_dim * 4),
            nn.ReLU(),
            nn.Linear(self.embed_dim * 4, self.embed_dim),
            # nn.BatchNorm1d(embed_dim)
        )

    @staticmethod
    def init_weights(m):
        """
        initialize nn weights，called in main.py
        :param m: parameter or the nn
        :return:
        """
        if type(m) == torch.nn.Linear:
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                torch.nn.init.normal_(m.bias, mean=0.0, std=0.01)
        elif type(m) == torch.nn.Embedding:
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, embeddings):
        com_embeddings = self.network(embeddings)
        return com_embeddings

    def save_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        dir_path = os.path.dirname(model_path)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        torch.save(self.state_dict(), model_path)
        # logging.info('Save CML' + ' to ' + model_path)

    def load_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        self.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.eval()
        # logging.info('Load ' + self.name + ' discriminator model from ' + model_path)