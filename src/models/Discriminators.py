import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassAttacker(nn.Module):
    def __init__(self, embed_dim, feature_info,
                 model_dir_path='../model/Model/', model_name=''):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.feature_info = feature_info
        self.out_dim = feature_info.num_class
        self.name = feature_info.name
        self.model_file_name = \
            self.name + '_' + model_name + '_disc.pt' if model_name != '' else self.name + '_disc.pt'
        self.model_path = os.path.join(model_dir_path, self.model_file_name)
        self.optimizer = None

        if self.name == 'gender':
            self.num = [1682, 4268]
        elif self.name == 'age':
            self.num = [218, 1076, 2076, 1176, 540, 488, 376]
        elif self.name == 'occupation':
            self.num = [703, 519, 261, 173, 750, 109, 236, 662, 16, 91, 192, 129, 383, 140, 301, 139, 234, 496, 68, 71,
                        277]
        self.weight = (1 / torch.tensor(self.num))
        self.weight = self.weight / self.weight.sum()
        self.criterion = nn.NLLLoss(weight=self.weight)


        self.network = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.out_dim)
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

    def forward(self, embeddings, labels):
        output = self.predict(embeddings)['output']
        loss = self.criterion(output.squeeze(), labels)
        return loss

    def predict(self, embeddings):
        scores = self.network(embeddings)
        output = F.log_softmax(scores, dim=1)
        prediction = output.max(1, keepdim=True)[1]     # get the index of the max
        result_dict = {'output': output,
                       'prediction': prediction}
        return result_dict

    def save_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        dir_path = os.path.dirname(model_path)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        torch.save(self.state_dict(), model_path)
        # logging.info('Save ' + self.name + ' discriminator model to ' + model_path)

    def load_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        self.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.eval()
        # logging.info('Load ' + self.name + ' discriminator model from ' + model_path)


class AFRL_Discriminator(nn.Module):

    def __init__(self, embed_dim, feature_info,
                 model_dir_path='../model/Model/', model_name='',target=False):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.feature_info = feature_info
        self.out_dim = feature_info.num_class
        self.name = feature_info.name
        if target:
            self.model_file_name = \
                self.name + '_' + model_name + '_target.pt' if model_name != '' else self.name + '_target.pt'
        else:
            self.model_file_name = \
                self.name + '_' + model_name + '_disc.pt' if model_name != '' else self.name + '_disc.pt'
        self.model_path = os.path.join(model_dir_path, self.model_file_name)
        self.optimizer = None

        if self.name == 'gender':
            self.num = [1682, 4268]
        elif self.name == 'age':
            self.num = [218, 1076, 2076, 1176, 540, 488, 376]
        elif self.name == 'occupation':
            self.num = [703, 519, 261, 173, 750, 109, 236, 662, 16, 91, 192, 129, 383, 140, 301, 139, 234, 496, 68, 71,
                        277]
        self.weight = (1 / torch.tensor(self.num))
        self.weight = self.weight / self.weight.sum()
        self.criterion = nn.NLLLoss(weight=self.weight)

        #
        self.network = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.out_dim)
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

    def forward(self, embeddings, labels):
        output = self.predict(embeddings)['output']
        loss = self.criterion(output.squeeze(), labels)
        return loss

    def predict(self, embeddings):
        scores = self.network(embeddings)
        output = F.log_softmax(scores, dim=1)
        prediction = output.max(1, keepdim=True)[1]     # get the index of the max
        result_dict = {'output': output,
                       'prediction': prediction}
        return result_dict

    def save_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        dir_path = os.path.dirname(model_path)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        torch.save(self.state_dict(), model_path)
        logging.info('Save ' + self.name + ' discriminator model to ' + model_path)

    def load_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        self.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.eval()
        logging.info('Load ' + self.name + ' discriminator model from ' + model_path)

