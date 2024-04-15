import torch
from models.BaseRecModel import BaseRecModel


class PMF(BaseRecModel):
    def _init_nn(self):
        self.uid_embeddings = torch.nn.Embedding(self.user_num + 1, self.u_vector_size)
        self.iid_embeddings = torch.nn.Embedding(self.item_num + 1, self.u_vector_size)

    def predict(self, feed_dict):
        check_list = []
        u_ids = feed_dict['uid']
        pos_ids = feed_dict['pos']
        neg_ids = feed_dict['neg']

        pmf_u_vectors = self.uid_embeddings(u_ids)
        pmf_pos_vectors = self.iid_embeddings(pos_ids)
        pmf_neg_vectors = self.iid_embeddings(neg_ids)

        pos_prediction = (pmf_u_vectors.unsqueeze(1) * pmf_pos_vectors).sum(dim=-1)
        neg_prediction = (pmf_u_vectors.unsqueeze(1) * pmf_neg_vectors).sum(dim=-1)
        prediction = torch.cat((pos_prediction, neg_prediction), -1)

        out_dict = {'pos_prediction': pos_prediction,
                    'neg_prediction': neg_prediction,
                    'prediction' : prediction,
                    'check': check_list,
                    'u_vectors': pmf_u_vectors}
        return out_dict

    def predict_vectors(self, vectors, feed_dict):
        check_list = []
        pos_ids = feed_dict['pos']
        neg_ids = feed_dict['neg']

        pmf_u_vectors = vectors
        pmf_pos_vectors = self.iid_embeddings(pos_ids)
        pmf_neg_vectors = self.iid_embeddings(neg_ids)

        pos_prediction = (pmf_u_vectors.unsqueeze(1) * pmf_pos_vectors).sum(dim=-1)
        neg_prediction = (pmf_u_vectors.unsqueeze(1) * pmf_neg_vectors).sum(dim=-1)
        prediction = torch.cat((pos_prediction, neg_prediction), -1)

        out_dict = {'pos_prediction': pos_prediction,
                    'neg_prediction': neg_prediction,
                    'prediction' : prediction,
                    'check': check_list,
                    'u_vectors': pmf_u_vectors}
        return out_dict