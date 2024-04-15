# coding=utf-8
from utils.metrics import *
from sklearn.metrics import *
from sklearn.preprocessing import LabelBinarizer
from utils.generic import *
from utils.constants import *
import itertools as it
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
from time import time
import numpy as np
import pandas as pd
import gc
import os
import torch.nn.functional as F
import torch.nn as nn


class RecRunner:
    @staticmethod
    def parse_runner_args(parser):
        """
        跑模型的命令行参数
        :param parser:
        :return:
        """
        parser.add_argument('--load', type=int, default=0,
                            help='Whether load model and continue to train')
        parser.add_argument('--load_attack', action='store_true',
                            help='Whether load attacker model and continue to train')
        parser.add_argument('--epoch', type=int, default=500,
                            help='Number of epochs.')
        parser.add_argument('--disc_epoch', type=int, default=300,
                            help='Number of epochs for training extra discriminator.')
        parser.add_argument('--check_epoch', type=int, default=1,
                            help='Check every epochs.')
        parser.add_argument('--early_stop', type=int, default=1,
                            help='whether to early-stop.')
        parser.add_argument('--lr', type=float, default=1e-3,
                            help='Learning rate.')
        parser.add_argument('--lr_attack', type=float, default=1e-3,
                            help='attacker learning rate.')
        parser.add_argument('--batch_size', type=int, default=1024,
                            help='Batch size during training.')
        parser.add_argument('--vt_batch_size', type=int, default=1024,
                            help='Batch size during testing.')
        parser.add_argument('--dropout', type=float, default=0.2,
                            help='Dropout probability for each deep layer')
        parser.add_argument('--l2', type=float, default=1e-8,
                            help='Weight of l2_regularize in loss.')
        parser.add_argument('--l2_attack', type=float, default=1e-4,
                            help='Weight of attacker l2_regularize in loss.')
        parser.add_argument('--no_filter', action='store_true',
                            help='if or not use filters')
        parser.add_argument('--filter_mode', type=str, default="none")
        parser.add_argument('--lambda_weight', type=float,
                            default=1, help='afrl: the value of lambda')
        parser.add_argument('--beta_weight', type=float,
                            default=1, help='afrl: the value of beta')
        parser.add_argument('--d_steps', type=int,
                            default=1,
                            help='the number of steps of updating discriminator')
        parser.add_argument('--optimizer', type=str, default='GD',
                            help='optimizer: GD, Adam, Adagrad')
        parser.add_argument('--metric', type=str, default="RMSE",
                            help='metrics: RMSE, MAE, AUC, F1, Accuracy, Precision, Recall')
        parser.add_argument('--skip_eval', type=int, default=0,
                            help='number of epochs without evaluation')
        parser.add_argument('--num_worker', type=int, default=0,
                            help='number of processes for multi-processing data loading.')
        parser.add_argument('--fix_one', action='store_true',
                            help='fix one feature for evaluation.')
        parser.add_argument('--eval_disc', action='store_true',
                            help='train extra discriminator for evaluation.')
        return parser

    def __init__(self, optimizer='GD', learning_rate=0.01, epoch=100, batch_size=128, eval_batch_size=128 * 128,
                 dropout=0, l2=1e-5, metrics='RMSE', check_epoch=10, early_stop=1, num_worker=1, no_filter=False,
                 filter_mode='none',
                 beta_weight=1, lambda_weight=1,
                 d_steps=100, disc_epoch=1000, model_name=None):
        """
        初始化
        :param optimizer: optimizer name
        :param learning_rate: learning rate
        :param epoch: total training epochs
        :param batch_size: batch size for training
        :param eval_batch_size: batch size for evaluation
        :param dropout: dropout rate
        :param l2: l2 weight
        :param metrics: evaluation metrics list
        :param check_epoch: check intermediate results in every n epochs
        :param early_stop: 1 for early stop, 0 for not.
        """
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.dropout = dropout
        self.no_dropout = 0.0
        self.l2_weight = l2
        self.filter_mode = filter_mode
        self.beta_weight = beta_weight
        self.lambda_weight = lambda_weight
        self.d_steps = d_steps
        self.no_filter = no_filter
        self.disc_epoch = disc_epoch
        self.model_name = model_name

        # convert metrics to list of str
        self.metrics = metrics.lower().split(',')
        self.check_epoch = check_epoch
        self.early_stop = early_stop
        self.time = None

        # record train, validation, test results
        self.train_results, self.valid_results, self.test_results = [], [], []
        self.disc_results = []
        self.num_worker = num_worker

    def _build_optimizer(self, model, lr=None, l2_weight=None):
        optimizer_name = self.optimizer_name.lower()
        if lr is None:
            lr = self.learning_rate
        if l2_weight is None:
            l2_weight = self.l2_weight

        if optimizer_name == 'gd':
            logging.info("Optimizer: GD")
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=l2_weight)
        elif optimizer_name == 'adagrad':
            logging.info("Optimizer: Adagrad")
            optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=l2_weight)
        elif optimizer_name == 'adam':
            logging.info("Optimizer: Adam")
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_weight)
        else:
            logging.error("Unknown Optimizer: " + self.optimizer_name)
            assert self.optimizer_name in ['GD', 'Adagrad', 'Adam']
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=l2_weight)
        return optimizer

    def _check_time(self, start=False):
        if self.time is None or start:
            self.time = [time()] * 2
            return self.time[0]
        tmp_time = self.time[1]
        self.time[1] = time()
        return self.time[1] - tmp_time

    @staticmethod
    def get_filter_mask(filter_num):
        return np.random.choice([0, 1], size=(filter_num,))

    @staticmethod
    def _get_masked_disc(disc_dict, labels, mask):
        if np.sum(mask) == 0:
            return []
        masked_disc_label = [(disc_dict[i + 1], labels[:, i]) for i, val in enumerate(mask) if val != 0]
        return masked_disc_label

    def fit(self, model, batches, epoch=-1):  # fit the results for an input set
        """
        Train the model
        :param model: model instance
        :param batches: train data in batches
        :param epoch: epoch number
        :return: return the output of the last round
        """
        gc.collect()
        torch.cuda.empty_cache()

        if model.optimizer is None:
            model.optimizer = self._build_optimizer(model)
        model.train()

        loss_list = list()
        output_dict = dict()
        for batch in tqdm(batches, leave=False, desc='Epoch %5d' % (epoch + 1),
                          ncols=100, mininterval=1):
            batch = batch_to_gpu(batch)
            model.optimizer.zero_grad()
            result_dict = model(batch)
            rec_loss = result_dict['loss']
            rec_loss.backward()
            model.optimizer.step()

            loss_list.append(result_dict['loss'].detach().cpu().data.numpy())
            output_dict['check'] = result_dict['check']

        output_dict['loss'] = np.mean(loss_list)
        return output_dict

    def train(self, model, dp_dict, skip_eval=0, fix_one=False,
              ex_dp_dict=None, ex_fair_disc_dict=None, lr_attack=0.1, l2_attack=0.1,
              afrl_feature_generator_dict=None, afrl_target_disc_dict=None,
              afrl_fair_disc_dict=None, afrl_combine_mlp=None):

        """
        Train model
        :param model: model obj
        :param dp_dict: Data processors for train valid and test
        :param skip_eval: number of epochs to skip for evaluations
        :param fair_disc_dict: fairness discriminator dictionary
        :return:
        """
        train_data = DataLoader(dp_dict['train'], batch_size=self.batch_size, num_workers=self.num_worker,
                                shuffle=True, collate_fn=dp_dict['train'].collate_fn)
        validation_data = DataLoader(dp_dict['valid'], batch_size=None, num_workers=self.num_worker,
                                     pin_memory=True, collate_fn=dp_dict['test'].collate_fn)

        self._check_time(start=True)  # start time
        try:
            for epoch in range(self.epoch):
                self._check_time()

                # if epoch % 10 == 0:
                #     if self.no_filter:
                #         fair_result_dict \
                #             = self.train_discriminator(model, ex_dp_dict, ex_fair_disc_dict, lr_attack, l2_attack)
                #     else:
                #         fair_result_dict \
                #             = self.train_discriminator(model, ex_dp_dict, ex_fair_disc_dict, lr_attack, l2_attack,
                #                                        afrl_feature_generator_dict=afrl_feature_generator_dict,
                #                                        afrl_combine_mlp=afrl_combine_mlp)
                #
                #     for key in fair_result_dict:
                #         auc_dict = fair_result_dict[key]
                #         key = key.split(',')
                #         key = list(map(int, key))
                #         auc = 0
                #         for i in range(len(key)):
                #             if key[i] == 1:
                #                 auc += list(auc_dict.values())[i]
                #         if sum(key) != 0:
                #             auc = auc / sum(key)
                #         logging.info("mask: {} avg.auc:{}".format(key, auc))

                if self.no_filter:
                    output_dict = self.fit(model, train_data, epoch=epoch)
                else:
                    output_dict = self.fit_afrl(model, afrl_feature_generator_dict, afrl_target_disc_dict,
                                           afrl_fair_disc_dict, afrl_combine_mlp, train_data, epoch=epoch)
                if self.check_epoch > 0 and (epoch == 1 or epoch % self.check_epoch == 0):
                    if self.no_filter:
                        self.check(model, output_dict)
                    else:
                        self.check_afrl(model, output_dict)
                training_time = self._check_time()


                if epoch >= skip_eval or epoch%10==0:
                    valid_result_dict, test_result_dict = None, None
                    if self.no_filter:
                        valid_result = self.evaluate(model, validation_data) if \
                            validation_data is not None else [-1.0] * len(self.metrics)
                    else:
                        valid_result, valid_result_dict = \
                            self.eval_multi_combination(model, validation_data, fix_one,
                                                        afrl_feature_generator_dict=afrl_feature_generator_dict,
                                                        afrl_combine_mlp=afrl_combine_mlp) \
                                if validation_data is not None else [-1.0] * len(self.metrics)

                    testing_time = self._check_time()

                    self.valid_results.append(valid_result)

                    if self.no_filter:
                        logging.info("Epoch %5d [%.1f s]\n validation= %s [%.1f s] "
                                     % (epoch + 1, training_time,
                                        format_metric(valid_result),
                                        testing_time) + ','.join(self.metrics))
                    else:
                        logging.info("Epoch %5d [%.1f s]\t Average: validation= %s [%.1f s] "
                                     % (epoch + 1, training_time,
                                        format_metric(valid_result),
                                        testing_time) + ','.join(self.metrics))
                        for key in valid_result_dict:
                            logging.info("validation= %s "
                                         % (format_metric(valid_result_dict[key])) + ','.join(self.metrics) +
                                         ' (' + key + ') ')

                    if best_result(self.metrics[0], self.valid_results) == self.valid_results[-1]:
                        if self.no_filter:
                            model.save_model()
                        else:
                            for idx in afrl_feature_generator_dict:
                                afrl_feature_generator_dict[idx].save_model()
                            for idx in afrl_target_disc_dict:
                                afrl_target_disc_dict[idx].save_model()
                            for idx in afrl_fair_disc_dict:
                                afrl_fair_disc_dict[idx].save_model()
                            afrl_combine_mlp.save_model()

                    if self.eva_termination() and self.early_stop == 1:
                        logging.info("Early stop at %d based on validation result." % (epoch + 1))
                        break
                if epoch < skip_eval:
                    logging.info("Epoch %5d [%.1f s]" % (epoch + 1, training_time))
        except KeyboardInterrupt:
            logging.info("Early stop manually")
            save_here = input("Save here? (1/0) (default 0):")
            if str(save_here).lower().startswith('1'):
                if self.no_filter:
                    model.save_model()
                else:
                    for idx in afrl_feature_generator_dict:
                        afrl_feature_generator_dict[idx].save_model()
                    for idx in afrl_target_disc_dict:
                        afrl_target_disc_dict[idx].save_model()
                    for idx in afrl_fair_disc_dict:
                        afrl_fair_disc_dict[idx].save_model()
                    afrl_combine_mlp.save_model()

        # Find the best validation result across iterations
        best_valid_score = best_result(self.metrics[0], self.valid_results)
        best_epoch = self.valid_results.index(best_valid_score)

        logging.info("Best Iter(validation)= %5d\t valid= %s [%.1f s] "
                     % (best_epoch + 1,
                        format_metric(self.valid_results[best_epoch]),
                        self.time[1] - self.time[0]))
        if self.no_filter:
            model.load_model()
        else:
            for idx in afrl_feature_generator_dict:
                afrl_feature_generator_dict[idx].load_model()
            for idx in afrl_target_disc_dict:
                afrl_target_disc_dict[idx].load_model()
            for idx in afrl_fair_disc_dict:
                afrl_fair_disc_dict[idx].load_model()
            afrl_combine_mlp.load_model()

    def eval_multi_combination(self, model, data, fix_one=False,
                               afrl_feature_generator_dict=None, afrl_combine_mlp=None):
        """
        The output is the averaged result over all the possible combinations.
        :param model: trained model
        :param data: validation or test data (not train data)
        :return: averaged evaluated result on given dataset
        """
        feature_info = model.data_processor_dict['train'].data_reader.feature_info

        if not fix_one:
            mask_list = [[0, 0, 0],[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
            # mask_list = [list(i) for i in it.product([0, 1], repeat=n_features)]
            # mask_list.pop(0)
            # mask_list = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        else:
            mask_list = [[0,0,0],[1,1,1]]
            # feature_range = np.arange(n_features)
            # shape = (feature_range.size, feature_range.max() + 1)
            # one_hot = np.zeros(shape).astype(int)
            # one_hot[feature_range, feature_range] = 1
            # mask_list = one_hot.tolist()
        result_dict = {}
        acc_result = None
        for mask in mask_list:
            mask = np.asarray(mask)
            feature_idx = np.where(mask == 1)[0]
            f_name_list = [feature_info[i + 1].name for i in feature_idx]
            f_name = ' '.join(f_name_list)

            if self.no_filter:
                cur_result = self.evaluate(model, data, mask) if data is not None else [-1.0] * len(self.metrics)
            else:
                cur_result = self.evaluate(model, data, mask,
                                           afrl_feature_generator_dict=afrl_feature_generator_dict, afrl_combine_mlp=afrl_combine_mlp) \
                    if data is not None else [-1.0] * len(self.metrics)
            acc_result = np.array(cur_result) if acc_result is None else acc_result + np.asarray(cur_result)

            result_dict[f_name] = cur_result

        if acc_result is not None:
            acc_result /= len(mask_list)

        return list(acc_result), result_dict

    @torch.no_grad()
    def evaluate(self, model, batches, mask=None, metrics=None,
                 afrl_feature_generator_dict=None, afrl_combine_mlp=None):
        """
        evaluate recommendation performance
        :param model:
        :param batches: data batches, each batch is a dict.
        :param mask: filter mask
        :param metrics: list of str
        :return: list of float number for each metric
        """
        if metrics is None:
            metrics = self.metrics
        model.eval()
        if self.filter_mode=='afrl':
            for idx in afrl_feature_generator_dict:
                afrl_feature_generator_dict[idx].eval()
            afrl_combine_mlp.eval()

        result_dict = defaultdict(list)
        for batch in tqdm(batches, leave=False, ncols=100, mininterval=1, desc='Predict'):
            batch = batch_to_gpu(batch)
            if self.no_filter:
                out_dict = model.predict(batch)
            else:
                if self.model_name == 'PMF' or self.model_name == 'MLP':
                    uids = batch['uid']
                    vectors = model.uid_embeddings(uids)
                elif self.model_name == 'SASRec' or self.model_name == 'BERT4rec':
                    seq = batch['seq']
                    vectors = model.log2feats(seq)[:, -1]
                combine_vectors = afrl_feature_generator_dict[0](vectors)
                for idx, val in enumerate(mask):
                    feature_vectors = afrl_feature_generator_dict[idx + 1](vectors)
                    if val == 1:
                        feature_vectors = feature_vectors * 0
                    combine_vectors = torch.cat((combine_vectors, feature_vectors), dim=1)
                vectors = afrl_combine_mlp(combine_vectors)
                out_dict = model.predict_vectors(vectors, batch)
            prediction = out_dict['prediction']
            prediction = prediction.cpu().numpy()
            results = self.evaluate_method(prediction, metrics=metrics)
            for key in results:
                result_dict[key].extend(results[key])

        evaluations = []
        for metric in metrics:
            evaluations.append(np.average(result_dict[metric]))

        return evaluations

    @staticmethod
    def evaluate_method(p, metrics):
        """
        Evaluate model predictions.
        :param p: predicted values, np.array
        :param data: data dictionary which include ground truth labels
        :param metrics: metrics list
        :return: a list of results. The order is consistent to metric list.
        """
        label = []
        evaluations = {}
        for metric in metrics:
            if metric == 'rmse':
                evaluations[metric] = [np.sqrt(mean_squared_error(label, p))]
            elif metric == 'mae':
                evaluations[metric] = [mean_absolute_error(label, p)]
            elif metric == 'auc':
                evaluations[metric] = [roc_auc_score(label, p)]
            else:
                k = int(metric.split('@')[-1])
                label = torch.cat((torch.Tensor([1]), torch.Tensor([0] * (len(p[0]) - 1))), 0)
                if metric.startswith('ndcg@'):
                    ndcgs = []
                    for idx in range(len(p)):
                        df = pd.DataFrame()
                        df['l'] = label
                        df['p'] = p[idx]
                        df = df.sort_values(by='p', ascending=False)
                        ndcgs.append(ndcg_at_k(df['l'].tolist()[:k], k=k, method=1))
                    evaluations[metric] = ndcgs
                elif metric.startswith('hit@'):
                    hits = []
                    for idx in range(len(p)):
                        df = pd.DataFrame()
                        df['l'] = label
                        df['p'] = p[idx]
                        df = df.sort_values(by='p', ascending=False)
                        hits.append(int(np.sum(df['l'][:k]) > 0))
                    evaluations[metric] = hits
        return evaluations

    def eva_termination(self):
        """
        Early stopper
        :return:
        """
        metric = self.metrics[0]
        valid = self.valid_results
        if len(valid) > 10 and metric in LOWER_METRIC_LIST and strictly_increasing(valid[-5:]):
            return True
        elif len(valid) > 10 and metric not in LOWER_METRIC_LIST and strictly_decreasing(valid[-5:]):
            return True
        elif len(valid) - valid.index(best_result(metric, valid)) > 10:
            return True
        return False

    def eva_termination_cla(self, valid_results, metric):
        """
        Early stopper
        :return:
        """
        valid = valid_results
        if len(valid) > 10 and metric in LOWER_METRIC_LIST and strictly_increasing(valid[-5:]):
            return True
        elif len(valid) > 10 and metric not in LOWER_METRIC_LIST and strictly_decreasing(valid[-5:]):
            return True
        elif len(valid) - valid.index(best_result(metric, valid)) > 10:
            return True
        return False

    @torch.no_grad()
    def _eval_discriminator(self, model, labels, u_vectors, fair_disc_dict, num_disc):
        feature_info = model.data_processor_dict['train'].data_reader.feature_info
        feature_eval_dict = {}
        for i in range(num_disc):
            discriminator = fair_disc_dict[i + 1]
            label = labels[:, i]
            # metric = 'auc' if feature_info[i + 1].num_class == 2 else 'f1'
            feature_name = feature_info[i + 1].name
            discriminator.eval()
            if feature_info[i + 1].num_class == 2:
                prediction = discriminator.predict(u_vectors)['output'].sigmoid()[:,1]
            else:
                prediction = discriminator.predict(u_vectors)['output']
            feature_eval_dict[feature_name] = {'label': label.cpu(), 'prediction': prediction.detach().cpu(),
                                               'num_class': feature_info[i + 1].num_class}
            discriminator.train()
        return feature_eval_dict

    @staticmethod
    def _disc_eval_method(label, prediction, num_class, metric='auc'):
        if metric == 'auc':
            if num_class == 2:
                score = roc_auc_score(label, prediction, average='micro')
                # score = roc_auc_score(label, prediction)
                score = max(score, 1 - score)
                return score
            else:
                lb = LabelBinarizer()
                classes = [i for i in range(num_class)]
                lb.fit(classes)
                label = lb.transform(label)
                # label = lb.fit_transform(label)
                score = roc_auc_score(label, prediction, multi_class='ovo')
                score = max(score, 1 - score)
                return score
        else:
            raise ValueError('Unknown evaluation metric in _disc_eval_method().')

    def check(self, model, out_dict):
        """
        Check intermediate results
        :param model: model obj
        :param out_dict: output dictionary
        :return:
        """
        check = out_dict
        logging.info(os.linesep)
        for i, t in enumerate(check['check']):
            d = np.array(t[1].detach().cpu())
            logging.info(os.linesep.join([t[0] + '\t' + str(d.shape), np.array2string(d, threshold=20)]) + os.linesep)

        loss, l2 = check['loss'], model.l2()
        l2 = l2 * self.l2_weight
        l2 = l2.detach()
        logging.info('loss = %.4f, l2 = %.4f' % (loss, l2))
        if not (np.absolute(loss) * 0.005 < l2 < np.absolute(loss) * 0.1):
            logging.warning('l2 inappropriate: loss = %.4f, l2 = %.4f' % (loss, l2))


    def train_discriminator(self, model, dp_dict, fair_disc_dict, lr_attack=None, l2_attack=None,
                            afrl_feature_generator_dict=None, afrl_combine_mlp=None):
        """
        Train discriminator to evaluate the quality of learned embeddings
        :param model: trained model
        :param dp_dict: Data processors for train valid and test
        :param fair_disc_dict: fairness discriminator dictionary
        :return:
        """
        model.eval()
        if self.filter_mode=='afrl':
            for idx in afrl_feature_generator_dict:
                afrl_feature_generator_dict[idx].eval()
            afrl_combine_mlp.eval()

        train_data = DataLoader(dp_dict['train'], batch_size=dp_dict['train'].batch_size, num_workers=self.num_worker,
                                shuffle=True, collate_fn=dp_dict['train'].collate_fn)
        valid_data = DataLoader(dp_dict['valid'], batch_size=dp_dict['valid'].batch_size, num_workers=self.num_worker,
                                pin_memory=True, collate_fn=dp_dict['valid'].collate_fn)
        test_data = DataLoader(dp_dict['test'], batch_size=dp_dict['test'].batch_size, num_workers=self.num_worker,
                               pin_memory=True, collate_fn=dp_dict['test'].collate_fn)
        self._check_time(start=True)  # 记录初始时间s

        mask_list = [[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1,1,1]]

        results_dict = dict()
        for mask in mask_list:
            for idx in fair_disc_dict:
                fair_disc_dict[idx].apply(fair_disc_dict[idx].init_weights)
            valid_results_fair = []
            for epoch in range(self.disc_epoch):
                self._check_time()
                if self.no_filter:
                    output_dict = \
                        self.fit_disc(model, train_data, fair_disc_dict, np.array(mask), epoch=epoch,
                                      lr_attack=lr_attack, l2_attack=l2_attack)
                    valid_result_dict = \
                        self.evaluation_disc(model, fair_disc_dict, valid_data, dp_dict['test'], mask)
                else:
                    output_dict = \
                        self.fit_disc(model, train_data, fair_disc_dict, np.array(mask), epoch=epoch,
                                      lr_attack=lr_attack, l2_attack=l2_attack,
                                      afrl_feature_generator_dict=afrl_feature_generator_dict, afrl_combine_mlp=afrl_combine_mlp)
                    valid_result_dict = \
                        self.evaluation_disc(model, fair_disc_dict, valid_data, dp_dict['test'], mask,
                        afrl_feature_generator_dict = afrl_feature_generator_dict, afrl_combine_mlp = afrl_combine_mlp)
                valid_result_auc = []
                if mask == [0,0,0] or mask == [1,1,1]:
                    for f_name in valid_result_dict['d_score']:
                        valid_result_auc.append(valid_result_dict['d_score'][f_name])
                    valid_results_fair.append(np.mean(valid_result_auc))
                else:
                    mask_id = 0
                    for f_name in valid_result_dict['d_score']:
                        if mask[mask_id]==1:
                            valid_result_auc.append(valid_result_dict['d_score'][f_name])
                        mask_id+=1
                    valid_results_fair.append(np.mean(valid_result_auc))

                if best_result('AUC', valid_results_fair) == valid_results_fair[-1]:
                    for idx in fair_disc_dict:
                        fair_disc_dict[idx].save_model()

                if self.eva_termination_cla(valid_results_fair,'AUC') and epoch>100:
                    logging.info("Early stop at %d based on result." % (epoch + 1))
                    break

            if self.no_filter:
                result_dict = \
                    self.evaluation_disc(model, fair_disc_dict, test_data, dp_dict['train'], mask)
            else:
                result_dict = \
                    self.evaluation_disc(model, fair_disc_dict, test_data, dp_dict['train'], mask,
                                         afrl_feature_generator_dict=afrl_feature_generator_dict, afrl_combine_mlp=afrl_combine_mlp)
            mask = list(map(str, mask))
            mask = ",".join(mask)
            results_dict[mask] = result_dict['d_score']
        return results_dict


    def fit_disc(self, model, batches, fair_disc_dict, mask, epoch=-1, lr_attack=None, l2_attack=None,
                 afrl_feature_generator_dict=None, afrl_combine_mlp=None):
        """
        Train the discriminator
        :param model: model instance
        :param batches: train data in batches
        :param fair_disc_dict: fairness discriminator dictionary
        :param epoch: epoch number
        :param lr_attack: attacker learning rate
        :param l2_attack: l2 regularization weight for attacker
        """
        gc.collect()
        torch.cuda.empty_cache()

        for idx in fair_disc_dict:
            discriminator = fair_disc_dict[idx]
            if discriminator.optimizer is None:
                discriminator.optimizer = self._build_optimizer(discriminator, lr=lr_attack, l2_weight=l2_attack)
            discriminator.train()

        output_dict = dict()
        loss_acc = defaultdict(list)

        eval_dict = None
        for batch in tqdm(batches, leave=False, desc='Epoch %5d' % (epoch + 1),
                          ncols=100, mininterval=1):
            batch = batch_to_gpu(batch)

            labels = batch['features']
            masked_disc_label = \
                    self._get_masked_disc(fair_disc_dict, labels, mask + 1)

            if self.no_filter:
                if self.model_name == 'PMF' or self.model_name == 'MLP':
                    uids = batch['uid']
                    vectors = model.uid_embeddings(uids)
                elif self.model_name == 'SASRec' or self.model_name == 'BERT4rec':
                    seq = batch['seq']
                    vectors = model.log2feats(seq)[:, -1]
            else:
                if self.model_name == 'PMF' or self.model_name == 'MLP':
                    uids = batch['uid']
                    vectors = model.uid_embeddings(uids)
                elif self.model_name == 'SASRec' or self.model_name == 'BERT4rec':
                    seq = batch['seq']
                    vectors = model.log2feats(seq)[:, -1]
                combine_vectors = afrl_feature_generator_dict[0](vectors)
                for idx, val in enumerate(mask):
                    feature_vectors = afrl_feature_generator_dict[idx + 1](vectors)
                    if val == 1:
                        feature_vectors = feature_vectors * 0
                    combine_vectors = torch.cat((combine_vectors, feature_vectors), dim=1)
                vectors = afrl_combine_mlp(combine_vectors)

            output_dict['check'] = []

            # update discriminator
            if len(masked_disc_label) != 0:
                for idx, (discriminator, label) in enumerate(masked_disc_label):
                    discriminator.optimizer.zero_grad()
                    disc_loss = discriminator(vectors.detach(), label)
                    disc_loss.backward()
                    discriminator.optimizer.step()
                    loss_acc[discriminator.name].append(disc_loss.detach().cpu())

        for key in loss_acc:
            loss_acc[key] = np.mean(loss_acc[key])

        output_dict['loss'] = loss_acc
        return output_dict

    @torch.no_grad()
    def evaluation_disc(self, model, fair_disc_dict, test_data, dp, mask,
                        afrl_feature_generator_dict=None, afrl_combine_mlp=None):
        num_features = dp.data_reader.num_features

        def eval_disc(labels, u_vectors, fair_disc_dict, mask):
            feature_info = dp.data_reader.feature_info
            feature_eval_dict = {}
            for i, val in enumerate(mask):
                # if val == 0:
                #     continue
                discriminator = fair_disc_dict[i + 1]
                label = labels[:, i]
                # metric = 'auc' if feature_info[i + 1].num_class == 2 else 'f1'
                feature_name = feature_info[i + 1].name
                discriminator.eval()
                if feature_info[i + 1].num_class == 2:
                    prediction = discriminator.predict(u_vectors)['prediction'].squeeze()
                else:
                    prediction = discriminator.predict(u_vectors)['output']
                feature_eval_dict[feature_name] = {'label': label.cpu(), 'prediction': prediction.detach().cpu(),'num_class': feature_info[i + 1].num_class}
                discriminator.train()
            return feature_eval_dict

        eval_dict = {}
        for batch in test_data:
            batch = batch_to_gpu(batch)

            labels = batch['features']
            if self.no_filter:
                if self.model_name == 'PMF' or self.model_name == 'MLP':
                    uids = batch['uid']
                    vectors = model.uid_embeddings(uids)
                elif self.model_name == 'SASRec' or self.model_name == 'BERT4rec':
                    seq = batch['seq']
                    vectors = model.log2feats(seq)[:, -1]
            else:
                if self.model_name == 'PMF' or self.model_name == 'MLP':
                    uids = batch['uid']
                    vectors = model.uid_embeddings(uids)
                elif self.model_name == 'SASRec' or self.model_name == 'BERT4rec':
                    seq = batch['seq']
                    vectors = model.log2feats(seq)[:, -1]
                combine_vectors = afrl_feature_generator_dict[0](vectors)
                for idx, val in enumerate(mask):
                    feature_vectors = afrl_feature_generator_dict[idx + 1](vectors)
                    if val == 1:
                        feature_vectors = feature_vectors * 0
                    combine_vectors = torch.cat((combine_vectors, feature_vectors), dim=1)
                vectors = afrl_combine_mlp(combine_vectors)
            batch_eval_dict = eval_disc(labels, vectors.detach(), fair_disc_dict, mask)

            for f_name in batch_eval_dict:
                if f_name not in eval_dict:
                    eval_dict[f_name] = batch_eval_dict[f_name]
                else:
                    new_label = batch_eval_dict[f_name]['label']
                    current_label = eval_dict[f_name]['label']
                    eval_dict[f_name]['label'] = torch.cat((current_label, new_label), dim=0)

                    new_prediction = batch_eval_dict[f_name]['prediction']
                    current_prediction = eval_dict[f_name]['prediction']
                    eval_dict[f_name]['prediction'] = torch.cat((current_prediction, new_prediction), dim=0)

        # generate discriminator evaluation scores
        d_score_dict = {}
        if eval_dict is not None:
            for f_name in eval_dict:
                l = eval_dict[f_name]['label']
                pred = eval_dict[f_name]['prediction']
                n_class = eval_dict[f_name]['num_class']
                d_score_dict[f_name] = self._disc_eval_method(l, pred, n_class)

        output_dict = dict()
        output_dict['d_score'] = d_score_dict
        return output_dict

    @staticmethod
    def check_disc(out_dict):
        check = out_dict
        logging.info(os.linesep)
        for i, t in enumerate(check['check']):
            d = np.array(t[1].detach().cpu())
            logging.info(os.linesep.join([t[0] + '\t' + str(d.shape), np.array2string(d, threshold=20)]) + os.linesep)

        loss_dict = check['loss']
        for disc_name, disc_loss in loss_dict.items():
            logging.info('%s loss = %.4f' % (disc_name, disc_loss))

        # for discriminator
        if 'd_score' in out_dict:
            disc_score_dict = out_dict['d_score']
            for feature in disc_score_dict:
                logging.info('{} AUC = {:.4f}'.format(feature, disc_score_dict[feature]))

    def fit_afrl(self, model, feature_generator_dict, target_disc_dict, fair_disc_dict, combine_mlp, batches, epoch=-1):  # fit the results for an input set
        """
        Train the model and the afrl model
        :param model: model instance
        :param batches: train data in batches
        :param feature_generator_dict: attribute encoder
        :param target_disc_dict: information alignment classifier dictionary
        :param fair_disc_dict: fairness discriminator dictionary
        :param combine_mlp: Information Aggregation mlp
        :param epoch: epoch number
        :return: return the output of the last round
        """
        gc.collect()
        torch.cuda.empty_cache()

        model.eval()

        for idx in feature_generator_dict:
            feature_generator = feature_generator_dict[idx]
            if feature_generator.optimizer is None:
                feature_generator.optimizer = \
                    torch.optim.Adam(feature_generator.parameters(), lr=self.learning_rate, weight_decay=self.l2_weight)
            feature_generator.train()

        for idx in target_disc_dict:
            discriminator = target_disc_dict[idx]
            if discriminator.optimizer is None:
                discriminator.optimizer = \
                    torch.optim.Adam(discriminator.parameters(), lr=self.learning_rate, weight_decay=self.l2_weight)
            discriminator.train()

        for idx in fair_disc_dict:
            discriminator = fair_disc_dict[idx]
            if discriminator.optimizer is None:
                discriminator.optimizer = \
                    torch.optim.Adam(discriminator.parameters(), lr=self.learning_rate, weight_decay=self.l2_weight)
            discriminator.train()

        if combine_mlp.optimizer is None:
            combine_mlp.optimizer = \
                    torch.optim.Adam(combine_mlp.parameters(), lr=self.learning_rate, weight_decay=self.l2_weight)
        combine_mlp.train()

        recommend_loss_list = list()
        adv_loss_list_dict = dict()
        target_loss_list_dict = dict()
        for idx in feature_generator_dict:
            adv_loss_list_dict[idx] = list()
            target_loss_list_dict[idx] = list()
        output_dict = dict()
        for batch in tqdm(batches, leave=False, desc='Epoch %5d' % (epoch + 1),
                          ncols=100, mininterval=1):
            mask = self.get_filter_mask(3)
            batch = batch_to_gpu(batch)
            labels = batch['features']

            result_dict = model(batch)
            vectors = result_dict['u_vectors']

            feature_vectors_dict = dict()
            combine_vectors = None
            # mask = [0,0,0]
            for idx in feature_generator_dict:
                feature_vectors = feature_generator_dict[idx](vectors.detach())
                feature_vectors_dict[idx] = feature_vectors
                if idx!=0 and mask[idx-1] == 1:
                    feature_vectors = feature_vectors * 0
                try:
                    combine_vectors = torch.cat((combine_vectors, feature_vectors), 1)
                except:
                    combine_vectors = feature_vectors

            target_disc_label = self._get_masked_disc(target_disc_dict, labels, [1, 1, 1])
            fair_disc_label = self._get_masked_disc(fair_disc_dict, labels, [1, 1, 1])
            feature_loss = 0
            for feature_idx, feature_vectors in feature_vectors_dict.items():
                if feature_idx != 0:
                    target_disc, label = target_disc_label[feature_idx-1]
                    target_loss = target_disc(feature_vectors, label)
                    adv_loss = torch.norm(feature_vectors, dim=1).mean()
                    feature_loss += target_loss + self.beta_weight * adv_loss

                    target_loss_list_dict[feature_idx].append(target_loss.detach().cpu().data.numpy())
                    adv_loss_list_dict[feature_idx].append(adv_loss.detach().cpu().data.numpy())
                else:
                    adv_loss = 0
                    for fair_disc, label in fair_disc_label:
                        adv_loss += fair_disc(feature_vectors, label)
                    target_loss = F.mse_loss(feature_vectors, vectors.detach())
                    ce_loss = target_loss - self.lambda_weight * adv_loss

                    target_loss_list_dict[feature_idx].append(target_loss.detach().cpu().data.numpy())
                    adv_loss_list_dict[feature_idx].append(adv_loss.detach().cpu().data.numpy())

            for idx in feature_generator_dict:
                if idx!=0:
                    feature_generator_dict[idx].optimizer.zero_grad()
                    target_disc_dict[idx].optimizer.zero_grad()
            feature_loss.backward()
            for idx in feature_generator_dict:
                if idx!=0:
                    feature_generator_dict[idx].optimizer.step()
                    target_disc_dict[idx].optimizer.step()

            feature_generator_dict[0].optimizer.zero_grad()
            ce_loss.backward()
            feature_generator_dict[0].optimizer.step()

            for _ in range(self.d_steps):
                for discriminator, label in fair_disc_label:
                    discriminator.optimizer.zero_grad()
                    disc_loss = discriminator(feature_vectors_dict[0].detach(), label)
                    disc_loss.backward()
                    discriminator.optimizer.step()

            combine_vectors = combine_mlp(combine_vectors.detach())
            result_dict = model.forward_vectors(combine_vectors, batch)
            loss = result_dict['loss']
            combine_mlp.optimizer.zero_grad()
            loss.backward()
            combine_mlp.optimizer.step()

            recommend_loss_list.append(loss.detach().cpu().data.numpy())
            output_dict['check'] = result_dict['check']

        output_dict['target_loss'] = target_loss_list_dict
        output_dict['adv_loss'] = adv_loss_list_dict
        output_dict['recommend_loss'] = np.mean(recommend_loss_list)
        return output_dict

    def check_afrl(self, model, out_dict):
        """
        Check intermediate results
        :param model: model obj
        :param out_dict: output dictionary
        :return:
        """
        check = out_dict
        logging.info(os.linesep)
        for i, t in enumerate(check['check']):
            d = np.array(t[1].detach().cpu())
            logging.info(os.linesep.join([t[0] + '\t' + str(d.shape), np.array2string(d, threshold=20)]) + os.linesep)

        recommend_loss = check['recommend_loss']
        l2 = model.l2()
        l2 = l2 * self.l2_weight
        l2 = l2.detach()
        logging.info('recommend_loss = %.4f, l2 = %.4f' % (recommend_loss, l2))

        target_loss_list_dict = check['target_loss']
        adv_loss_list_dict = check['adv_loss']
        for idx in target_loss_list_dict:
            logging.info('idx = %.1f, target_loss = %.4f, adv_loss = %.4f'
                         % (idx, np.mean(target_loss_list_dict[idx]), np.mean(adv_loss_list_dict[idx])))


