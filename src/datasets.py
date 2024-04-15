from utils.constants import *
from utils.generic import *
from tqdm import tqdm
import pickle
import os


class RecDataset:
    @staticmethod
    def parse_dp_args(parser):
        """
        DataProcessor related argument parser
        :param parser: argument parser
        :return: updated argument parser
        """
        parser.add_argument('--train_num_neg', type=int, default=1,
                            help='Negative sample num for each instance in train set.')
        parser.add_argument('--vt_num_neg', type=int, default=99,
                            help='Number of negative sample in validation/testing stage.')
        return parser

    def __init__(self, data_reader, stage, batch_size=1024, num_neg=1):
        self.data_reader = data_reader
        self.num_user = len(data_reader.user_ids_set)
        self.num_item = len(data_reader.item_ids_set)
        self.batch_size = batch_size
        self.stage = stage
        self.num_neg = num_neg
        # prepare test/validation dataset
        valid_pkl_path = os.path.join(self.data_reader.path, self.data_reader.file_name + VALID_PKL_SUFFIX)
        test_pkl_path = os.path.join(self.data_reader.path, self.data_reader.file_name + TEST_PKL_SUFFIX)
        if self.stage == 'valid':
            if os.path.exists(valid_pkl_path):
                with open(valid_pkl_path, 'rb') as file:
                    logging.info('Load validation data from pickle file.')
                    self.data = pickle.load(file)
            else:
                self.data = self._get_data()
                with open(valid_pkl_path, 'wb') as file:
                    pickle.dump(self.data, file)
        elif self.stage == 'test':
            if os.path.exists(test_pkl_path):
                with open(test_pkl_path, 'rb') as file:
                    logging.info('Load test data from pickle file.')
                    self.data = pickle.load(file)
            else:
                self.data = self._get_data()
                with open(test_pkl_path, 'wb') as file:
                    pickle.dump(self.data, file)
        else:
            self.data = self._get_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def _get_data(self):
        if self.stage == 'train':
            return self._get_train_data()
        else:
            return self._get_vt_data()

    def _get_train_data(self):
        df = self.data_reader.train_df
        df[SAMPLE_ID] = df.index
        columns_order = [USER, IID, SAMPLE_ID, LABEL] + [f_col for f_col in self.data_reader.feature_columns] + [SEQ]
        data = df[columns_order].to_numpy()
        return data

    def _get_vt_data(self):
        if self.stage == 'valid':
            df = self.data_reader.validation_df
            logging.info('Prepare validation data...')
        elif self.stage == 'test':
            df = self.data_reader.test_df
            logging.info('Prepare test data...')
        else:
            raise ValueError('Wrong stage in dataset.')
        df[SAMPLE_ID] = df.index
        columns_order = [USER, IID, SAMPLE_ID, LABEL] + [f_col for f_col in self.data_reader.feature_columns] + [SEQ]
        data = df[columns_order].to_numpy()

        data_seq = df[SEQ].tolist()
        data_seq_list = []
        for seq in data_seq:
            tmp_seq = seq.split(',')
            tmp_seq = list(map(int, tmp_seq))
            data_seq_list.append(tmp_seq)
        data_seq = np.asarray(data_seq_list)

        total_batches = int((len(df) + self.batch_size - 1) / self.batch_size)
        batches = []
        for n_batch in tqdm(range(total_batches), leave=False, ncols=100, mininterval=1, desc='Prepare Batches'):
            batch_start = n_batch * self.batch_size
            batch_end = min(len(df), batch_start + self.batch_size)

            real_batch_size = batch_end - batch_start

            batch = data[batch_start:batch_start + real_batch_size, :]
            batch_seq = data_seq[batch_start:batch_start + real_batch_size, :]

            inputs = np.asarray(batch)[:, 0:3].astype(int)
            features = np.asarray(batch)[:, 4:-1].astype(int)
            inputs = np.concatenate((inputs, features), axis=1)

            neg_samples = self._neg_samples_from_all(inputs, self.num_neg)

            uid = torch.from_numpy(inputs[:, 0])
            pos = torch.from_numpy(inputs[:, 1]).unsqueeze(1)
            features = torch.from_numpy(features)
            neg = torch.from_numpy(neg_samples)
            seq = torch.from_numpy(batch_seq)

            feed_dict = {'uid': uid, 'pos': pos, 'neg': neg, 'features': features, 'seq':seq}
            batches.append(feed_dict)
        return batches

    def collate_fn(self, batch):
        if self.stage == 'train':
            feed_dict = self._collate_train(batch)
        else:
            feed_dict = self._collate_vt(batch)
        return feed_dict

    def _collate_train(self, batch):
        inputs = np.asarray(batch)[:, 0:3].astype(int)
        features = np.asarray(batch)[:, 4:-1].astype(int)
        neg_samples = self._neg_sampler(inputs)

        batch_seq = (np.asarray(batch)[:, -1]).tolist()
        seq_list = []
        for seq in batch_seq:
            tmp_seq = seq.split(',')
            tmp_seq = list(map(int, tmp_seq))
            seq_list.append(tmp_seq)
        seq = np.asarray(seq_list)

        uid = torch.from_numpy(inputs[:, 0])
        pos = torch.from_numpy(inputs[:, 1]).unsqueeze(1)
        features = torch.from_numpy(features)
        neg = torch.from_numpy(neg_samples)
        seq = torch.from_numpy(seq)
        feed_dict = {'uid': uid, 'pos': pos, 'neg': neg, 'features': features, 'seq':seq}
        return feed_dict

    @staticmethod
    def _collate_vt(data):
        return data

    def _neg_sampler(self, batch):
        neg_items = np.random.randint(1, self.num_item, size=(len(batch), self.num_neg))
        for i, (user, _, _) in enumerate(batch):
            user_clicked_set = self.data_reader.all_user2items_dict[user]
            for j in range(self.num_neg):
                while neg_items[i][j] in user_clicked_set:
                    neg_items[i][j] = np.random.randint(1, self.num_item)
        return neg_items

    def _neg_samples_from_all(self, batch, num_neg=-1):
        neg_items = None
        for idx, data in enumerate(batch):
            user = data[0]
            neg_candidates = list(self.data_reader.item_ids_set - self.data_reader.all_user2items_dict[user])
            if num_neg != -1:
                if num_neg <= len(neg_candidates):
                    neg_candidates = np.random.choice(neg_candidates, num_neg, replace=False)
                else:
                    neg_candidates = np.random.choice(neg_candidates, len(neg_candidates), replace=False)
            neg_candidates = np.expand_dims(np.asarray(neg_candidates), axis=0)

            if neg_items is None:
                neg_items = neg_candidates
            else:
                neg_items = np.concatenate((neg_items, neg_candidates), axis=0)

        return neg_items


class DiscriminatorDataset:
    @staticmethod
    def parse_dp_args(parser):
        """
        DataProcessor related argument parser
        :param parser: argument parser
        :return: updated argument parser
        """
        parser.add_argument('--disc_batch_size', type=int, default=128,
                            help='discriminator train batch size')
        return parser

    def __init__(self, data_reader, stage, batch_size=1000):
        self.data_reader = data_reader
        self.stage = stage
        self.batch_size = batch_size
        self.data = self._get_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def _get_data(self):
        if self.stage == 'train':
            return self._get_train_data()
        if self.stage == 'valid':
            return self._get_valid_data()
        else:
            return self._get_test_data()

    def _get_train_data(self):
        data = self.data_reader.train_df.to_numpy()
        return data

    def _get_valid_data(self):
        data = self.data_reader.valid_df.to_numpy()
        return data

    def _get_test_data(self):
        data = self.data_reader.test_df.to_numpy()
        return data

    @staticmethod
    def collate_fn(data):
        feed_dict = dict()
        feed_dict['uid'] = torch.from_numpy(np.asarray(data)[:, 0].astype(int))
        feed_dict['features'] = torch.from_numpy(np.asarray(data)[:, 1:-1].astype(int))

        batch_seq = (np.asarray(data)[:, -1]).tolist()
        seq_list = []
        for seq in batch_seq:
            tmp_seq = seq.split(',')
            tmp_seq = list(map(int, tmp_seq))
            seq_list.append(tmp_seq)
        seq = np.asarray(seq_list)
        feed_dict['seq'] = torch.from_numpy(seq)

        return feed_dict
