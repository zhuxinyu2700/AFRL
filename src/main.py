# coding=utf-8
import logging
import sys
import os
from utils.generic import *
from data_reader import RecDataReader
from runner import RecRunner
from datasets import RecDataset
from models.SASRec import SASRec
from models.BERT4rec import BERT4rec
from models.PMF import PMF
from models.MLP import MLP
from models.BaseRecModel import BaseRecModel
from models.AFRL_Generator import AFRL_Generator
from models.AFRL_CML import AFRL_CombineMLP
from models.Discriminators import ClassAttacker, AFRL_Discriminator
from data_reader import DiscriminatorDataReader
from datasets import DiscriminatorDataset
from torch.utils.data import DataLoader

#beta lager
def main():
    # init args
    init_parser = argparse.ArgumentParser(description='Model')
    init_parser.add_argument('--data_reader', type=str, default='RecDataReader',
                             help='Choose data_reader')
    init_parser.add_argument('--data_processor', type=str, default='RecDataset',
                             help='Choose data_processor')
    init_parser.add_argument('--model_name', type=str, default='PMF',
                             help='Choose model to run.')
    init_parser.add_argument('--runner', type=str, default='RecRunner',
                             help='Choose runner')
    init_parser.add_argument('--file_name', type=str, default='ml1M',
                             help='Choose runner')
    init_args, init_extras = init_parser.parse_known_args()

    # choose data_reader
    data_reader_name = eval(init_args.data_reader)

    # choose model
    model_name = eval(init_args.model_name)
    runner_name = eval(init_args.runner)

    # choose data_processor
    data_processor_name = eval(init_args.data_processor)

    # cmd line paras
    parser = argparse.ArgumentParser(description='')
    parser = parse_global_args(parser)
    parser = data_reader_name.parse_data_args(parser)
    parser = model_name.parse_model_args(parser, model_name=init_args.model_name)
    parser = runner_name.parse_runner_args(parser)
    parser = data_processor_name.parse_dp_args(parser)
    parser = DiscriminatorDataset.parse_dp_args(parser)

    args, extras = parser.parse_known_args()

    # log,model,result filename
    log_file_name = [init_args.model_name,
                     args.dataset]
    if args.no_filter:
        log_file_name.append("no_filter=" + str(args.no_filter))
    else:
        log_file_name.append("filter_mode=" + str(args.filter_mode))
        if args.filter_mode=='afrl':
            log_file_name.append("lambda=" + str(args.lambda_weight))
            log_file_name.append("beta=" + str(args.beta_weight))
    log_file_name = '__'.join(log_file_name).replace(' ', '_')
    if args.log_file == '../log/log.txt':
        if args.no_filter:
            args.log_file = '../log/%s.txt' % (log_file_name)
        else:
            args.log_file = '../log/afrl/%s.txt' % (log_file_name)
    if args.result_file == '../result/result.npy':
        args.result_file = '../result/%s.npy' % log_file_name
    if args.model_path == '../model/%s/%s.pt' % (init_args.model_name, init_args.model_name):
        args.model_path = '../model/%s/%s.pt' % (init_args.model_name, log_file_name)

    # logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=args.log_file, level=args.verbose)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    # convert the namespace into dictionary e.g. init_args.model_name -> {'model_name': BaseModel}
    logging.info(vars(init_args))
    logging.info(vars(args))

    logging.info('DataReader: ' + init_args.data_reader)
    logging.info('Model: ' + init_args.model_name)
    logging.info('Runner: ' + init_args.runner)
    logging.info('DataProcessor: ' + init_args.data_processor)

    # random seed
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)

    # cuda
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    logging.info("# cuda devices: %d" % torch.cuda.device_count())
    # create data_reader
    data_reader = data_reader_name(path=args.path, dataset_name=args.dataset, sep=args.sep, file_name=init_args.file_name)

    # create data processor
    data_processor_dict = {}
    for stage in ['train', 'valid', 'test']:
        if stage == 'train':
            if init_args.data_processor in ['RecDataset']:
                data_processor_dict[stage] = data_processor_name(
                    data_reader, stage, batch_size=args.batch_size, num_neg=args.train_num_neg)
            else:
                raise ValueError('Unknown DataProcessor')
        else:
            if init_args.data_processor in ['RecDataset']:
                data_processor_dict[stage] = data_processor_name(
                    data_reader, stage, batch_size=args.vt_batch_size, num_neg=args.vt_num_neg)
            else:
                raise ValueError('Unknown DataProcessor')

    # create model
    if args.no_filter:
        if init_args.model_name in ['PMF']:
            model = model_name(data_processor_dict, user_num=len(data_reader.user_ids_set),
                               item_num=len(data_reader.item_ids_set), u_vector_size=args.u_vector_size,
                               i_vector_size=args.i_vector_size, random_seed=args.random_seed, dropout=args.dropout,
                               model_path=args.model_path)
        elif init_args.model_name in ['MLP']:
            model = model_name(data_processor_dict, user_num=len(data_reader.user_ids_set),
                               item_num=len(data_reader.item_ids_set), u_vector_size=args.u_vector_size,
                               i_vector_size=args.i_vector_size, num_layers=args.num_layers,
                               random_seed=args.random_seed, dropout=args.dropout,
                               model_path=args.model_path)
        elif init_args.model_name in ['SASRec', 'BERT4rec']:
            model = model_name(data_processor_dict, user_num=len(data_reader.user_ids_set),
                               item_num=len(data_reader.item_ids_set),
                               num_heads=args.num_heads, num_blocks=args.num_blocks,
                               u_vector_size=args.u_vector_size, i_vector_size=args.i_vector_size, maxlen=args.maxlen,
                               random_seed=args.random_seed, dropout=args.dropout,
                               model_path=args.model_path)
        else:
            logging.error('Unknown Model: ' + init_args.model_name)
            return
    else:
        if init_args.model_name in ['PMF']:
            model = model_name(data_processor_dict, user_num=len(data_reader.user_ids_set),
                               item_num=len(data_reader.item_ids_set), u_vector_size=args.u_vector_size,
                               i_vector_size=args.i_vector_size, random_seed=args.random_seed, dropout=args.dropout,
                               model_path=args.base_model_path)
        elif init_args.model_name in ['MLP']:
            model = model_name(data_processor_dict, user_num=len(data_reader.user_ids_set),
                               item_num=len(data_reader.item_ids_set), u_vector_size=args.u_vector_size,
                               i_vector_size=args.i_vector_size, num_layers=args.num_layers,
                               random_seed=args.random_seed, dropout=args.dropout,
                               model_path=args.base_model_path)
        elif init_args.model_name in ['SASRec', 'BERT4rec']:
            model = model_name(data_processor_dict, user_num=len(data_reader.user_ids_set),
                               item_num=len(data_reader.item_ids_set),
                               num_heads=args.num_heads, num_blocks=args.num_blocks,
                               u_vector_size=args.u_vector_size, i_vector_size=args.i_vector_size, maxlen=args.maxlen,
                               random_seed=args.random_seed, dropout=args.dropout,
                               model_path=args.base_model_path)
        else:
            logging.error('Unknown Model: ' + init_args.model_name)
            return
    # init model paras
    model.apply(model.init_weights)

    # use gpu
    if torch.cuda.device_count() > 0:
        model = model.cuda()

    # create runner
    # batch_size is the training batch size, eval_batch_size is the batch size for evaluation
    if init_args.runner in ['BaseRunner']:
        runner = runner_name(
            optimizer=args.optimizer, learning_rate=args.lr,
            epoch=args.epoch, batch_size=args.batch_size, eval_batch_size=args.vt_batch_size,
            dropout=args.dropout, l2=args.l2,
            metrics=args.metric, check_epoch=args.check_epoch, early_stop=args.early_stop, model_name=init_args.model_name)
    elif init_args.runner in ['RecRunner']:
        runner = runner_name(
            optimizer=args.optimizer, learning_rate=args.lr,
            epoch=args.epoch, batch_size=args.batch_size, eval_batch_size=args.vt_batch_size,
            dropout=args.dropout, l2=args.l2,
            metrics=args.metric, check_epoch=args.check_epoch, early_stop=args.early_stop, num_worker=args.num_worker,
            no_filter=args.no_filter, filter_mode=args.filter_mode,
            beta_weight = args.beta_weight, lambda_weight = args.lambda_weight,
            d_steps=args.d_steps, disc_epoch=args.disc_epoch, model_name=init_args.model_name)
    else:
        logging.error('Unknown Runner: ' + init_args.runner)
        return

    disc_data_reader = DiscriminatorDataReader(path=args.path, dataset_name=args.dataset, sep=args.sep, file_name=init_args.file_name)

    # create data processor
    extra_data_processor_dict = {}
    for stage in ['train', 'valid', 'test']:
        extra_data_processor_dict[stage] = DiscriminatorDataset(disc_data_reader, stage, args.disc_batch_size)

    # create discriminators
    extra_fair_disc_dict = {}
    for feat_idx in disc_data_reader.feature_info:
        extra_fair_disc_dict[feat_idx] = \
            ClassAttacker(args.u_vector_size, disc_data_reader.feature_info[feat_idx],
                          model_dir_path=os.path.dirname(args.model_path), model_name='eval')
        extra_fair_disc_dict[feat_idx].apply(extra_fair_disc_dict[feat_idx].init_weights)
        if torch.cuda.device_count() > 0:
            extra_fair_disc_dict[feat_idx] = extra_fair_disc_dict[feat_idx].cuda()

    if args.filter_mode=='afrl':
        model.load_model()
        afrl_target_disc_dict = {}
        for feat_idx in data_reader.feature_info:
            afrl_target_disc_dict[feat_idx] = \
                AFRL_Discriminator(args.u_vector_size, data_reader.feature_info[feat_idx],
                              model_dir_path=os.path.dirname(args.model_path), target=True)
            afrl_target_disc_dict[feat_idx].apply(afrl_target_disc_dict[feat_idx].init_weights)
            if torch.cuda.device_count() > 0:
                afrl_target_disc_dict[feat_idx] = afrl_target_disc_dict[feat_idx].cuda()

        # create discriminators
        afrl_fair_disc_dict = {}
        for feat_idx in data_reader.feature_info:
            afrl_fair_disc_dict[feat_idx] = \
                AFRL_Discriminator(args.u_vector_size, data_reader.feature_info[feat_idx],
                              model_dir_path=os.path.dirname(args.model_path))
            afrl_fair_disc_dict[feat_idx].apply(afrl_fair_disc_dict[feat_idx].init_weights)
            if torch.cuda.device_count() > 0:
                afrl_fair_disc_dict[feat_idx] = afrl_fair_disc_dict[feat_idx].cuda()

        # create feature_generators
        afrl_feature_generator_dict = {}
        for feat_idx in data_reader.feature_info_uid:
            afrl_feature_generator_dict[feat_idx - 1] = \
                AFRL_Generator(args.u_vector_size, data_reader.feature_info_uid[feat_idx],
                          model_dir_path=os.path.dirname(args.model_path))
            afrl_feature_generator_dict[feat_idx - 1].apply(afrl_feature_generator_dict[feat_idx - 1].init_weights)
            if torch.cuda.device_count() > 0:
                afrl_feature_generator_dict[feat_idx - 1] = afrl_feature_generator_dict[feat_idx - 1].cuda()

        afrl_combine_mlp = AFRL_CombineMLP(args.u_vector_size, model_dir_path=os.path.dirname(args.model_path))
        afrl_combine_mlp.apply(afrl_combine_mlp.init_weights)
        if torch.cuda.device_count() > 0:
            afrl_combine_mlp = afrl_combine_mlp.cuda()


    if args.load > 0:
        model.load_model()
        if args.filter_mode=='afrl':
            model.load_model()
            for idx in afrl_feature_generator_dict:
                afrl_feature_generator_dict[idx].load_model()
            afrl_combine_mlp.load_model()
    if args.train > 0:
        if args.no_filter:
            runner.train(model, data_processor_dict,
                         skip_eval=args.skip_eval, fix_one=args.fix_one,
                         ex_dp_dict=extra_data_processor_dict, ex_fair_disc_dict=extra_fair_disc_dict,
                         lr_attack=args.lr_attack, l2_attack=args.l2_attack)
        else:
            runner.train(model, data_processor_dict,
                         skip_eval=args.skip_eval, fix_one=args.fix_one,
                         ex_dp_dict=extra_data_processor_dict, ex_fair_disc_dict=extra_fair_disc_dict,
                         lr_attack=args.lr_attack, l2_attack=args.l2_attack,
                         afrl_feature_generator_dict=afrl_feature_generator_dict, afrl_target_disc_dict=afrl_target_disc_dict,
                         afrl_fair_disc_dict=afrl_fair_disc_dict, afrl_combine_mlp=afrl_combine_mlp)


    # reset seed
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)

    if args.no_filter:
        model.load_model()
        model.freeze_model()
        fair_result_dict \
            = runner.train_discriminator(model, extra_data_processor_dict, extra_fair_disc_dict,
                                         args.lr_attack, args.l2_attack)
    else:
        model.load_model()
        for idx in afrl_feature_generator_dict:
            afrl_feature_generator_dict[idx].load_model()
        afrl_combine_mlp.load_model()
        fair_result_dict \
            = runner.train_discriminator(model, extra_data_processor_dict, extra_fair_disc_dict,
                                         args.lr_attack, args.l2_attack,
                                         afrl_feature_generator_dict=afrl_feature_generator_dict, afrl_combine_mlp=afrl_combine_mlp)


    test_data = DataLoader(data_processor_dict['test'], batch_size=None, num_workers=args.num_worker,
                           pin_memory=True, collate_fn=data_processor_dict['test'].collate_fn)

    test_result_dict = dict()
    if args.no_filter:
        test_result = runner.evaluate(model, test_data)
    else:
        args.fix_one = False
        if args.filter_mode == 'afrl':
            test_result, test_result_dict = runner.eval_multi_combination(model, test_data, args.fix_one,
                                                                          afrl_feature_generator_dict=afrl_feature_generator_dict,
                                                                          afrl_combine_mlp=afrl_combine_mlp)
    n_list = 0
    for key in fair_result_dict:
        auc_dict = fair_result_dict[key]
        key = key.split(',')
        key = list(map(int, key))
        auc = 0
        for i in range(len(key)):
            if key[i] == 1:
                auc += list(auc_dict.values())[i]
        if sum(key)!=0:
            auc = auc / sum(key)
        if args.no_filter:
            logging.info("mask: {} avg.auc,{}:{},{}".format(key, runner.metrics, auc, format_metric(test_result)))
        else:
            logging.info("mask: {} avg.auc,{}:{},{}".format(key, runner.metrics, auc,
                                                            format_metric(list(test_result_dict.values())[n_list])))
            n_list += 1

    return


if __name__ == '__main__':
    main()
