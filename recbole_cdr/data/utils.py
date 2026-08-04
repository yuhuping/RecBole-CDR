# @Time   : 2022/3/11
# @Author : Zihan Lin
# @Email  : zhlin@ruc.edu.cn
# UPDATE
# @Time   : 2022/4/9
# @Author : Gaowei Zhang
# @email  : 1462034631@qq.com

"""
recbole_cdr.data.utils
########################
"""

import importlib
import os
import pickle
from logging import getLogger

import numpy as np

from recbole.data.dataloader import NegSampleEvalDataLoader
from recbole.data.utils import load_split_dataloaders, save_split_dataloaders, create_samplers
from recbole.sampler import Sampler
from recbole.utils import set_color
from recbole.utils.argument_list import dataset_arguments

from recbole_cdr.data.dataloader import *
from recbole_cdr.sampler import CrossDomainSourceSampler
from recbole_cdr.utils import ModelType


class SeenItemSampler(Sampler):
    """Sample negatives only from items observed in the target training split."""

    def __init__(
        self,
        phases,
        datasets,
        distribution,
        candidate_item_ids,
        seed,
    ):
        self.candidate_item_ids = np.asarray(candidate_item_ids, dtype=np.int64)
        self.base_seed = seed
        self.sampling_seed = seed
        self.rng = np.random.RandomState(seed)
        super().__init__(phases, datasets, distribution)

    def set_phase(self, phase):
        sampler = super().set_phase(phase)
        phase_offset = {'train': 0, 'valid': 1, 'test': 2}[phase]
        sampler.sampling_seed = sampler.base_seed + phase_offset
        sampler.reset_sampling()
        return sampler

    def reset_sampling(self):
        self.rng = np.random.RandomState(self.sampling_seed)

    def _uni_sampling(self, sample_num):
        return self.rng.choice(
            self.candidate_item_ids,
            size=sample_num,
            replace=True,
        )

    def _get_candidates_list(self):
        return self.datasets[0].inter_feat[self.iid_field].numpy().tolist()


def _build_domain_eval_config(config, domain):
    eval_config = config.update(config[f'{domain}_domain'])
    eval_config['LABEL_FIELD'] = eval_config[f'{domain}_domain']['LABEL_FIELD']
    eval_config['NEG_PREFIX'] = eval_config[f'{domain}_domain']['NEG_PREFIX']
    return eval_config


def _filter_overlap_users(dataset, eval_dataset, phase):
    uid_field = dataset.target_domain_dataset.uid_field
    overlap_mask = eval_dataset.inter_feat[uid_field] < dataset.num_overlap_user
    filtered_dataset = eval_dataset.copy(eval_dataset.inter_feat[overlap_mask])
    getLogger().info(
        '%s evaluation restricted to %d overlapped users (%d interactions).',
        phase.capitalize(),
        len(filtered_dataset.inter_feat[uid_field].unique()),
        len(filtered_dataset),
    )
    return filtered_dataset


def _filter_seen_target_items(train_dataset, eval_dataset, phase):
    iid_field = train_dataset.iid_field
    seen_items = np.unique(train_dataset.inter_feat[iid_field].numpy())
    seen_mask = np.isin(eval_dataset.inter_feat[iid_field].numpy(), seen_items)
    filtered_dataset = eval_dataset.copy(eval_dataset.inter_feat[seen_mask])
    getLogger().info(
        '%s evaluation retained %d/%d interactions whose items occur in target training.',
        phase.capitalize(),
        len(filtered_dataset),
        len(eval_dataset),
    )
    return filtered_dataset


def _create_seen_item_samplers(config, built_datasets):
    phases = ['train', 'valid', 'test']
    train_dataset = built_datasets[0]
    iid_field = train_dataset.iid_field
    candidate_item_ids = np.unique(train_dataset.inter_feat[iid_field].numpy())
    candidate_item_ids = candidate_item_ids[candidate_item_ids != 0]

    train_args = config['train_neg_sample_args']
    eval_args = config['eval_neg_sample_args']
    eval_seed = getattr(config, 'final_config_dict', {}).get(
        'eval_candidate_seed', 2024
    )
    sampler = None
    train_sampler = valid_sampler = test_sampler = None

    if train_args['strategy'] != 'none':
        sampler = SeenItemSampler(
            phases,
            built_datasets,
            train_args['distribution'],
            candidate_item_ids,
            eval_seed,
        )
        train_sampler = sampler.set_phase('train')

    if eval_args['strategy'] != 'none':
        if sampler is None:
            sampler = SeenItemSampler(
                phases,
                built_datasets,
                eval_args['distribution'],
                candidate_item_ids,
                eval_seed,
            )
        else:
            sampler.set_distribution(eval_args['distribution'])
        valid_sampler = sampler.set_phase('valid')
        test_sampler = sampler.set_phase('test')

    return train_sampler, valid_sampler, test_sampler


def create_dataset(config):
    """Create cross domain dataset.
    If :attr:`config['dataset_save_path']` file exists and
    its :attr:`config` of dataset is equal to current :attr:`config` of dataset.
    It will return the saved dataset in :attr:`config['dataset_save_path']`.

    Args:
        config (CDRConfig): An instance object of Config, used to record parameter information.

    Returns:
        Dataset: Constructed dataset.
    """
    dataset_module = importlib.import_module('recbole_cdr.data.dataset')
    if hasattr(dataset_module, config['model'] + 'Dataset'):
        dataset_class = getattr(dataset_module, config['model'] + 'Dataset')
    else:
        model_type = config['MODEL_TYPE']
        type2class = {
            ModelType.CROSSDOMAIN: 'CrossDomainDataset'
        }
        dataset_class = getattr(dataset_module, type2class[model_type])

    default_file = os.path.join(config['checkpoint_dir'], f'{config["dataset"]}-{dataset_class.__name__}.pth')
    file = config['dataset_save_path'] or default_file
    if os.path.exists(file):
        with open(file, 'rb') as f:
            dataset = pickle.load(f)
        dataset_args_unchanged = True
        for arg in dataset_arguments + ['seed', 'repeatable']:
            if config[arg] != dataset.config[arg]:
                dataset_args_unchanged = False
                break
        if dataset_args_unchanged:
            logger = getLogger()
            logger.info(set_color('Load filtered dataset from', 'pink') + f': [{file}]')
            return dataset

    dataset = dataset_class(config)
    if config['save_dataset']:
        dataset.save()
    return dataset


def data_preparation(config, dataset):
    """Split the dataset by :attr:`config['eval_args']` and create training, validation and test dataloader.

    Note:
        If we can load split dataloaders by :meth:`load_split_dataloaders`, we will not create new split dataloaders.

    Args:
        config (CDRConfig): An instance object of Config, used to record parameter information.
        dataset (CrossDomainDataset): An instance object of Dataset, which contains all interaction records.

    Returns:
        tuple:
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    dataloaders = load_split_dataloaders(config)
    if dataloaders is not None:
        train_data, valid_data, test_data = dataloaders
    else:
        built_datasets = dataset.build()

        source_train_dataset, source_valid_dataset, target_train_dataset, \
            target_valid_dataset, target_test_dataset = built_datasets

        if config['eval_overlap_users_only']:
            target_valid_dataset = _filter_overlap_users(dataset, target_valid_dataset, 'validation')
            target_test_dataset = _filter_overlap_users(dataset, target_test_dataset, 'test')
            built_datasets[3] = target_valid_dataset
            built_datasets[4] = target_test_dataset

        if config['seen_target_items_only']:
            target_valid_dataset = _filter_seen_target_items(
                target_train_dataset, target_valid_dataset, 'validation'
            )
            target_test_dataset = _filter_seen_target_items(
                target_train_dataset, target_test_dataset, 'test'
            )
            built_datasets[3] = target_valid_dataset
            built_datasets[4] = target_test_dataset
            target_train_sampler, target_valid_sampler, target_test_sampler = \
                _create_seen_item_samplers(config, built_datasets[2:])
        else:
            target_train_sampler, target_valid_sampler, target_test_sampler = \
                create_samplers(config, dataset.target_domain_dataset, built_datasets[2:])

        if source_valid_dataset is not None:
            source_train_sampler, source_valid_sampler = create_source_samplers(config, dataset, built_datasets[:2])
            source_valid_data = get_dataloader(config, 'evaluation', 'source')(config, dataset, source_valid_dataset, source_valid_sampler, shuffle=False)
            target_valid_config = _build_domain_eval_config(config, 'target')
            target_valid_data = get_dataloader(config, 'evaluation', 'target')(target_valid_config, target_valid_dataset, target_valid_sampler, shuffle=False)

            valid_data = (source_valid_data, target_valid_data)
        else:
            source_train_sampler = CrossDomainSourceSampler('train', dataset, config['train_neg_sample_args']['distribution']).set_phase('train')
            target_valid_config = _build_domain_eval_config(config, 'target')
            valid_data = get_dataloader(config, 'evaluation', 'target')(target_valid_config, target_valid_dataset, target_valid_sampler, shuffle=False)

        train_data = get_dataloader(config, 'train', 'target')(config, dataset, source_train_dataset, source_train_sampler,
                                                           target_train_dataset, target_train_sampler, shuffle=True)

        target_test_config = _build_domain_eval_config(config, 'target')
        test_data = get_dataloader(config, 'evaluation', 'target')(target_test_config, target_test_dataset, target_test_sampler, shuffle=False)

        if config['save_dataloaders']:
            save_split_dataloaders(config, dataloaders=(train_data, valid_data, test_data))

    logger = getLogger()
    logger.info(
        set_color('[Training]: ', 'pink') + set_color('train_batch_size', 'cyan') + ' = ' +
        set_color(f'[{config["train_batch_size"]}]', 'yellow') + set_color(' negative sampling', 'cyan') + ': ' +
        set_color(f'[{config["neg_sampling"]}]', 'yellow')
    )
    logger.info(
        set_color('[Evaluation]: ', 'pink') + set_color('eval_batch_size', 'cyan') + ' = ' +
        set_color(f'[{config["eval_batch_size"]}]', 'yellow') + set_color(' eval_args', 'cyan') + ': ' +
        set_color(f'[{config["eval_args"]}]', 'yellow')
    )
    return train_data, valid_data, test_data


def get_dataloader(config, phase, domain='target'):
    """Return a dataloader class according to :attr:`config` and :attr:`phase`.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        phase (str): The stage of dataloader. It can only take two values: 'train' or 'evaluation'.
        domain (str): The domain of Evaldataloader. It can only take two values: 'source' or 'target'.

    Returns:
        type: The dataloader class that meets the requirements in :attr:`config` and :attr:`phase`.
    """
    model_type = config['MODEL_TYPE']
    if phase == 'train':
        if model_type == ModelType.CROSSDOMAIN:
            return CrossDomainDataloader
    else:
        if domain == 'source':
            return CrossDomainFullSortEvalDataLoader
        eval_strategy = config['eval_neg_sample_args']['strategy']
        if eval_strategy in {'none', 'by'}:
            if eval_strategy == 'by':
                return DeterministicNegSampleEvalDataLoader
            return NegSampleEvalDataLoader
        elif eval_strategy == 'full':
            return FullSortEvalDataLoader


def create_source_samplers(config, dataset, built_datasets):
    """Create sampler for training, validation and testing.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        dataset (Dataset): An instance object of Dataset, which contains all interaction records.
        built_datasets (list of Dataset): A list of split Dataset, which contains dataset for
            training, validation and testing.

    Returns:
        tuple:
            - train_sampler (AbstractSampler): The sampler for training.
            - valid_sampler (AbstractSampler): The sampler for validation.
    """
    phases = ['train', 'valid']
    train_neg_sample_args = config['train_neg_sample_args']
    eval_neg_sample_args = config['eval_neg_sample_args']

    sampler = CrossDomainSourceSampler(phases, dataset, built_datasets, train_neg_sample_args['distribution'])
    train_sampler = sampler.set_phase('train')

    sampler = CrossDomainSourceSampler(phases, dataset, built_datasets, eval_neg_sample_args['distribution'])
    valid_sampler = sampler.set_phase('valid')

    return train_sampler, valid_sampler
