# @Time   : 2022/3/10
# @Author : Zihan Lin
# @Email  : zhlin@ruc.edu.cn
# UPDATE
# @Time   : 2022/4/9
# @Author : Gaowei Zhang
# @email  : 1462034631@qq.com

"""
recbole_cdr.data.dataloader
################################################
"""

from logging import getLogger
import numpy as np
import torch

from recbole.data.interaction import Interaction
from recbole.data.dataloader.abstract_dataloader import AbstractDataLoader
from recbole.data.dataloader.general_dataloader import (
    FullSortEvalDataLoader,
    NegSampleEvalDataLoader,
    TrainDataLoader,
)

from recbole_cdr.utils import CrossDomainDataLoaderState


class DeterministicNegSampleEvalDataLoader(NegSampleEvalDataLoader):
    """Reset the evaluation sampler before every complete pass."""

    def __iter__(self):
        if hasattr(self.sampler, 'reset_sampling'):
            self.sampler.reset_sampling()
        return super().__iter__()


class OverlapDataloader(AbstractDataLoader):
    """:class:`OverlapDataloader` is a dataloader for training algorithms with only overlapped users or items.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader in source domain.
        shuffle (bool, optional): Whether the dataloader will be shuffled after a round. Defaults to ``False``.
    """
    def __init__(self, config, dataset, sampler=None, shuffle=False):
        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _init_batch_size_and_step(self):
        batch_size = self.config['overlap_batch_size']
        self.step = batch_size
        self.set_batch_size(batch_size)

    @property
    def pr_end(self):
        return len(self.dataset)

    def _shuffle(self):
        self.dataset.shuffle()

    def _next_batch_data(self):
        cur_data = self.dataset[self.pr:self.pr + self.step]
        self.pr += self.step
        return cur_data


class CrossDomainDataloader(AbstractDataLoader):
    """:class:`CrossDomainDataLoader` is a dataloader for training Cross domain algorithms.

    Args:
        config (Config): The config of dataloader.
        source_dataset (Dataset): The dataset of dataloader in source domain.
        source_sampler (Sampler): The sampler of dataloader in source domain.
        target_dataset (Dataset): The dataset of dataloader in target domain.
        target_sampler (Sampler): The sampler of dataloader in target domain.
        shuffle (bool, optional): Whether the dataloader will be shuffled after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, source_dataset, source_sampler, target_dataset, target_sampler,
                 shuffle=False):
        self.paired_domain_sampling = bool(config['paired_domain_sampling'])
        self.batches_per_epoch = config['batches_per_epoch']
        self.batches_yielded = 0
        config.update(config['source_domain'])
        config['LABEL_FIELD'] = source_dataset.label_field
        config['NEG_PREFIX'] = source_dataset.neg_prefix
        self.source_dataloader = TrainDataLoader(config, source_dataset, source_sampler, shuffle=shuffle)
        config.update(config['target_domain'])
        config['LABEL_FIELD'] = target_dataset.label_field
        config['NEG_PREFIX'] = target_dataset.neg_prefix
        self.target_dataloader = TrainDataLoader(config, target_dataset, target_sampler, shuffle=shuffle)
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset

        self.state = CrossDomainDataLoaderState.BOTH

        super().__init__(config, dataset, target_sampler, shuffle=shuffle)
        self.dataset.target_domain_dataset = target_dataset
        self.overlap_dataset = self.dataset.overlap_dataset
        self.overlap_dataloader = OverlapDataloader(config, self.overlap_dataset, sampler=None, shuffle=shuffle)
        if self.paired_domain_sampling:
            self._init_paired_domain_sampling(config)

    def _init_paired_domain_sampling(self, config):
        if config['train_neg_sample_args']['strategy'] != 'by' \
                or config['train_neg_sample_args']['by'] != 1:
            raise ValueError('Paired domain sampling requires one pointwise negative per positive.')

        source_uid = self.source_dataset.uid_field
        source_iid = self.source_dataset.iid_field
        target_uid = self.target_dataset.uid_field
        target_iid = self.target_dataset.iid_field

        source_users = self.source_dataset.inter_feat[source_uid].numpy()
        source_items = self.source_dataset.inter_feat[source_iid].numpy()
        target_users = self.target_dataset.inter_feat[target_uid].numpy()
        target_items = self.target_dataset.inter_feat[target_iid].numpy()

        self.paired_users = np.concatenate((source_users, target_users))
        self.paired_source_items = np.concatenate((
            source_items,
            np.full(len(target_items), -1, dtype=np.int64),
        ))
        self.paired_target_items = np.concatenate((
            np.full(len(source_items), -1, dtype=np.int64),
            target_items,
        ))

        user_num = self.dataset.num_total_user
        self.source_user_items = [[] for _ in range(user_num)]
        self.target_user_items = [[] for _ in range(user_num)]
        for user, item in zip(source_users, source_items):
            self.source_user_items[user].append(item)
        for user, item in zip(target_users, target_items):
            self.target_user_items[user].append(item)

        missing_source = [user for user in np.unique(self.paired_users)
                          if not self.source_user_items[user]]
        missing_target = [user for user in np.unique(self.paired_users)
                          if not self.target_user_items[user]]
        if missing_source or missing_target:
            raise ValueError('Paired domain sampling requires every training user in both domains.')

        self.paired_step = max(config['train_batch_size'] // 2, 1)
        self.paired_order = np.arange(len(self.paired_users))
        self.paired_pr = 0

    def _init_batch_size_and_step(self):
        pass

    def reinit_pr_after_map(self):
        self.source_dataloader.pr = 0
        self.target_dataloader.pr = 0
        if self.paired_domain_sampling:
            self.paired_pr = 0

    def update_config(self, config):
        self.source_dataloader.update_config(config)
        self.target_dataloader.update_config(config)
        self.overlap_dataset.update_config(config)

    def __iter__(self):
        if self.state == CrossDomainDataLoaderState.SOURCE:
            return self.source_dataloader.__iter__()
        elif self.state == CrossDomainDataLoaderState.TARGET:
            return self.target_dataloader.__iter__()
        elif self.state == CrossDomainDataLoaderState.BOTH:
            if self.paired_domain_sampling:
                self.paired_pr = 0
                if self.shuffle:
                    np.random.shuffle(self.paired_order)
                return self
            self.batches_yielded = 0
            self.source_dataloader.__iter__()
            self.target_dataloader.__iter__()
            return self
        elif self.state == CrossDomainDataLoaderState.OVERLAP:
            return self.overlap_dataloader.__iter__()

    def _shuffle(self):
        pass

    def __next__(self):
        if self.state == CrossDomainDataLoaderState.BOTH and self.paired_domain_sampling:
            if self.paired_pr >= len(self.paired_order):
                self.paired_pr = 0
                raise StopIteration()
            return self._next_paired_batch_data()
        if self.state == CrossDomainDataLoaderState.BOTH and self.batches_per_epoch:
            if self.batches_yielded >= self.batches_per_epoch:
                raise StopIteration()
            self.batches_yielded += 1
            return self._next_cycling_batch_data()
        if self.state == CrossDomainDataLoaderState.SOURCE and self.source_dataloader.pr >= self.source_dataloader.pr_end:
            self.target_dataloader.pr = 0
            self.source_dataloader.pr = 0
            raise StopIteration()
        if self.state == CrossDomainDataLoaderState.TARGET or self.state == CrossDomainDataLoaderState.BOTH:
            if self.target_dataloader.pr >= self.target_dataloader.pr_end:
                self.target_dataloader.pr = 0
                self.source_dataloader.pr = 0
                raise StopIteration()
        if self.state == CrossDomainDataLoaderState.OVERLAP and self.overlap_dataloader.pr >= self.overlap_dataloader.pr_end:
            self.overlap_dataloader.pr = 0
            raise StopIteration()
        return self._next_batch_data()

    def __len__(self):
        if self.state == CrossDomainDataLoaderState.SOURCE:
            return len(self.source_dataloader)
        elif self.state == CrossDomainDataLoaderState.TARGET:
            return len(self.target_dataloader)
        elif self.state == CrossDomainDataLoaderState.BOTH:
            if self.paired_domain_sampling:
                return int(np.ceil(len(self.paired_order) / self.paired_step))
            if self.batches_per_epoch:
                return self.batches_per_epoch
            return len(self.target_dataloader)
        elif self.state == CrossDomainDataLoaderState.OVERLAP:
            return len(self.overlap_dataloader)

    @property
    def pr_end(self):
        if self.state == CrossDomainDataLoaderState.SOURCE:
            return self.source_dataloader.pr_end
        elif self.state == CrossDomainDataLoaderState.OVERLAP:
            return self.overlap_dataloader.pr_end
        else:
            if self.paired_domain_sampling:
                return len(self.paired_order)
            return self.target_dataloader.pr_end

    def _next_batch_data(self):
        if self.state == CrossDomainDataLoaderState.SOURCE:
            return self.source_dataloader.__next__()
        elif self.state == CrossDomainDataLoaderState.TARGET:
            return self.target_dataloader.__next__()
        elif self.state == CrossDomainDataLoaderState.OVERLAP:
            return self.overlap_dataloader.__next__()
        else:
            try:
                source_data = self.source_dataloader.__next__()
            except StopIteration:
                source_data = self.source_dataloader.__next__()
            target_data = self.target_dataloader.__next__()
            target_data.update(source_data)
            return target_data

    def _next_cycling_batch_data(self):
        try:
            source_data = self.source_dataloader.__next__()
        except StopIteration:
            self.source_dataloader.__iter__()
            source_data = self.source_dataloader.__next__()
        try:
            target_data = self.target_dataloader.__next__()
        except StopIteration:
            self.target_dataloader.__iter__()
            target_data = self.target_dataloader.__next__()
        target_data.update(source_data)
        return target_data

    def _next_paired_batch_data(self):
        indices = self.paired_order[self.paired_pr:self.paired_pr + self.paired_step]
        self.paired_pr += self.paired_step

        users = self.paired_users[indices]
        source_pos = self.paired_source_items[indices].copy()
        target_pos = self.paired_target_items[indices].copy()

        for idx, user in enumerate(users):
            if source_pos[idx] < 0:
                source_pos[idx] = np.random.choice(self.source_user_items[user])
            if target_pos[idx] < 0:
                target_pos[idx] = np.random.choice(self.target_user_items[user])

        source_neg = self.source_dataloader.sampler.sample_by_user_ids(users, source_pos, 1)
        target_neg = self.target_dataloader.sampler.sample_by_user_ids(users, target_pos, 1)

        users = torch.as_tensor(users, dtype=torch.int64)
        source_pos = torch.as_tensor(source_pos, dtype=torch.int64)
        target_pos = torch.as_tensor(target_pos, dtype=torch.int64)
        labels = torch.cat((torch.ones(len(users)), torch.zeros(len(users))))

        return Interaction({
            self.source_dataset.uid_field: torch.cat((users, users)),
            self.source_dataset.iid_field: torch.cat((source_pos, source_neg)),
            self.source_dataset.label_field: labels,
            self.target_dataset.uid_field: torch.cat((users, users)),
            self.target_dataset.iid_field: torch.cat((target_pos, target_neg)),
            self.target_dataset.label_field: labels.clone(),
        })

    def set_mode(self, state):
        """Set the mode of :class:`CrossDomainDataloaderDataLoader`, it can be set to three states:

            - CrossDomainDataLoaderState.BOTH
            - CrossDomainDataLoaderState.SOURCE
            - CrossDomainDataLoaderState.TARGET

        The state of :class:`CrossDomainDataloaderDataLoader` would affect the result of _next_batch_data().

        Args:
            state (CrossDomainDataloaderState): the state of :class:`CrossDomainDataloaderDataLoader`.
        """
        if state not in set(CrossDomainDataLoaderState):
            raise NotImplementedError(f'Cross Domain data loader has no state named [{state}].')
        if self.source_dataloader.pr != 0 or self.target_dataloader.pr != 0:
            raise PermissionError('Cannot change dataloader\'s state within an epoch')
        self.state = state

    def get_model(self, model):
        """Let the dataloader get the model, used for dynamic sampling.
        """
        self.source_dataloader.get_model(model)
        self.target_dataloader.get_model(model)


class CrossDomainFullSortEvalDataLoader(FullSortEvalDataLoader):
    """:class:`CrossdomainFullSortEvalDataLoader` is a dataloader for full-sort evaluation. In order to speed up calculation,
    this dataloader would only return then user part of interactions, positive items and used items.
    It would not return negative items.

    Args:
        config (Config): The config of dataloader.
        dataset (CrossDomainDataset): The dataset from both domain.
        source_dataset(CrossDomainSingleDataset): The dataset that only from source domain.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, source_dataset, sampler, shuffle=False):
        self.uid_field = source_dataset.uid_field
        self.iid_field = source_dataset.iid_field
        self.is_sequential = False

        user_num = dataset.num_total_user
        self.overlap_item_num = dataset.num_overlap_item
        self.revoke_item_num = dataset.num_target_only_item
        self.uid_list = []
        self.uid2items_num = np.zeros(user_num, dtype=np.int64)
        self.uid2positive_item = np.array([None] * user_num)
        self.uid2history_item = np.array([None] * user_num)

        source_dataset.sort(by=self.uid_field, ascending=True)
        last_uid = None
        positive_item = set()
        uid2used_item = sampler.used_ids
        for uid, iid in zip(source_dataset.inter_feat[self.uid_field].numpy(),
                            source_dataset.inter_feat[self.iid_field].numpy()):
            if uid != last_uid:
                self._set_user_property(last_uid, uid2used_item[last_uid], positive_item)
                last_uid = uid
                self.uid_list.append(uid)
                positive_item = set()
            positive_item.add(iid)
        self._set_user_property(last_uid, uid2used_item[last_uid], positive_item)
        self.uid_list = torch.tensor(self.uid_list, dtype=torch.int64)
        self.user_df = source_dataset.join(Interaction({self.uid_field: self.uid_list}))

        self.config = config
        self.logger = getLogger()
        self.dataset = source_dataset
        self.sampler = sampler
        self.batch_size = self.step = self.model = None
        self.shuffle = shuffle
        self.pr = 0
        self._init_batch_size_and_step()

    def _set_user_property(self, uid, used_item, positive_item):
        if uid is None:
            return
        history_item = used_item - positive_item
        revoke_map_pos_item = [iid if iid < self.overlap_item_num else iid - self.revoke_item_num for iid in list(positive_item)]
        revoke_map_his_item = [iid if iid < self.overlap_item_num else iid - self.revoke_item_num for iid in list(history_item)]
        self.uid2positive_item[uid] = torch.tensor(revoke_map_pos_item, dtype=torch.int64)
        self.uid2items_num[uid] = len(positive_item)
        self.uid2history_item[uid] = torch.tensor(revoke_map_his_item, dtype=torch.int64)
