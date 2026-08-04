# @Time   : 2022/3/12
# @Author : zihan Lin
# @Email  : zhlin@ruc.edu.cn
# UPDATE
# @Time   : 2022/4/9
# @Author : Gaowei Zhang
# @email  : 1462034631@qq.com

r"""
recbole_cdr.trainer.trainer
################################
"""

import numpy as np
import torch
from recbole.trainer import Trainer
from recbole.utils import EvaluatorType, set_color
from recbole_cdr.utils import train_mode2state


class CrossDomainTrainer(Trainer):
    r"""Trainer for training cross-domain models. It contains four training mode: SOURCE, TARGET, BOTH, OVERLAP
    which can be set by the parameter of `train_epochs`
    """

    def __init__(self, config, model):
        super(CrossDomainTrainer, self).__init__(config, model)
        self.train_modes = config['train_modes']
        self.train_epochs = config['epoch_num']
        self.split_valid_flag = config['source_split']
        self.lr_decay = config['lr_decay']
        self.decay_epoch = config['decay_epoch']

    def _reinit(self, phase):
        """Reset the parameters when start a new training phase.
        """
        self.start_epoch = 0
        self.cur_step = 0
        self.best_valid_score = -np.inf if self.valid_metric_bigger else np.inf
        self.best_valid_result = None
        self.item_tensor = None
        self.tot_item_num = None
        self.train_loss_dict = dict()
        self.epochs = int(self.train_epochs[phase])
        self.eval_step = min(self.config['eval_step'], self.epochs)
        self.dev_score_history = [0]
        self.current_lr = self.learning_rate

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        if hasattr(self.model, 'set_train_epoch'):
            self.model.set_train_epoch(epoch_idx)
        return super()._train_epoch(train_data, epoch_idx, loss_func, show_progress)

    def _decay_lr_on_plateau(self, epoch_idx, valid_score):
        r"""Decay the learning rate once validation stops improving, mirroring the
        plateau-triggered schedule used by the original DisenCDR/DRLCDR training scripts
        (``current_lr *= lr_decay`` once ``len(history) > decay_epoch`` epochs have passed
        and the latest score does not exceed the previous one). Constant Adam LR otherwise
        leaves these GNN-based models oscillating around an early peak instead of converging.
        """
        if self.lr_decay and self.decay_epoch and \
                epoch_idx > self.decay_epoch and \
                valid_score <= self.dev_score_history[-1]:
            self.current_lr *= self.lr_decay
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.current_lr
            self.logger.info(set_color('Validation plateaued, decay learning rate to ', 'blue')
                             + '%.6g' % self.current_lr)
        self.dev_score_history.append(valid_score)

    def _wrap_callback_fn(self, callback_fn):
        if not (self.lr_decay and self.decay_epoch):
            return callback_fn

        def wrapped(epoch_idx, valid_score):
            self._decay_lr_on_plateau(epoch_idx, valid_score)
            if callback_fn:
                callback_fn(epoch_idx, valid_score)

        return wrapped

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        r"""Train the model based on the train data and the valid data.

            Args:
                train_data (DataLoader): the train data
                valid_data (DataLoader, optional): the valid data, default: None.
                                                    If it's None, the early_stopping is invalid.
                verbose (bool, optional): whether to write training and evaluation information to logger, default: True
                saved (bool, optional): whether to save the model parameters, default: True
                show_progress (bool): Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
                callback_fn (callable): Optional callback function executed at end of epoch.
                                        Includes (epoch_idx, valid_score) input arguments.

            Returns:
                    (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        for phase in range(len(self.train_modes)):
            self._reinit(phase)
            scheme = self.train_modes[phase]
            self.logger.info("Start training with {} mode".format(scheme))
            state = train_mode2state[scheme]
            train_data.set_mode(state)
            self.model.set_phase(scheme)
            wrapped_callback_fn = self._wrap_callback_fn(callback_fn)
            if self.split_valid_flag and valid_data is not None:
                source_valid_data, target_valid_data = valid_data
                if scheme == 'SOURCE':
                    super().fit(train_data, source_valid_data, verbose, saved, show_progress, wrapped_callback_fn)
                else:
                    super().fit(train_data, target_valid_data, verbose, saved, show_progress, wrapped_callback_fn)
            else:
                super().fit(train_data, valid_data, verbose, saved, show_progress, wrapped_callback_fn)

        self.model.set_phase('OVERLAP')
        return self.best_valid_score, self.best_valid_result

    def _neg_sample_batch_eval(self, batched_data):
        interaction, row_idx, positive_u, positive_i = batched_data
        batch_size = interaction.length
        if batch_size <= self.test_batch_size:
            origin_scores = self.model.predict(interaction.to(self.device))
        else:
            origin_scores = self._spilt_predict(interaction, batch_size)

        if self.config['eval_type'] == EvaluatorType.VALUE:
            return interaction, origin_scores, positive_u, positive_i
        if self.config['eval_type'] == EvaluatorType.RANKING:
            origin_scores = origin_scores.view(-1)
            col_idx = interaction[self.model.TARGET_ITEM_ID]
            batch_user_num = positive_u[-1] + 1
            scores = torch.full((batch_user_num, self.tot_item_num), -np.inf, device=self.device)
            scores[row_idx, col_idx] = origin_scores
            return interaction, scores, positive_u, positive_i
        raise ValueError(f"Unsupported eval_type [{self.config['eval_type']}]")


class DCDCSRTrainer(Trainer):
    r"""Trainer for training DCDCSR models."""

    def __init__(self, config, model):
        super(DCDCSRTrainer, self).__init__(config, model)
        self.train_modes = config['train_modes']
        self.train_epochs = config['epoch_num']
        self.split_valid_flag = config['source_split']

    def _reinit(self, phase):
        """Reset the parameters when start a new training phase.
        """
        self.start_epoch = 0
        self.cur_step = 0
        self.best_valid_score = -np.inf if self.valid_metric_bigger else np.inf
        self.best_valid_result = None
        self.item_tensor = None
        self.tot_item_num = None
        self.train_loss_dict = dict()
        self.epochs = int(self.train_epochs[phase])
        self.eval_step = min(self.config['eval_step'], self.epochs)

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        r"""Train the model based on the train data and the valid data.

            Args:
                train_data (DataLoader): the train data
                valid_data (DataLoader, optional): the valid data, default: None.
                                                    If it's None, the early_stopping is invalid.
                verbose (bool, optional): whether to write training and evaluation information to logger, default: True
                saved (bool, optional): whether to save the model parameters, default: True
                show_progress (bool): Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
                callback_fn (callable): Optional callback function executed at end of epoch.
                                        Includes (epoch_idx, valid_score) input arguments.

            Returns:
                    (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        for phase in range(len(self.train_modes)):
            self._reinit(phase)
            scheme = self.train_modes[phase]
            self.logger.info("Start training with {} mode".format(scheme))
            state = train_mode2state[scheme]
            train_data.set_mode(state)
            self.model.set_phase(scheme)
            if scheme == 'BOTH':
                super().fit(train_data, None, verbose, saved, show_progress, callback_fn)
            else:
                if self.split_valid_flag and valid_data is not None:
                    source_valid_data, target_valid_data = valid_data
                    if scheme == 'SOURCE':
                        super().fit(train_data, source_valid_data, verbose, saved, show_progress, callback_fn)
                    else:
                        super().fit(train_data, target_valid_data, verbose, saved, show_progress, callback_fn)
                else:
                    super().fit(train_data, valid_data, verbose, saved, show_progress, callback_fn)

        self.model.set_phase('OVERLAP')
        return self.best_valid_score, self.best_valid_result

    def _neg_sample_batch_eval(self, batched_data):
        interaction, row_idx, positive_u, positive_i = batched_data
        batch_size = interaction.length
        if batch_size <= self.test_batch_size:
            origin_scores = self.model.predict(interaction.to(self.device))
        else:
            origin_scores = self._spilt_predict(interaction, batch_size)

        if self.config['eval_type'] == EvaluatorType.VALUE:
            return interaction, origin_scores, positive_u, positive_i
        if self.config['eval_type'] == EvaluatorType.RANKING:
            origin_scores = origin_scores.view(-1)
            col_idx = interaction[self.model.TARGET_ITEM_ID]
            batch_user_num = positive_u[-1] + 1
            scores = torch.full((batch_user_num, self.tot_item_num), -np.inf, device=self.device)
            scores[row_idx, col_idx] = origin_scores
            return interaction, scores, positive_u, positive_i
        raise ValueError(f"Unsupported eval_type [{self.config['eval_type']}]")


class CUT_24Trainer(CrossDomainTrainer):
    r"""CUT_24 re-builds the optimizer after each phase switch because parameter freezing changes."""

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        for phase in range(len(self.train_modes)):
            self._reinit(phase)
            scheme = self.train_modes[phase]
            self.logger.info("Start training with {} mode".format(scheme))
            state = train_mode2state[scheme]
            train_data.set_mode(state)
            self.model.set_phase(scheme)
            if getattr(self.model, 'skip_target', 0) and scheme == 'TARGET':
                continue
            self.optimizer = self._build_optimizer(params=filter(lambda p: p.requires_grad, self.model.parameters()))
            if self.split_valid_flag and valid_data is not None:
                source_valid_data, target_valid_data = valid_data
                if scheme == 'SOURCE':
                    super(CrossDomainTrainer, self).fit(train_data, source_valid_data, verbose, saved, show_progress, callback_fn)
                else:
                    super(CrossDomainTrainer, self).fit(train_data, target_valid_data, verbose, saved, show_progress, callback_fn)
            else:
                super(CrossDomainTrainer, self).fit(train_data, valid_data, verbose, saved, show_progress, callback_fn)

        self.model.set_phase('OVERLAP')
        return self.best_valid_score, self.best_valid_result
