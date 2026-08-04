#!/usr/bin/env python

import argparse
import sys
import types
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recbole.utils import init_seed

from recbole_cdr.model.cross_domain_recommender.dpmcdr import (
    _build_training_history,
)
from recbole_cdr.quick_start.quick_start import load_data_and_model
from recbole_cdr.utils import get_trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    return parser.parse_args()


def evaluate(config, model, valid, test):
    trainer = get_trainer(
        config["MODEL_TYPE"], config["model"]
    )(config, model)
    init_seed(config["seed"], config["reproducibility"])
    valid_result = trainer.evaluate(
        valid, load_best_model=False, show_progress=False
    )
    init_seed(config["seed"], config["reproducibility"])
    test_result = trainer.evaluate(
        test, load_best_model=False, show_progress=False
    )
    return valid_result, test_result


def main():
    args = parse_args()
    np.float = np.float64
    config, model, dataset, _, valid, test = load_data_and_model(
        args.checkpoint
    )
    model.set_phase("OVERLAP")

    original = evaluate(config, model, valid, test)
    strict_history, strict_length = _build_training_history(
        dataset, model.history_length
    )
    model.history_item_id.copy_(strict_history.to(model.device))
    model.history_item_len.copy_(strict_length.to(model.device))
    strict = evaluate(config, model, valid, test)

    def direct_target_samples(self, users):
        return self._user_representation(users, 1)

    model._target_samples = types.MethodType(
        direct_target_samples, model
    )
    direct = evaluate(config, model, valid, test)

    print("original_history", original)
    print("strict_train_history", strict)
    print("direct_user_condition", direct)


if __name__ == "__main__":
    main()
