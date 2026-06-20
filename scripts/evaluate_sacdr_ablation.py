#!/usr/bin/env python

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recbole.utils import init_seed

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


@torch.no_grad()
def gate_statistics(model):
    users = torch.arange(
        1, model.overlapped_num_users, device=model.device
    )
    statistics = {}
    for domain, name in enumerate(("source", "target")):
        gate = model._user_state(users, domain)[3].squeeze(1)
        statistics[name] = {
            "mean": gate.mean().item(),
            "std": gate.std().item(),
            "min": gate.min().item(),
            "max": gate.max().item(),
        }
    return statistics


def main():
    args = parse_args()
    np.float = np.float64
    config, model, _, _, valid, test = load_data_and_model(
        args.checkpoint
    )
    model.set_phase("OVERLAP")

    full = evaluate(config, model, valid, test)
    gates = gate_statistics(model)

    original_transfer_scale = model.transfer_scale
    model.transfer_scale = 0
    local_only = evaluate(config, model, valid, test)

    original_local_scale = model.local_scale
    model.local_scale = 0
    shared_only = evaluate(config, model, valid, test)

    model.local_scale = original_local_scale
    model.transfer_scale = original_transfer_scale
    print("full", full)
    print("local_only", local_only)
    print("shared_only", shared_only)
    print("gate_statistics", gates)


if __name__ == "__main__":
    main()
