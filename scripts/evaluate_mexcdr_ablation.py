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
        states = model._expert_states(users, domain)
        local = states[1]
        transferred = model.transfer_projection[domain](
            model.local_user[1 - domain](users)
        )
        agreement = torch.nn.functional.cosine_similarity(
            local, transferred, dim=1, eps=1e-8
        ).unsqueeze(1)
        local_norm = local.norm(dim=1, keepdim=True) / (
            model.expert_size ** 0.5
        )
        transfer_norm = transferred.norm(dim=1, keepdim=True) / (
            model.expert_size ** 0.5
        )
        local_activity = model.user_activity[
            users, domain:domain + 1
        ]
        other_activity = model.user_activity[
            users, 1 - domain:2 - domain
        ]
        transfer_features = torch.cat((
            agreement,
            local_norm,
            transfer_norm,
            local_activity,
            other_activity,
        ), dim=1)
        transfer_gate = torch.sigmoid(
            model.transfer_gate[domain](transfer_features)
        ).squeeze(1)
        statistics[name] = {
            "transfer_mean": transfer_gate.mean().item(),
            "transfer_std": transfer_gate.std().item(),
        }
    return statistics


def set_weights(model, values):
    tensor = torch.tensor(
        values, dtype=model.expert_logits.dtype,
        device=model.expert_logits.device,
    )
    with torch.no_grad():
        model.expert_logits[1].copy_(tensor)


def main():
    args = parse_args()
    np.float = np.float64
    config, model, _, _, valid, test = load_data_and_model(
        args.checkpoint
    )
    original_logits = model.expert_logits[1].detach().clone()
    original_consensus = model.consensus_weight

    variants = {}
    names = ("shared", "local", "transfer", "history")
    for index, name in enumerate(names):
        logits = [-20.0] * 4
        logits[index] = 20.0
        set_weights(model, logits)
        model.consensus_weight = 0
        variants[name] = evaluate(config, model, valid, test)

    set_weights(model, [0, 0, 0, 0])
    model.consensus_weight = 0
    variants["equal"] = evaluate(config, model, valid, test)

    model.consensus_weight = original_consensus
    variants["equal_consensus"] = evaluate(
        config, model, valid, test
    )

    with torch.no_grad():
        model.expert_logits[1].copy_(original_logits)
    model.consensus_weight = 0
    variants["learned_no_consensus"] = evaluate(
        config, model, valid, test
    )

    model.consensus_weight = original_consensus
    variants["learned_consensus"] = evaluate(
        config, model, valid, test
    )

    print(
        "target_weights",
        torch.softmax(original_logits, dim=0).cpu().tolist(),
    )
    print("gate_statistics", gate_statistics(model))
    for name, result in variants.items():
        print(name, result)


if __name__ == "__main__":
    main()
