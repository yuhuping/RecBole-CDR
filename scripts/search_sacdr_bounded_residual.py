#!/usr/bin/env python

import argparse
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recbole.utils import init_seed

from recbole_cdr.quick_start.quick_start import load_data_and_model
from recbole_cdr.utils import get_trainer


LOCAL_SCALES = (-0.05, -0.02, -0.01, -0.005, 0, 0.005, 0.01, 0.02, 0.05)
TRANSFER_SCALES = (-2, -1, -0.5, -0.2, -0.1, 0, 0.1, 0.2, 0.5, 1, 2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    return parser.parse_args()


def evaluate(config, trainer, data):
    init_seed(config["seed"], config["reproducibility"])
    return trainer.evaluate(
        data, load_best_model=False, show_progress=False
    )


def result_key(result):
    return result["ndcg@10"], result["hit@10"], result["mrr@10"]


@torch.no_grad()
def gate_means(model, original_state):
    means = []
    for domain in (0, 1):
        users = torch.arange(
            1, model.overlapped_num_users, device=model.device
        )
        means.append(original_state(users, domain)[3].mean().item())
    return means


def main():
    args = parse_args()
    np.float = np.float64
    config, model, _, _, valid, test = load_data_and_model(
        args.checkpoint
    )
    model.set_phase("OVERLAP")
    trainer = get_trainer(
        config["MODEL_TYPE"], config["model"]
    )(config, model)
    original_state = model._user_state
    fixed_gate = gate_means(model, original_state)
    search = {
        "local_scale": 0.0,
        "transfer_scale": 0.0,
        "gate_mode": "learned",
    }

    def bounded_state(self, users, domain):
        shared, local, cross, gate, _, _ = original_state(users, domain)
        anchor_norm = shared.norm(dim=1, keepdim=True).clamp_min(1e-8)
        local = F.normalize(local, dim=1) * anchor_norm
        cross = F.normalize(cross, dim=1) * anchor_norm
        if search["gate_mode"] == "fixed":
            gate = torch.full_like(gate, fixed_gate[domain])
        elif search["gate_mode"] == "open":
            gate = torch.ones_like(gate)
        local_condition = shared + search["local_scale"] * local
        full_condition = (
            local_condition
            + search["transfer_scale"] * gate * cross
        )
        return shared, local, cross, gate, local_condition, full_condition

    model._user_state = types.MethodType(bounded_state, model)

    local_results = []
    for scale in LOCAL_SCALES:
        search["local_scale"] = scale
        result = evaluate(config, trainer, valid)
        local_results.append((result_key(result), scale, result))
        print("local", scale, result)
    _, best_local, best_local_result = max(local_results)
    search["local_scale"] = best_local

    transfer_results = []
    for scale in TRANSFER_SCALES:
        search["transfer_scale"] = scale
        result = evaluate(config, trainer, valid)
        transfer_results.append((result_key(result), scale, result))
        print("transfer", scale, result)
    _, best_transfer, best_transfer_result = max(transfer_results)
    search["transfer_scale"] = best_transfer

    gate_results = []
    for mode in ("learned", "fixed", "open"):
        search["gate_mode"] = mode
        result = evaluate(config, trainer, valid)
        gate_results.append((result_key(result), mode, result))
        print("gate", mode, result)
    _, best_gate, best_gate_result = max(gate_results)
    search["gate_mode"] = best_gate

    test_result = evaluate(config, trainer, test)
    print("best_local", best_local, best_local_result)
    print("best_transfer", best_transfer, best_transfer_result)
    print("best_gate", best_gate, best_gate_result)
    print("fixed_gate", fixed_gate)
    print("selected_test", test_result)


if __name__ == "__main__":
    main()
