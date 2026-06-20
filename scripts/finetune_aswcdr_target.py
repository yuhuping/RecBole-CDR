#!/usr/bin/env python

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recbole.utils import init_seed

from recbole_cdr.quick_start.quick_start import load_data_and_model
from recbole_cdr.utils import CrossDomainDataLoaderState, get_trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--mode", choices=("TARGET", "BOTH"), default="TARGET"
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def evaluate(config, trainer, data):
    init_seed(config["seed"], config["reproducibility"])
    return trainer.evaluate(
        data, load_best_model=False, show_progress=False
    )


def main():
    args = parse_args()
    np.float = np.float64
    config, model, _, train, valid, test = load_data_and_model(
        args.checkpoint
    )
    trainer = get_trainer(
        config["MODEL_TYPE"], config["model"]
    )(config, model)
    train.set_mode(getattr(CrossDomainDataLoaderState, args.mode))
    model.set_phase(args.mode)

    initial_valid = evaluate(config, trainer, valid)
    best_score = initial_valid["ndcg@10"]
    best_result = initial_valid
    best_epoch = -1
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(args.epochs):
        loss = trainer._train_epoch(
            train, epoch, show_progress=False
        )
        result = evaluate(config, trainer, valid)
        print("epoch", epoch, "loss", loss, "valid", result)
        if result["ndcg@10"] > best_score:
            best_score = result["ndcg@10"]
            best_result = result
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    torch.save(
        {
            "config": config,
            "state_dict": model.state_dict(),
            "other_parameter": model.other_parameter(),
        },
        args.output,
    )
    test_result = evaluate(config, trainer, test)
    print("initial_valid", initial_valid)
    print("best_epoch", best_epoch)
    print("best_valid", best_result)
    print("test", test_result)


if __name__ == "__main__":
    main()
