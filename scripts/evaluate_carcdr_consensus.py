#!/usr/bin/env python

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recbole.utils import init_seed

from recbole_cdr.quick_start.quick_start import load_data_and_model
from recbole_cdr.utils import get_trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--consensus-weight", type=float, default=0.36)
    return parser.parse_args()


def main():
    args = parse_args()
    np.float = np.float64
    config, model, _, _, valid, test = load_data_and_model(
        args.checkpoint
    )
    model.consensus_weight = args.consensus_weight
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
    print("consensus_weight", args.consensus_weight)
    print("valid", valid_result)
    print("test", test_result)


if __name__ == "__main__":
    main()
