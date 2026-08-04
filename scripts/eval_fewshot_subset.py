import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from recbole.data.interaction import Interaction
from recbole.utils import init_logger, init_seed

from recbole_cdr.data import create_dataset, data_preparation
from recbole_cdr.utils import get_model, get_trainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a RecBole-CDR checkpoint on a few-shot user subset."
    )
    parser.add_argument("--model_file", type=Path, required=True, help="Checkpoint produced by RecBole-CDR.")
    parser.add_argument(
        "--fewshot_users",
        type=Path,
        required=True,
        help="CSV file like Intra/data/ready/*/fewshot_test_users.csv.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["valid", "test"],
        help="Target-domain split to evaluate after user filtering.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force the checkpoint to be loaded and evaluated on CPU.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="Optional path to dump the filtered-user metrics as JSON.",
    )
    return parser.parse_args()


def load_checkpoint_bundle(model_file, use_cpu):
    checkpoint = torch.load(model_file, map_location="cpu")
    config = checkpoint["config"]
    if use_cpu:
        config["use_gpu"] = False
        config["device"] = "cpu"
        config["gpu_id"] = ""

    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config["seed"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data.dataset).to(config["device"])
    model.load_state_dict(checkpoint["state_dict"])
    model.load_other_parameter(checkpoint.get("other_parameter"))

    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    trainer.saved_model_file = str(model_file)
    trainer.eval_collector.data_collect(train_data)

    return config, dataset, trainer, valid_data, test_data


def get_eval_data(split_name, valid_data, test_data):
    if split_name == "test":
        return test_data
    if isinstance(valid_data, tuple):
        return valid_data[1]
    return valid_data


def map_raw_users_to_inner_ids(dataset, raw_uids):
    target_dataset = dataset.target_domain_dataset
    uid_field = target_dataset.uid_field
    token2id = target_dataset.field2token_id[uid_field]

    mapped = set()
    missing = []
    for raw_uid in raw_uids:
        inner_id = token2id.get(raw_uid)
        if inner_id is None:
            missing.append(raw_uid)
        else:
            mapped.add(inner_id)
    return uid_field, sorted(mapped), missing


def filter_fullsort_eval_data(eval_data, kept_inner_ids):
    if not hasattr(eval_data, "uid_list"):
        raise TypeError("few-shot subset evaluation currently expects full-sort evaluation mode.")

    kept_set = set(kept_inner_ids)
    current_uid_list = eval_data.uid_list.tolist()
    filtered_uid_list = [uid for uid in current_uid_list if uid in kept_set]
    if not filtered_uid_list:
        raise ValueError("No few-shot users are present in the selected evaluation split.")

    filtered_uid_tensor = torch.tensor(filtered_uid_list, dtype=torch.int64)
    eval_data.uid_list = filtered_uid_tensor
    eval_data.user_df = eval_data.dataset.join(Interaction({eval_data.uid_field: filtered_uid_tensor}))
    eval_data.pr = 0

    return len(current_uid_list), len(filtered_uid_list)


def main():
    args = parse_args()
    if not args.model_file.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.model_file}")
    if not args.fewshot_users.is_file():
        raise FileNotFoundError(f"Few-shot user CSV not found: {args.fewshot_users}")

    fewshot_df = pd.read_csv(args.fewshot_users)
    if "uid" not in fewshot_df.columns:
        raise ValueError(f"`uid` column is required in {args.fewshot_users}")

    config, dataset, trainer, valid_data, test_data = load_checkpoint_bundle(args.model_file, args.cpu)
    eval_data = get_eval_data(args.split, valid_data, test_data)

    uid_field, kept_inner_ids, missing_raw_uids = map_raw_users_to_inner_ids(
        dataset,
        fewshot_df["uid"].astype(str).tolist(),
    )
    original_user_count, filtered_user_count = filter_fullsort_eval_data(eval_data, kept_inner_ids)

    result = trainer.evaluate(eval_data, load_best_model=False, show_progress=config["show_progress"])
    payload = {
        "checkpoint": str(args.model_file),
        "fewshot_users_file": str(args.fewshot_users),
        "split": args.split,
        "target_uid_field": uid_field,
        "fewshot_user_count": int(len(fewshot_df)),
        "mapped_user_count": int(len(kept_inner_ids)),
        "missing_user_count": int(len(missing_raw_uids)),
        "original_eval_user_count": int(original_user_count),
        "filtered_eval_user_count": int(filtered_user_count),
        "metrics": result,
    }

    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
