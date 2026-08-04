#!/usr/bin/env python

import argparse
import gc
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recbole.utils import init_seed

from recbole_cdr.quick_start.quick_start import load_data_and_model
from recbole_cdr.utils import get_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--carcdr", required=True)
    parser.add_argument("--cut", required=True)
    parser.add_argument("--unicdr", required=True)
    parser.add_argument("--users-per-batch", type=int, default=64)
    parser.add_argument("--output-npz")
    return parser.parse_args()


def normalize_rows(scores):
    mean = scores.mean(axis=1, keepdims=True)
    std = scores.std(axis=1, keepdims=True)
    return (scores - mean) / np.maximum(std, 1e-8)


def rank_rows(scores, reciprocal=False):
    order = np.argsort(scores, axis=1)
    ranks = np.empty(scores.shape, dtype=np.float32)
    values = np.arange(scores.shape[1], dtype=np.float32)
    np.put_along_axis(ranks, order, values[None, :], axis=1)
    if reciprocal:
        ranks = 1.0 / (60.0 + scores.shape[1] - ranks)
    return normalize_rows(ranks)


def ranking_metrics(scores, labels):
    positive = np.take_along_axis(
        scores, labels.argmax(axis=1, keepdims=True), axis=1
    )
    ranks = 1 + (scores > positive).sum(axis=1)
    hit = (ranks <= 10).mean()
    ndcg = np.where(ranks <= 10, 1 / np.log2(ranks + 1), 0).mean()
    mrr = np.where(ranks <= 10, 1 / ranks, 0).mean()
    return float(hit), float(ndcg), float(mrr)


def collect_rows(model, loader, scorer):
    score_rows = []
    label_rows = []
    digest = hashlib.sha256()
    model.eval()
    with torch.no_grad():
        for interaction, row_idx, _, _ in loader:
            interaction = interaction.to(model.device)
            scores = scorer(model, interaction)
            labels = interaction["label"].bool()
            users = interaction[model.TARGET_USER_ID]
            items = interaction[model.TARGET_ITEM_ID]

            digest.update(users.cpu().numpy().tobytes())
            digest.update(items.cpu().numpy().tobytes())
            digest.update(labels.cpu().numpy().tobytes())
            num_rows = int(row_idx.max().item()) + 1
            counts = torch.bincount(row_idx, minlength=num_rows)
            if not torch.all(counts.eq(counts[0])):
                raise RuntimeError(
                    "Candidate counts differ within one evaluation batch."
                )
            width = int(counts[0].item())
            score_rows.append(
                scores.reshape(num_rows, width).cpu().numpy()
            )
            label_rows.append(
                labels.reshape(num_rows, width).cpu().numpy()
            )

    return (
        np.concatenate(score_rows).astype(np.float32),
        np.concatenate(label_rows),
        digest.hexdigest(),
    )


def collect_split(model, loader, scorer, seed, reproducibility):
    init_seed(seed, reproducibility)
    return collect_rows(model, loader, scorer)


def collect_model_scores(
    model, valid, test, scorer, seed, reproducibility
):
    model.set_phase("OVERLAP")
    valid_result = collect_split(
        model, valid, scorer, seed, reproducibility
    )
    test_result = collect_split(
        model, test, scorer, seed, reproducibility
    )
    return (
        valid_result[0],
        test_result[0],
        valid_result[1],
        test_result[1],
        valid_result[2],
        test_result[2],
    )


def load_expert(path, dataset, device):
    checkpoint = torch.load(path, map_location="cpu")
    config = checkpoint["config"]
    model = get_model(config["model"])(config, dataset).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.load_other_parameter(checkpoint.get("other_parameter"))
    return model


def release_cuda_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def set_eval_user_batch(loader, users_per_batch):
    if loader.pr != 0:
        raise RuntimeError("Evaluation loader has already been consumed.")
    loader.step = min(users_per_batch, len(loader.uid_list))
    loader.set_batch_size(loader.step * loader.times)


def direct_score(model, interaction):
    return model.predict(interaction)


def anchor_score(model, interaction):
    users = interaction[model.TARGET_USER_ID]
    items = interaction[model.TARGET_ITEM_ID]
    return model._anchor_score(users, items, 1)


def cmf_score(model, interaction):
    users = interaction[model.TARGET_USER_ID]
    items = interaction[model.TARGET_ITEM_ID]
    return (
        model.shared_user_embedding(users) * model.item_embedding(items)
    ).sum(dim=1)


def score_weights(experts, labels, weights):
    fused = experts["cd"].copy()
    for name, weight in weights.items():
        fused += weight * experts[name]
    return ranking_metrics(fused, labels)


def coordinate_search(experts, labels):
    weights = {"cmf": 0.0, "cut": 0.0, "unicdr": 0.0}
    steps = (0.25, 0.1, 0.05, 0.02)
    for step in steps:
        improved = True
        while improved:
            improved = False
            base_ndcg = score_weights(experts, labels, weights)[1]
            for name in weights:
                candidates = np.arange(
                    max(0.0, weights[name] - 5 * step),
                    weights[name] + 5 * step + step / 2,
                    step,
                )
                best_weight = weights[name]
                best_ndcg = base_ndcg
                for candidate in candidates:
                    trial = dict(weights)
                    trial[name] = float(candidate)
                    ndcg = score_weights(experts, labels, trial)[1]
                    if ndcg > best_ndcg:
                        best_ndcg = ndcg
                        best_weight = float(candidate)
                if best_weight != weights[name]:
                    weights[name] = best_weight
                    base_ndcg = best_ndcg
                    improved = True
    return weights


def main():
    args = parse_args()
    np.float = np.float64

    config, model, _, train, valid, test = load_data_and_model(
        args.carcdr
    )
    seed = config["seed"]
    reproducibility = config["reproducibility"]
    device = model.device
    set_eval_user_batch(valid, args.users_per_batch)
    set_eval_user_batch(test, args.users_per_batch)

    cd = collect_model_scores(
        model, valid, test, anchor_score, seed, reproducibility
    )
    cmf = collect_model_scores(
        model, valid, test, cmf_score, seed, reproducibility
    )
    model = None
    release_cuda_cache()

    model = load_expert(args.cut, train.dataset, device)
    cut = collect_model_scores(
        model, valid, test, direct_score, seed, reproducibility
    )
    model = None
    release_cuda_cache()

    model = load_expert(args.unicdr, train.dataset, device)
    unicdr = collect_model_scores(
        model, valid, test, direct_score, seed, reproducibility
    )
    model = None
    release_cuda_cache()

    references = (cd[4], cd[5])
    for name, result in (("cmf", cmf), ("cut", cut), ("unicdr", unicdr)):
        if (result[4], result[5]) != references:
            raise RuntimeError("%s candidate sequence does not match." % name)
        if not np.array_equal(result[2], cd[2]):
            raise RuntimeError("%s validation labels do not match." % name)
        if not np.array_equal(result[3], cd[3]):
            raise RuntimeError("%s test labels do not match." % name)

    raw_valid_experts = {
        "cd": cd[0],
        "cmf": cmf[0],
        "cut": cut[0],
        "unicdr": unicdr[0],
    }
    raw_test_experts = {
        "cd": cd[1],
        "cmf": cmf[1],
        "cut": cut[1],
        "unicdr": unicdr[1],
    }
    valid_experts = {
        name: normalize_rows(scores)
        for name, scores in raw_valid_experts.items()
    }
    test_experts = {
        name: normalize_rows(scores)
        for name, scores in raw_test_experts.items()
    }

    if args.output_npz:
        np.savez_compressed(
            args.output_npz,
            valid_labels=cd[2],
            test_labels=cd[3],
            **{
                "valid_" + name: scores
                for name, scores in raw_valid_experts.items()
            },
            **{
                "test_" + name: scores
                for name, scores in raw_test_experts.items()
            },
        )

    print("candidate_hash", references)
    print("cd_valid", ranking_metrics(cd[0], cd[2]))
    for name in ("cmf", "cut", "unicdr"):
        best = None
        for weight in np.arange(0.0, 1.51, 0.05):
            metrics = score_weights(
                valid_experts, cd[2], {name: float(weight)}
            )
            if best is None or metrics[1] > best[1][1]:
                best = (float(weight), metrics)
        print("single", name, best)

    weights = coordinate_search(valid_experts, cd[2])
    print("best_weights", weights)
    print("best_valid", score_weights(valid_experts, cd[2], weights))
    print("test", score_weights(test_experts, cd[3], weights))
    for removed in ("cmf", "cut", "unicdr"):
        ablated = dict(weights)
        ablated[removed] = 0.0
        print(
            "without_" + removed,
            score_weights(valid_experts, cd[2], ablated),
            score_weights(test_experts, cd[3], ablated),
        )

    raw_weights = coordinate_search(raw_valid_experts, cd[2])
    print("raw_weights", raw_weights)
    print(
        "raw_valid",
        score_weights(raw_valid_experts, cd[2], raw_weights),
    )
    print(
        "raw_test",
        score_weights(raw_test_experts, cd[3], raw_weights),
    )

    for transform_name, reciprocal in (
        ("rank", False),
        ("rrf", True),
    ):
        transformed_valid = {
            name: rank_rows(scores, reciprocal)
            for name, scores in valid_experts.items()
        }
        transformed_test = {
            name: rank_rows(scores, reciprocal)
            for name, scores in test_experts.items()
        }
        transformed_weights = coordinate_search(
            transformed_valid, cd[2]
        )
        print(transform_name + "_weights", transformed_weights)
        print(
            transform_name + "_valid",
            score_weights(
                transformed_valid, cd[2], transformed_weights
            ),
        )
        print(
            transform_name + "_test",
            score_weights(
                transformed_test, cd[3], transformed_weights
            ),
        )


if __name__ == "__main__":
    main()
