#!/usr/bin/env python

import argparse

import numpy as np

from search_carcdr_ensemble import (
    coordinate_search,
    normalize_rows,
    ranking_metrics,
    score_weights,
)


EXPERTS = ("cmf", "cut", "unicdr")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    return parser.parse_args()


def row_correlation(left, right):
    left = normalize_rows(left)
    right = normalize_rows(right)
    return (left * right).mean(axis=1)


def confidence_features(experts):
    anchor = experts["cd"]
    features = {}
    for name in EXPERTS:
        scores = experts[name]
        top_two = np.partition(scores, -2, axis=1)[:, -2:]
        gap = top_two[:, 1] - top_two[:, 0]
        gap = (gap - gap.mean()) / max(gap.std(), 1e-8)
        agreement = row_correlation(scores, anchor)
        agreement = (
            agreement - agreement.mean()
        ) / max(agreement.std(), 1e-8)
        features[name] = (gap, agreement)
    return features


def reliability_fusion(experts, weights, feature_set, scale, floor):
    fused = experts["cd"].copy()
    features = confidence_features(experts)
    for name in EXPERTS:
        gap, agreement = features[name]
        if feature_set == "gap":
            feature = gap
        elif feature_set == "agreement":
            feature = agreement
        else:
            feature = 0.5 * (gap + agreement)
        gate = 1 / (1 + np.exp(-scale * feature))
        gate = floor + (1 - floor) * gate
        gate = gate / gate.mean()
        fused += weights[name] * gate[:, None] * experts[name]
    return fused


def consensus_features(experts):
    stack = np.stack(
        [experts[name] for name in ("cd",) + EXPERTS], axis=0
    )
    positive = np.maximum(stack, 0)
    sorted_scores = np.sort(stack, axis=0)
    pair_products = []
    for left in range(stack.shape[0]):
        for right in range(left + 1, stack.shape[0]):
            pair_products.append(
                positive[left] * positive[right]
            )
    return {
        "agreement": -stack.std(axis=0),
        "positive_mean": positive.mean(axis=0),
        "second_best": sorted_scores[-2],
        "pair_support": np.mean(pair_products, axis=0),
    }


def main():
    args = parse_args()
    data = np.load(args.scores)
    valid = {
        name: normalize_rows(data["valid_" + name])
        for name in ("cd",) + EXPERTS
    }
    test = {
        name: normalize_rows(data["test_" + name])
        for name in ("cd",) + EXPERTS
    }
    valid_labels = data["valid_labels"]
    test_labels = data["test_labels"]
    weights = coordinate_search(valid, valid_labels)
    print("fixed_weights", weights)
    print("fixed_valid", score_weights(valid, valid_labels, weights))
    print("fixed_test", score_weights(test, test_labels, weights))

    best = None
    for feature_set in ("gap", "agreement", "joint"):
        for scale in (0.25, 0.5, 1.0, 2.0, 4.0):
            for floor in (0.0, 0.25, 0.5, 0.75):
                scores = reliability_fusion(
                    valid, weights, feature_set, scale, floor
                )
                metrics = ranking_metrics(scores, valid_labels)
                trial = (metrics[1], feature_set, scale, floor, metrics)
                if best is None or trial[0] > best[0]:
                    best = trial

    _, feature_set, scale, floor, valid_metrics = best
    test_scores = reliability_fusion(
        test, weights, feature_set, scale, floor
    )
    print("reliability", feature_set, scale, floor)
    print("reliability_valid", valid_metrics)
    print(
        "reliability_test",
        ranking_metrics(test_scores, test_labels),
    )

    fixed_valid = valid["cd"].copy()
    fixed_test = test["cd"].copy()
    for name in EXPERTS:
        fixed_valid += weights[name] * valid[name]
        fixed_test += weights[name] * test[name]
    valid_features = consensus_features(valid)
    test_features = consensus_features(test)
    best = None
    for name in valid_features:
        valid_feature = normalize_rows(valid_features[name])
        for weight in np.arange(-0.5, 0.5001, 0.02):
            metrics = ranking_metrics(
                fixed_valid + weight * valid_feature,
                valid_labels,
            )
            trial = (metrics[1], name, float(weight), metrics)
            if best is None or trial[0] > best[0]:
                best = trial

    _, feature_name, weight, valid_metrics = best
    consensus_test = fixed_test + weight * normalize_rows(
        test_features[feature_name]
    )
    print("consensus", feature_name, weight)
    print("consensus_valid", valid_metrics)
    print(
        "consensus_test",
        ranking_metrics(consensus_test, test_labels),
    )


if __name__ == "__main__":
    main()
