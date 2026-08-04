# -*- coding: utf-8 -*-

import argparse
import os
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recbole_cdr.quick_start.quick_start import load_data_and_model
from recbole_cdr.utils import CrossDomainDataLoaderState


def _field_tensor(dataset, field):
    return dataset.inter_feat[field].long()


def _popularity(users_or_items, minlength):
    return torch.bincount(users_or_items.cpu(), minlength=minlength).float()


def _pop_bin(pop):
    if pop <= 1:
        return "pop=1"
    if pop <= 2:
        return "pop=2"
    if pop <= 5:
        return "pop=3-5"
    if pop <= 10:
        return "pop=6-10"
    return "pop>10"


def _metric_from_rank(rank):
    if rank <= 10:
        return 1.0, 1.0 / torch.log2(torch.tensor(rank + 1.0)).item(), 1.0 / rank
    return 0.0, 0.0, 1.0 / rank


def _eval_by_popularity(model, test_data, item_pop, device):
    bins = defaultdict(lambda: {"n": 0, "hr": 0.0, "ndcg": 0.0, "mrr": 0.0})
    model.eval()
    with torch.no_grad():
        for interaction, row_idx, _positive_u, positive_i in test_data:
            interaction = interaction.to(device)
            scores = model.predict(interaction).detach().cpu()
            labels = interaction["label"].detach().cpu()
            row_idx = row_idx.cpu()
            positive_i = positive_i.cpu()
            for row in torch.unique(row_idx, sorted=True).tolist():
                mask = row_idx == row
                group_scores = scores[mask]
                group_labels = labels[mask]
                positive_positions = torch.nonzero(group_labels > 0, as_tuple=False).view(-1)
                if positive_positions.numel() == 0:
                    continue
                positive_pos = int(positive_positions[0].item())
                positive_score = group_scores[positive_pos]
                rank = int((group_scores > positive_score).sum().item()) + 1
                item = int(positive_i[row].item())
                key = _pop_bin(float(item_pop[item].item()))
                hr, ndcg, mrr = _metric_from_rank(rank)
                bins[key]["n"] += 1
                bins[key]["hr"] += hr
                bins[key]["ndcg"] += ndcg
                bins[key]["mrr"] += mrr
    return bins


def _print_metric_bins(title, bins_a, bins_b):
    print(title)
    print("bin\tn\tDirect_HR\tDirect_NDCG\tDirect_MRR\tGDirect_HR\tGDirect_NDCG\tGDirect_MRR\tDelta_NDCG")
    for key in ["pop=1", "pop=2", "pop=3-5", "pop=6-10", "pop>10"]:
        a = bins_a.get(key, {"n": 0, "hr": 0.0, "ndcg": 0.0, "mrr": 0.0})
        b = bins_b.get(key, {"n": 0, "hr": 0.0, "ndcg": 0.0, "mrr": 0.0})
        n = max(a["n"], b["n"])
        if n == 0:
            print(f"{key}\t0\tNA\tNA\tNA\tNA\tNA\tNA\tNA")
            continue
        ah, an, am = a["hr"] / a["n"], a["ndcg"] / a["n"], a["mrr"] / a["n"]
        bh, bn, bm = b["hr"] / b["n"], b["ndcg"] / b["n"], b["mrr"] / b["n"]
        print(f"{key}\t{n}\t{ah:.4f}\t{an:.4f}\t{am:.4f}\t{bh:.4f}\t{bn:.4f}\t{bm:.4f}\t{bn - an:+.4f}")


def _user_centroids(model, users, items, num_users):
    device = model.device
    users = users.to(device)
    items = items.to(device)
    item_emb = model.item_emb(items).detach()
    sums = torch.zeros(num_users, item_emb.size(1), device=device)
    counts = torch.zeros(num_users, 1, device=device)
    sums.index_add_(0, users, item_emb)
    counts.index_add_(0, users, torch.ones(users.size(0), 1, device=device))
    centroids = sums / counts.clamp_min(1.0)
    return F.normalize(centroids, dim=1), counts.squeeze(1).detach().cpu()


def _collect_weight_records(model, train_data):
    train_data.set_mode(CrossDomainDataLoaderState.BOTH)
    rows = []
    center = F.normalize(model.target_direction_center.detach(), dim=0)
    with torch.no_grad():
        for interaction in train_data:
            users = interaction[model.TARGET_USER_ID].to(model.device)
            positives = interaction[model.TARGET_ITEM_ID].to(model.device)
            negatives = interaction[model.TARGET_NEG_ITEM_ID].to(model.device)
            pos_emb = model.item_emb(positives)
            neg_emb = model.item_emb(negatives)
            directions = F.normalize(pos_emb - neg_emb, dim=1)
            alignment = torch.matmul(directions, center)
            weight = torch.sigmoid((alignment - model.direction_tau) / model.direction_temperature)
            weight = model.direction_min_weight + (1.0 - model.direction_min_weight) * weight
            rows.append(
                (
                    users.cpu(),
                    positives.cpu(),
                    negatives.cpu(),
                    alignment.cpu(),
                    weight.cpu(),
                )
            )
    return tuple(torch.cat(parts, dim=0) for parts in zip(*rows))


def _print_weight_bins(title, users, positives, alignments, weights, item_pop):
    stats = defaultdict(lambda: {"n": 0, "alignment": 0.0, "weight": 0.0, "low04": 0, "bottom": 0})
    bottom_threshold = torch.quantile(weights, 0.20).item()
    for item, alignment, weight in zip(positives.tolist(), alignments.tolist(), weights.tolist()):
        key = _pop_bin(float(item_pop[item].item()))
        stats[key]["n"] += 1
        stats[key]["alignment"] += alignment
        stats[key]["weight"] += weight
        stats[key]["low04"] += int(weight <= 0.4)
        stats[key]["bottom"] += int(weight <= bottom_threshold)
    print(title)
    print(f"bottom20_weight_threshold\t{bottom_threshold:.4f}")
    print("bin\tn\tavg_alignment\tavg_weight\tweight<=0.4_pct\tbottom20_pct")
    for key in ["pop=1", "pop=2", "pop=3-5", "pop=6-10", "pop>10"]:
        s = stats.get(key)
        if not s or s["n"] == 0:
            print(f"{key}\t0\tNA\tNA\tNA\tNA")
            continue
        n = s["n"]
        print(
            f"{key}\t{n}\t{s['alignment']/n:.4f}\t{s['weight']/n:.4f}\t"
            f"{s['low04']/n:.4f}\t{s['bottom']/n:.4f}"
        )
    return bottom_threshold


def _print_origin(title, model, train_data, users, positives, weights, bottom_threshold, margin):
    source_users = _field_tensor(train_data.source_dataset, train_data.source_dataset.uid_field)
    source_items = _field_tensor(train_data.source_dataset, train_data.source_dataset.iid_field)
    target_users = _field_tensor(train_data.target_dataset, train_data.target_dataset.uid_field)
    target_items = _field_tensor(train_data.target_dataset, train_data.target_dataset.iid_field)
    source_centroid, source_counts = _user_centroids(model, source_users, source_items, model.total_num_users)
    target_centroid, target_counts = _user_centroids(model, target_users, target_items, model.total_num_users)

    mask = weights <= bottom_threshold
    low_users = users[mask].to(model.device)
    low_items = positives[mask].to(model.device)
    pos = F.normalize(model.item_emb(low_items).detach(), dim=1)
    source_sim = (pos * source_centroid[low_users]).sum(dim=1).cpu()
    target_sim = (pos * target_centroid[low_users]).sum(dim=1).cpu()
    diff = source_sim - target_sim
    source_aligned = diff > margin
    target_aligned = diff < -margin
    mixed = ~(source_aligned | target_aligned)

    print(title)
    print("low_weight_samples\t", int(mask.sum().item()))
    print("avg_source_profile_sim\t", f"{source_sim.mean().item():.4f}")
    print("avg_target_profile_sim\t", f"{target_sim.mean().item():.4f}")
    print("source_aligned_pct\t", f"{source_aligned.float().mean().item():.4f}")
    print("target_aligned_pct\t", f"{target_aligned.float().mean().item():.4f}")
    print("mixed_pct\t", f"{mixed.float().mean().item():.4f}")
    print("avg_source_history_len\t", f"{source_counts[users[mask]].float().mean().item():.4f}")
    print("avg_target_history_len\t", f"{target_counts[users[mask]].float().mean().item():.4f}")


def analyze_pair(name, direct_ckpt, gdirect_ckpt, margin):
    print("=" * 100)
    print(name)
    direct_config, direct_model, _dataset, _train, _valid, direct_test = load_data_and_model(direct_ckpt)
    g_config, g_model, _dataset, g_train, _valid, g_test = load_data_and_model(gdirect_ckpt)
    direct_model.eval()
    g_model.eval()

    target_items = _field_tensor(g_train.target_dataset, g_train.target_dataset.iid_field)
    item_pop = _popularity(target_items, g_model.total_num_items)

    users, positives, _negatives, alignments, weights = _collect_weight_records(g_model, g_train)
    bottom = _print_weight_bins("TRAIN_TARGET_WEIGHT_BY_ITEM_POP", users, positives, alignments, weights, item_pop)
    _print_origin("LOW_WEIGHT_ORIGIN_PROFILE", g_model, g_train, users, positives, weights, bottom, margin)

    direct_bins = _eval_by_popularity(direct_model, direct_test, item_pop, direct_model.device)
    g_bins = _eval_by_popularity(g_model, g_test, item_pop, g_model.device)
    _print_metric_bins("TEST_METRICS_BY_POSITIVE_ITEM_POP", direct_bins, g_bins)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--margin", type=float, default=0.02)
    parser.add_argument("--sport-cloth-direct", required=True)
    parser.add_argument("--sport-cloth-gdirect", required=True)
    parser.add_argument("--cloth-sport-direct", required=True)
    parser.add_argument("--cloth-sport-gdirect", required=True)
    args = parser.parse_args()

    analyze_pair(
        "Sport->Cloth",
        args.sport_cloth_direct,
        args.sport_cloth_gdirect,
        args.margin,
    )
    analyze_pair(
        "Cloth->Sport",
        args.cloth_sport_direct,
        args.cloth_sport_gdirect,
        args.margin,
    )


if __name__ == "__main__":
    main()
