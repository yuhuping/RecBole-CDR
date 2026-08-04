# -*- coding: utf-8 -*-

import argparse
import os
import sys
from collections import Counter

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recbole_cdr.quick_start.quick_start import load_data_and_model
from recbole_cdr.utils import CrossDomainDataLoaderState


def _as_int_tensor(values):
    if torch.is_tensor(values):
        return values.long()
    return torch.as_tensor(values, dtype=torch.long)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-batches", type=int, default=200)
    parser.add_argument("--bins", type=float, nargs="+", default=[-0.1, 0.0, 0.1, 0.2, 0.3])
    args = parser.parse_args()

    config, model, dataset, train_data, _, _ = load_data_and_model(args.checkpoint)
    model.eval()
    train_data.set_mode(CrossDomainDataLoaderState.BOTH)

    target_dataset = train_data.target_dataset
    user_field = target_dataset.uid_field
    item_field = target_dataset.iid_field
    target_users = _as_int_tensor(target_dataset.inter_feat[user_field])
    target_items = _as_int_tensor(target_dataset.inter_feat[item_field])
    user_activity = torch.bincount(target_users, minlength=model.total_num_users).float()
    item_popularity = torch.bincount(target_items, minlength=model.total_num_items).float()

    directions = []
    positives_all = []
    users_all = []
    with torch.no_grad():
        for batch_idx, interaction in enumerate(train_data):
            positives = interaction[model.TARGET_ITEM_ID].to(model.device)
            negatives = interaction[model.TARGET_NEG_ITEM_ID].to(model.device)
            users = interaction[model.TARGET_USER_ID].to(model.device)
            pos_emb = model.item_emb(positives)
            neg_emb = model.item_emb(negatives)
            directions.append(F.normalize(pos_emb - neg_emb, dim=1).cpu())
            positives_all.append(positives.cpu())
            users_all.append(users.cpu())
            if batch_idx + 1 >= args.max_batches:
                break

    directions = torch.cat(directions, dim=0)
    positives = torch.cat(positives_all, dim=0)
    users = torch.cat(users_all, dim=0)
    center = F.normalize(directions.mean(dim=0), dim=0)
    alignment = torch.matmul(directions, center)

    edges = [-1.0] + sorted(args.bins) + [1.0]
    print("checkpoint:", args.checkpoint)
    print("model:", config["model"])
    print("samples:", int(alignment.numel()))
    print(
        "alignment mean/std/min/max:",
        f"{alignment.mean().item():.6f}",
        f"{alignment.std(unbiased=False).item():.6f}",
        f"{alignment.min().item():.6f}",
        f"{alignment.max().item():.6f}",
    )
    print("bin_low\tbin_high\tn\tpct\tavg_user_activity\tavg_pos_item_pop")
    total = alignment.numel()
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (alignment >= low) & (alignment < high)
        if high == edges[-1]:
            mask = (alignment >= low) & (alignment <= high)
        n = int(mask.sum().item())
        if n == 0:
            print(f"{low:.3f}\t{high:.3f}\t0\t0.0000\tNA\tNA")
            continue
        avg_user_activity = user_activity[users[mask]].mean().item()
        avg_item_pop = item_popularity[positives[mask]].mean().item()
        print(
            f"{low:.3f}\t{high:.3f}\t{n}\t{n / total:.4f}\t"
            f"{avg_user_activity:.4f}\t{avg_item_pop:.4f}"
        )

    low_threshold = sorted(args.bins)[1] if len(args.bins) > 1 else 0.0
    low_users = users[alignment < low_threshold].tolist()
    print("low_alignment_unique_users:", len(set(low_users)))
    print("low_alignment_top_users:", Counter(low_users).most_common(10))


if __name__ == "__main__":
    main()
