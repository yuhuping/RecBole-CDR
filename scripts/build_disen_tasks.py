#!/usr/bin/env python
"""Convert DisenCDR domain-pair data into RecBole-CDR atomic datasets for the
DUCDR/GDUCDR model family, applying the same leakage-free holdout protocol as
scripts/build_validation_split.py.

For a directed task src->tgt:
  source folder = DisenCDR/dataset/{src}_{tgt}  (src-domain interactions)
  target folder = DisenCDR/dataset/{tgt}_{src}  (tgt-domain interactions)
Outputs (under RecBole-CDR/dataset):
  Disen_{src}_{tgt}_S/Disen_{src}_{tgt}_S.inter                 (source train)
  Disen_{src}_{tgt}_T/Disen_{src}_{tgt}_T.{train,valid,test}.inter (target holdout)
"""

import random
from collections import Counter, defaultdict
from pathlib import Path

DISEN_ROOT = Path("/home/yuhp/Rec/DisenCDR/dataset")
OUT_ROOT = Path("/home/yuhp/Rec/RecBole-CDR/dataset")
SEED = 2024
HEADER = ["user_id:token", "item_id:token", "rating:float", "timestamp:float"]

# directed tasks: (src, tgt). source folder = {src}_{tgt}, target folder = {tgt}_{src}
TASKS = [
    ("cloth", "electronic"),
    ("electronic", "cloth"),
    ("electronic", "phone"),
    ("phone", "electronic"),
    ("phone", "sport"),
    ("sport", "phone"),
]


# ---- Dinic max-flow + holdout (copied from scripts/build_validation_split.py) ----
class Dinic:
    def __init__(self, node_count):
        self.graph = [[] for _ in range(node_count)]

    def add_edge(self, source, target, capacity):
        forward = [target, capacity, None]
        backward = [source, 0, forward]
        forward[2] = backward
        self.graph[source].append(forward)
        self.graph[target].append(backward)
        return forward

    def max_flow(self, source, sink):
        total = 0
        while True:
            level = [-1] * len(self.graph)
            level[source] = 0
            queue = [source]
            for node in queue:
                for target, capacity, _ in self.graph[node]:
                    if capacity > 0 and level[target] < 0:
                        level[target] = level[node] + 1
                        queue.append(target)
            if level[sink] < 0:
                return total
            cursor = [0] * len(self.graph)

            def send(node, flow):
                if node == sink:
                    return flow
                while cursor[node] < len(self.graph[node]):
                    edge = self.graph[node][cursor[node]]
                    target, capacity, reverse = edge
                    if capacity > 0 and level[target] == level[node] + 1:
                        pushed = send(target, min(flow, capacity))
                        if pushed:
                            edge[1] -= pushed
                            reverse[1] += pushed
                            return pushed
                    cursor[node] += 1
                return 0

            while True:
                pushed = send(source, 10 ** 9)
                if not pushed:
                    break
                total += pushed


def hold_out_validation(train_rows, test_rows, seed):
    per_user = defaultdict(list)
    item_counts = Counter()
    for index, row in enumerate(train_rows):
        per_user[row[0]].append(index)
        item_counts[row[1]] += 1

    test_users = sorted({row[0] for row in test_rows})
    eligible_users = [
        user for user in test_users
        if any(item_counts[train_rows[index][1]] > 1 for index in per_user[user])
    ]
    dropped_users = set(test_users) - set(eligible_users)

    rng = random.Random(seed)
    rng.shuffle(eligible_users)
    eligible_items = sorted({
        train_rows[index][1]
        for user in eligible_users
        for index in per_user[user]
        if item_counts[train_rows[index][1]] > 1
    })
    rng.shuffle(eligible_items)

    source = 0
    user_offset = 1
    item_offset = user_offset + len(eligible_users)
    sink = item_offset + len(eligible_items)
    flow = Dinic(sink + 1)
    item_nodes = {item: item_offset + i for i, item in enumerate(eligible_items)}
    assignment_edges = defaultdict(list)

    for user_index, user in enumerate(eligible_users):
        user_node = user_offset + user_index
        flow.add_edge(source, user_node, 1)
        candidate_items = sorted({
            train_rows[index][1]
            for index in per_user[user]
            if item_counts[train_rows[index][1]] > 1
        })
        rng.shuffle(candidate_items)
        for item in candidate_items:
            edge = flow.add_edge(user_node, item_nodes[item], 1)
            assignment_edges[user].append((item, edge))

    for item, item_node in item_nodes.items():
        flow.add_edge(item_node, sink, item_counts[item] - 1)

    matched = flow.max_flow(source, sink)
    if matched != len(eligible_users):
        raise RuntimeError(f"matched {matched}/{len(eligible_users)} eligible users")

    selected_indices = set()
    valid_rows = []
    for user in eligible_users:
        selected_item = next(item for item, edge in assignment_edges[user] if edge[1] == 0)
        selected = next(i for i in per_user[user] if train_rows[i][1] == selected_item)
        selected_indices.add(selected)
        valid_rows.append(train_rows[selected])

    remaining_train = [r for i, r in enumerate(train_rows) if i not in selected_indices]
    filtered_test = [r for r in test_rows if r[0] not in dropped_users]
    return remaining_train, valid_rows, filtered_test, dropped_users


# ---- DisenCDR txt -> atomic rows ----
def read_disen(path, item_prefix, start_ts):
    rows = []
    ts = start_ts
    with open(path) as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            uid, iid = parts[0], parts[1]
            rows.append([uid, f"{item_prefix}{iid}", "1.0", f"{float(ts)}"])
            ts += 1
    return rows, ts


def write_inter(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        fh.write("\t".join(HEADER) + "\n")
        for r in rows:
            fh.write("\t".join(r) + "\n")


def audit(name, train, valid, test):
    tp = {(r[0], r[1]) for r in train}
    vp = {(r[0], r[1]) for r in valid}
    sp = {(r[0], r[1]) for r in test}
    train_items = {r[1] for r in train}
    assert {r[0] for r in valid} == {r[0] for r in test}, f"{name}: valid/test users differ"
    assert not (vp & sp), f"{name}: valid/test positive overlap"
    assert not (tp & vp) and not (tp & sp), f"{name}: train/heldout overlap"
    assert all(r[1] in train_items for r in valid + test), f"{name}: cold-start held-out item"


def main():
    for src, tgt in TASKS:
        src_folder = DISEN_ROOT / f"{src}_{tgt}"   # src-domain interactions
        tgt_folder = DISEN_ROOT / f"{tgt}_{src}"   # tgt-domain interactions
        src_name = f"Disen_{src}_{tgt}_S"
        tgt_name = f"Disen_{src}_{tgt}_T"

        # source domain: single .inter from source train
        src_rows, _ = read_disen(src_folder / "train.txt", f"{src}::", 0)
        write_inter(OUT_ROOT / src_name / f"{src_name}.inter", src_rows)

        # target domain: train + test -> holdout split
        tgt_train_rows, ts = read_disen(tgt_folder / "train.txt", f"{tgt}::", 0)
        tgt_test_rows, _ = read_disen(tgt_folder / "test.txt", f"{tgt}::", ts)
        remaining, valid, test, dropped = hold_out_validation(tgt_train_rows, tgt_test_rows, SEED)
        audit(tgt_name, remaining, valid, test)
        write_inter(OUT_ROOT / tgt_name / f"{tgt_name}.train.inter", remaining)
        write_inter(OUT_ROOT / tgt_name / f"{tgt_name}.valid.inter", valid)
        write_inter(OUT_ROOT / tgt_name / f"{tgt_name}.test.inter", test)

        print(f"{src}->{tgt}: src_inter={len(src_rows)} | "
              f"tgt train={len(remaining)} valid={len(valid)} test={len(test)} "
              f"dropped={len(dropped)}")


if __name__ == "__main__":
    main()
