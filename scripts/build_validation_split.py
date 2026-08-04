#!/usr/bin/env python

import argparse
import csv
from collections import Counter, defaultdict
import random
from pathlib import Path


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=2024)
    return parser.parse_args()


def read_rows(path):
    with path.open(newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        rows = list(reader)
    return header, rows


def write_rows(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def hold_out_validation(train_rows, test_rows, seed):
    per_user = defaultdict(list)
    item_counts = Counter()
    for index, row in enumerate(train_rows):
        per_user[row[0]].append(index)
        item_counts[row[1]] += 1

    test_users = sorted({row[0] for row in test_rows})
    eligible_users = [
        user for user in test_users
        if any(item_counts[train_rows[index][1]] > 1
               for index in per_user[user])
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
    item_nodes = {
        item: item_offset + index
        for index, item in enumerate(eligible_items)
    }
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

    matched_count = flow.max_flow(source, sink)
    if matched_count != len(eligible_users):
        raise RuntimeError(
            f"capacity assignment matched {matched_count}/"
            f"{len(eligible_users)} eligible users"
        )

    selected_indices = set()
    valid_rows = []
    for user in eligible_users:
        selected_item = next(
            item for item, edge in assignment_edges[user]
            if edge[1] == 0
        )
        selected = next(
            index for index in per_user[user]
            if train_rows[index][1] == selected_item
        )
        selected_indices.add(selected)
        valid_rows.append(train_rows[selected])

    remaining_train = [
        row for index, row in enumerate(train_rows)
        if index not in selected_indices
    ]
    filtered_test = [
        row for row in test_rows if row[0] not in dropped_users
    ]
    return remaining_train, valid_rows, filtered_test, dropped_users


def build_target(root, source_name, output_name, seed):
    source_dir = root / "dataset" / source_name
    output_dir = root / "dataset" / output_name
    source_train = source_dir / f"{source_name}.train.inter"
    source_test = source_dir / f"{source_name}.test.inter"
    output_train = output_dir / f"{output_name}.train.inter"
    output_valid = output_dir / f"{output_name}.valid.inter"
    output_test = output_dir / f"{output_name}.test.inter"

    output_dir.mkdir(parents=True, exist_ok=True)
    train_header, train_rows = read_rows(source_train)
    test_header, test_rows = read_rows(source_test)
    if train_header != test_header:
        raise RuntimeError(f"{source_name}: train/test headers differ")

    remaining_train, valid_rows, test_rows, dropped_users = hold_out_validation(
        train_rows, test_rows, seed
    )
    write_rows(output_train, train_header, remaining_train)
    write_rows(output_valid, train_header, valid_rows)
    write_rows(output_test, test_header, test_rows)

    train_pairs = {(row[0], row[1]) for row in remaining_train}
    valid_pairs = {(row[0], row[1]) for row in valid_rows}
    test_pairs = {(row[0], row[1]) for row in test_rows}
    valid_users = {row[0] for row in valid_rows}
    test_users = {row[0] for row in test_rows}
    train_items = {row[1] for row in remaining_train}
    if valid_users != test_users:
        raise RuntimeError(f"{output_name}: valid/test users differ")
    if valid_pairs & test_pairs:
        raise RuntimeError(f"{output_name}: valid/test positive overlap")
    if train_pairs & valid_pairs or train_pairs & test_pairs:
        raise RuntimeError(f"{output_name}: train/held-out positive overlap")
    if any(row[1] not in train_items for row in valid_rows + test_rows):
        raise RuntimeError(f"{output_name}: held-out cold-start item")

    print(
        output_name,
        f"train_rows={len(remaining_train)}",
        f"valid_rows={len(valid_rows)}",
        f"test_rows={len(test_rows)}",
        f"valid_users={len(valid_users)}",
        f"test_users={len(test_users)}",
        f"dropped_users={len(dropped_users)}",
    )


def main():
    args = parse_args()
    build_target(
        args.root,
        "AmazonClothUniCDR",
        "AmazonClothUniCDRHoldout",
        args.seed,
    )
    build_target(
        args.root,
        "AmazonSportUniCDRTarget",
        "AmazonSportUniCDRTargetHoldout",
        args.seed,
    )


if __name__ == "__main__":
    main()
