#!/usr/bin/env python
"""Parse DisenCDR queue logs into per-pair result tables (image format)."""
import re
from pathlib import Path

LOGDIR = Path("/home/yuhp/Rec/RecBole-CDR/log/disen_queue")
MODELS = ["GDUCDRDirect", "DUCDRDirect", "DUCDRLite", "DUCDR", "GDUCDR"]
# (pair label, dir1 task, dir1 label, dir2 task, dir2 label)
PAIRS = [
    ("Cloth/Electronic", "cloth_electronic", "C->E", "electronic_cloth", "E->C"),
    ("Electronic/Phone", "electronic_phone", "E->P", "phone_electronic", "P->E"),
    ("Phone/Sport", "phone_sport", "P->S", "sport_phone", "S->P"),
]


def parse_test(task, model):
    log = LOGDIR / f"{task}_{model}.log"
    if not log.is_file():
        return None
    text = log.read_text(errors="ignore")
    matches = re.findall(r"test result: OrderedDict\((\[.*?\])\)", text)
    if not matches:
        return None
    d = dict(re.findall(r"'([a-z@0-9]+)',\s*([0-9.]+)", matches[-1]))
    return d


def fmt(d, k):
    return f"{float(d[k]):.4f}" if d and k in d else "  -   "


for label, t1, l1, t2, l2 in PAIRS:
    print(f"\n### {label}\n")
    print(f"| 模型 | {l1} HR | {l1} NDCG | {l1} MRR | {l2} HR | {l2} NDCG | {l2} MRR |")
    print("|---|---|---|---|---|---|---|")
    for m in MODELS:
        d1 = parse_test(t1, m)
        d2 = parse_test(t2, m)
        print(f"| {m} | {fmt(d1,'hit@10')} | {fmt(d1,'ndcg@10')} | {fmt(d1,'mrr@10')} "
              f"| {fmt(d2,'hit@10')} | {fmt(d2,'ndcg@10')} | {fmt(d2,'mrr@10')} |")
