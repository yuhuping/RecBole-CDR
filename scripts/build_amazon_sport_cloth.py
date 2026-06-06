import argparse
import gzip
import json
from pathlib import Path


SPORT_FILE = "reviews_Sports_and_Outdoors_5.json.gz"
CLOTH_FILE = "reviews_Clothing_Shoes_and_Jewelry_5.json.gz"
INTER_HEADER = "user_id:token\titem_id:token\trating:float\ttimestamp:float\n"


def parse_args():
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Convert raw Amazon Sport/Cloth reviews into RecBole-CDR .inter files."
    )
    parser.add_argument(
        "--raw_root",
        type=Path,
        default=repo_root / "Intra" / "data" / "raw",
        help="Directory that contains the raw Amazon json.gz files.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=repo_root / "RecBole-CDR" / "dataset",
        help="RecBole-CDR dataset root.",
    )
    parser.add_argument(
        "--sport_dataset",
        type=str,
        default="AmazonSport5",
        help="Dataset folder/name to use for the source or target sport domain.",
    )
    parser.add_argument(
        "--cloth_dataset",
        type=str,
        default="AmazonCloth5",
        help="Dataset folder/name to use for the source or target cloth domain.",
    )
    parser.add_argument(
        "--prefix_items",
        action="store_true",
        help="Add domain-specific prefixes to item ids so the two domains have no overlapped items.",
    )
    parser.add_argument(
        "--sport_item_prefix",
        type=str,
        default="sport::",
        help="Item-id prefix for the sport domain when --prefix_items is enabled.",
    )
    parser.add_argument(
        "--cloth_item_prefix",
        type=str,
        default="cloth::",
        help="Item-id prefix for the cloth domain when --prefix_items is enabled.",
    )
    return parser.parse_args()


def iter_records(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            uid = record.get("reviewerID")
            iid = record.get("asin")
            rating = record.get("overall")
            timestamp = record.get("unixReviewTime", 0)
            if uid is None or iid is None or rating is None:
                continue
            yield uid, iid, float(rating), float(timestamp)


def write_inter_file(raw_path, output_dir, dataset_name, item_prefix=""):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset_name}.inter"

    user_set = set()
    item_set = set()
    inter_count = 0

    with output_path.open("w", encoding="utf-8") as sink:
        sink.write(INTER_HEADER)
        for uid, iid, rating, timestamp in iter_records(raw_path):
            iid = f"{item_prefix}{iid}"
            sink.write(f"{uid}\t{iid}\t{rating}\t{timestamp}\n")
            user_set.add(uid)
            item_set.add(iid)
            inter_count += 1

    return {
        "path": str(output_path),
        "user_count": len(user_set),
        "item_count": len(item_set),
        "interaction_count": inter_count,
    }


def main():
    args = parse_args()

    sport_raw = args.raw_root / SPORT_FILE
    cloth_raw = args.raw_root / CLOTH_FILE
    if not sport_raw.is_file():
        raise FileNotFoundError(f"Sport raw file not found: {sport_raw}")
    if not cloth_raw.is_file():
        raise FileNotFoundError(f"Cloth raw file not found: {cloth_raw}")

    sport_stats = write_inter_file(
        sport_raw,
        args.output_root / args.sport_dataset,
        args.sport_dataset,
        item_prefix=args.sport_item_prefix if args.prefix_items else "",
    )
    cloth_stats = write_inter_file(
        cloth_raw,
        args.output_root / args.cloth_dataset,
        args.cloth_dataset,
        item_prefix=args.cloth_item_prefix if args.prefix_items else "",
    )

    print(
        "sport  "
        f"users={sport_stats['user_count']} items={sport_stats['item_count']} "
        f"inters={sport_stats['interaction_count']} file={sport_stats['path']}"
    )
    print(
        "cloth  "
        f"users={cloth_stats['user_count']} items={cloth_stats['item_count']} "
        f"inters={cloth_stats['interaction_count']} file={cloth_stats['path']}"
    )


if __name__ == "__main__":
    main()
