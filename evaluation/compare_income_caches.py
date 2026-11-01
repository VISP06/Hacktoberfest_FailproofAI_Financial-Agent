"""Compare two income-classification cache CSV files without making model calls."""

import argparse
import csv
from collections import Counter


def load_cache(path):
    with open(path, newline="", encoding="utf-8") as cache_file:
        rows = csv.DictReader(cache_file)
        required = {"user_id", "series_key", "class"}
        missing = required - set(rows.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")
        # If a cache contains repeated snapshots for one user/series, the last
        # row wins so that each series contributes once to the comparison.
        return {
            (row["user_id"], row["series_key"]): row["class"]
            for row in rows
        }


def compare_caches(cache_a, cache_b):
    shared = sorted(cache_a.keys() & cache_b.keys())
    matches = sum(cache_a[key] == cache_b[key] for key in shared)
    confusion = Counter((cache_a[key], cache_b[key]) for key in shared)
    disagreements = [
        (user_id, series_key, cache_a[key], cache_b[key])
        for key in shared
        if cache_a[key] != cache_b[key]
        for user_id, series_key in [key]
    ]
    return shared, matches, confusion, disagreements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache_a", help="first income cache CSV")
    parser.add_argument("cache_b", help="second income cache CSV")
    args = parser.parse_args()

    shared, matches, confusion, disagreements = compare_caches(
        load_cache(args.cache_a), load_cache(args.cache_b)
    )
    total = len(shared)
    agreement = matches / total if total else 0.0
    print(f"Shared series: {total}")
    print(f"Overall agreement rate: {agreement:.2%} ({matches}/{total})")

    classes = sorted({class_a for class_a, _ in confusion} | {class_b for _, class_b in confusion})
    print("\nConfusion table (rows=class A, columns=class B):")
    if classes:
        print("class A \\ class B," + ",".join(classes))
        for class_a in classes:
            print(class_a + "," + ",".join(str(confusion[(class_a, class_b)]) for class_b in classes))
    else:
        print("No shared series.")

    print("\nDisagreements (user_id, series_key, class A, class B):")
    if disagreements:
        for disagreement in disagreements:
            print(", ".join(disagreement))
    else:
        print("None")


if __name__ == "__main__":
    main()
