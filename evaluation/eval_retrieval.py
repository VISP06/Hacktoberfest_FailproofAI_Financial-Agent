"""Offline retrieval relevance evaluation using request-linked messages."""

import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))

from retrieval import generic_request_query, retrieve_messages


def main():
    requests = pd.read_csv(os.path.join(ROOT, "dataset", "sample_requests.csv"), dtype={"user_id": str})
    messages = pd.read_csv(os.path.join(ROOT, "dataset", "messages.csv"), dtype=str).fillna("")
    relevant = messages.groupby("request_id")["message_id"].apply(set).to_dict()
    results = {}
    misses = []
    for row in requests.to_dict("records"):
        request_id = str(row["request_id"])
        query = generic_request_query(row)
        found = retrieve_messages(row["user_id"], row["request_date"], query, k=5)
        returned = [str(message["message_id"]) for message in found]
        target = relevant.get(request_id, set())
        results[request_id] = (returned, target)
        if not target.intersection(returned):
            misses.append(request_id)

    for k in (1, 3, 5):
        hits = sum(bool(target.intersection(ids[:k])) for ids, target in results.values())
        hit_rate = hits / len(results) if results else 0.0
        recall = sum(len(target.intersection(ids[:k])) / len(target) if target else 0.0
                     for ids, target in results.values()) / len(results) if results else 0.0
        print(f"hit-rate@{k}: {hit_rate:.2%}; recall@{k}: {recall:.2%}")
    print("Requests with no relevant result at @5: " + (", ".join(misses) if misses else "none"))


if __name__ == "__main__":
    main()
