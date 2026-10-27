import math
import os
import sys

import pandas as pd


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "code"))

from main import run_pipeline


CHECK_COLUMNS = [
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "amount_safe_to_pay",
    "earliest_date_for_full_payment",
]
AMOUNT_TOLERANCE = 0.01


def values_match(expected, actual, column):
    if column == "amount_safe_to_pay":
        return math.isclose(
            float(expected), float(actual), abs_tol=AMOUNT_TOLERANCE, rel_tol=0.0
        )

    expected = "" if pd.isna(expected) else str(expected)
    actual = "" if pd.isna(actual) else str(actual)
    return expected == actual


def main():
    sample_path = os.path.join(ROOT_DIR, "dataset", "sample_requests.csv")
    expected = pd.read_csv(sample_path)

    # The pipeline still owns all decision logic; only its request input is overridden.
    actual = run_pipeline(requests_path=sample_path, output_file=os.devnull)
    actual_by_id = actual.set_index("request_id")

    print("\nPer-column accuracy:")
    mismatches = {}
    for column in CHECK_COLUMNS:
        column_mismatches = []
        for _, expected_row in expected.iterrows():
            request_id = str(expected_row["request_id"])
            actual_row = actual_by_id.loc[request_id]
            if not values_match(expected_row[column], actual_row[column], column):
                column_mismatches.append(request_id)
                mismatches.setdefault(request_id, []).append(column)

        accuracy = (len(expected) - len(column_mismatches)) / len(expected)
        print(f"  {column}: {accuracy:.2%}")

    if mismatches:
        print("\nMismatches:")
        for request_id, columns in mismatches.items():
            expected_row = expected[expected["request_id"].astype(str) == request_id].iloc[0]
            actual_row = actual_by_id.loc[request_id]
            print(f"  {request_id}:")
            for column in columns:
                print(
                    f"    {column}: expected={expected_row[column]!r}, "
                    f"actual={actual_row[column]!r}"
                )
    else:
        print("\nNo mismatches.")

    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
