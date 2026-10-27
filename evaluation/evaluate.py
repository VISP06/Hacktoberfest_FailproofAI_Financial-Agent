import math
import os
import sys
import contextlib
import io

import pandas as pd
import failproofai_sdk as fp


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
VARIABLE_CATEGORY_SETS = [
    "protected_categories_only",
    "all_non_recurring_fixed_flexibility_spending",
]
VARIABLE_FACTORS = [0.5 + 0.125 * i for i in range(9)]
TRAILING_WINDOWS = [30, 60, 90]


def values_match(expected, actual, column):
    if column == "amount_safe_to_pay":
        return math.isclose(
            float(expected), float(actual), abs_tol=AMOUNT_TOLERANCE, rel_tol=0.0
        )

    expected = "" if pd.isna(expected) else str(expected)
    actual = "" if pd.isna(actual) else str(actual)
    return expected == actual


def score(expected, actual):
    """Smooth tuning objective, with exact-match coverage as a tie-break signal."""
    joined = expected.set_index("request_id").join(actual.set_index("request_id"), lsuffix="_expected", rsuffix="_actual")
    relative_errors = []
    exact_matches = 0
    exact_total = 0
    for _, row in joined.iterrows():
        expected_amount = float(row["amount_safe_to_pay_expected"])
        actual_amount = float(row["amount_safe_to_pay_actual"])
        relative_errors.append(abs(actual_amount - expected_amount) / max(abs(expected_amount), 1.0))
        for column in CHECK_COLUMNS:
            exact_total += 1
            if values_match(row[f"{column}_expected"], row[f"{column}_actual"], column):
                exact_matches += 1
    median_relative_error = float(pd.Series(relative_errors).median())
    exact_rate = exact_matches / exact_total if exact_total else 0.0
    # Both terms matter; the smooth amount component prevents categorical ties.
    return median_relative_error + (1.0 - exact_rate), median_relative_error, exact_matches, exact_total


def calibrate(sample_path):
    expected = pd.read_csv(sample_path)
    train = expected.iloc[:17].copy()
    held_out = expected.iloc[17:].copy()
    candidates = []
    fp.configure(environment="calibration")
    try:
        for category_set in VARIABLE_CATEGORY_SETS:
            for factor in VARIABLE_FACTORS:
                for window in TRAILING_WINDOWS:
                    options = {
                        "variable_category_set": category_set,
                        "variable_factor": factor,
                        "trailing_window_days": window,
                    }
                    with contextlib.redirect_stdout(io.StringIO()):
                        actual = run_pipeline(
                            requests_path=sample_path, output_file=os.devnull,
                            forecast_options={**options, "enable_spending_changes": False},
                        )
                    train_actual = actual[actual["request_id"].isin(train["request_id"])]
                    candidates.append((score(train, train_actual), options))
    finally:
        fp.configure(environment="dev")
    candidates.sort(key=lambda item: item[0][0:2] + (-item[0][2],))
    best_score, best_options = candidates[0]
    held_actual = run_pipeline(requests_path=sample_path, output_file=os.devnull,
                               forecast_options=best_options)
    held_score = score(held_out, held_actual[held_actual["request_id"].isin(held_out["request_id"])])

    # Print neighbouring parameter values, rather than only a point estimate, so
    # calibration brittleness is visible in the evaluation artifact.
    neighbours = [item for item in candidates if sum(
        item[1][key] != best_options[key]
        for key in best_options
    ) == 1]
    neighbour_scores = [item[0][0] for item in neighbours]
    stability = (min(neighbour_scores), max(neighbour_scores)) if neighbour_scores else (best_score[0], best_score[0])
    print("\nCalibration (17 train / 8 held-out):")
    print(f"  selected={best_options}")
    print(f"  train smooth_loss={best_score[0]:.6f}, median_relative_error={best_score[1]:.6f}, exact={best_score[2]}/{best_score[3]}")
    print(f"  held_out smooth_loss={held_score[0]:.6f}, median_relative_error={held_score[1]:.6f}, exact={held_score[2]}/{held_score[3]}")
    print(f"  neighbouring smooth_loss range={stability[0]:.6f}..{stability[1]:.6f}")
    return best_options, held_score


def main():
    sample_path = os.path.join(ROOT_DIR, "dataset", "sample_requests.csv")
    expected = pd.read_csv(sample_path)

    # Tune only on the first 17 rows and report on the held-out 8.
    best_options, _ = calibrate(sample_path)
    actual = run_pipeline(requests_path=sample_path, output_file=os.devnull,
                          forecast_options=best_options)
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
