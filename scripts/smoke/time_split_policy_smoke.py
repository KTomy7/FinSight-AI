from __future__ import annotations

import argparse

import pandas as pd

from finsight.infrastructure.features import TimeSplitPolicy


def _build_demo_frame() -> pd.DataFrame:
    # Five dates around cutoff with intentionally unsorted rows to validate deterministic output ordering.
    return pd.DataFrame(
        {
            "date": ["2024-01-05", "2024-01-03", "2024-01-02", "2024-01-04", "2024-01-01"],
            "ticker": ["AAA", "AAA", "BBB", "BBB", "AAA"],
            "x": [5.0, 3.0, 2.0, 4.0, 1.0],
            "target_ret_1d": [0.50, 0.30, 0.20, 0.40, 0.10],
        }
    )


def _to_iso_dates(series: pd.Series) -> list[str]:
    return series.apply(lambda value: value.date().isoformat()).tolist()


def _print_dates(label: str, frame: pd.DataFrame) -> None:
    dates = _to_iso_dates(frame["date"])
    print(f"{label} dates: {dates}")


def _run_case(df: pd.DataFrame, cutoff: str, inclusive_test: bool) -> None:
    policy = TimeSplitPolicy(cutoff_date=cutoff, inclusive_test=inclusive_test)
    train_df, test_df = policy.split_frame(df)

    print("\n=== Split case ===")
    print(f"cutoff={cutoff}, inclusive_test={inclusive_test}")
    print(f"rows: input={len(df)} train={len(train_df)} test={len(test_df)}")
    _print_dates("train", train_df)
    _print_dates("test", test_df)

    boundary = cutoff
    train_dates = set(_to_iso_dates(train_df["date"]))
    test_dates = set(_to_iso_dates(test_df["date"]))
    print(
        "boundary membership:",
        f"in_train={boundary in train_dates}",
        f"in_test={boundary in test_dates}",
    )

    if inclusive_test and boundary not in test_dates:
        raise RuntimeError("Expected boundary date to be included in test when inclusive_test=True.")
    if not inclusive_test and boundary not in train_dates:
        raise RuntimeError("Expected boundary date to be included in train when inclusive_test=False.")
    if not inclusive_test and boundary in test_dates:
        raise RuntimeError("Expected boundary date to be excluded from test when inclusive_test=False.")



def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke check for global TimeSplitPolicy behavior.")
    parser.add_argument(
        "--cutoff",
        type=str,
        default="2024-01-03",
        help='Cutoff date in ISO format "YYYY-MM-DD" (default: 2024-01-03).',
    )
    args = parser.parse_args()

    df = _build_demo_frame()
    print("=== Input frame ===")
    print(df)

    _run_case(df, cutoff=args.cutoff, inclusive_test=True)
    _run_case(df, cutoff=args.cutoff, inclusive_test=False)


if __name__ == "__main__":
    pd.set_option("display.width", 120)
    pd.set_option("display.max_columns", 20)
    main()



