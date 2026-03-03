"""Walk-forward data splitting for robust backtesting."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config.constants import WALK_FORWARD_IN_SAMPLE_PCT, WALK_FORWARD_OUT_SAMPLE_PCT


@dataclass
class DataSplit:
    in_sample: pd.DataFrame
    out_of_sample: pd.DataFrame
    split_index: int
    fold: int


class DataSplitter:
    """Walk-forward and train/test data splitting."""

    def __init__(
        self,
        in_sample_pct: float = WALK_FORWARD_IN_SAMPLE_PCT,
        out_sample_pct: float = WALK_FORWARD_OUT_SAMPLE_PCT,
    ):
        self.in_sample_pct = in_sample_pct
        self.out_sample_pct = out_sample_pct

    def simple_split(self, df: pd.DataFrame) -> DataSplit:
        """Simple train/test split."""
        split_idx = int(len(df) * self.in_sample_pct)
        return DataSplit(
            in_sample=df.iloc[:split_idx].copy(),
            out_of_sample=df.iloc[split_idx:].copy(),
            split_index=split_idx,
            fold=0,
        )

    def walk_forward_splits(
        self,
        df: pd.DataFrame,
        n_folds: int = 5,
        min_in_sample: int = 500,
    ) -> list[DataSplit]:
        """Generate walk-forward splits.

        Each fold uses expanding in-sample window with fixed out-of-sample.
        """
        total_rows = len(df)
        if total_rows < min_in_sample:
            return [self.simple_split(df)]

        # Calculate fold sizes
        oos_size = int(total_rows * self.out_sample_pct / n_folds)
        if oos_size < 50:
            oos_size = 50

        splits = []
        for fold in range(n_folds):
            oos_end = total_rows - (n_folds - fold - 1) * oos_size
            oos_start = oos_end - oos_size
            is_end = oos_start

            if is_end < min_in_sample:
                continue

            splits.append(DataSplit(
                in_sample=df.iloc[:is_end].copy(),
                out_of_sample=df.iloc[oos_start:oos_end].copy(),
                split_index=is_end,
                fold=fold,
            ))

        if not splits:
            return [self.simple_split(df)]

        return splits

    def time_series_cv(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
        test_size: int | None = None,
    ) -> list[DataSplit]:
        """Time-series cross-validation (no data leakage)."""
        n = len(df)
        if test_size is None:
            test_size = n // (n_splits + 1)

        splits = []
        for i in range(n_splits):
            test_end = n - (n_splits - i - 1) * test_size
            test_start = test_end - test_size
            train_end = test_start

            if train_end < 100:
                continue

            splits.append(DataSplit(
                in_sample=df.iloc[:train_end].copy(),
                out_of_sample=df.iloc[test_start:test_end].copy(),
                split_index=train_end,
                fold=i,
            ))

        return splits
