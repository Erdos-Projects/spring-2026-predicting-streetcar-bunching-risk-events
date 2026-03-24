from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


def make_task_name(b: int, s: int, n: int, m: int) -> str:
    return f"cond{int(b)}of{int(s)}_to_next{int(m)}_ge{int(n)}_binary"


def normalize_task_definitions(task_definitions: list[dict[str, Any]]) -> list[dict[str, int | str]]:
    out: list[dict[str, int | str]] = []
    for d in task_definitions:
        b = int(d["b"])
        s = int(d["s"])
        n = int(d["n"])
        m = int(d["m"])
        if s <= 0 or m <= 0 or b <= 0 or n <= 0:
            raise ValueError(f"Invalid task definition values: {d}")
        if b > s:
            raise ValueError(f"Task has b > s: {d}")
        if n > m:
            raise ValueError(f"Task has n > m: {d}")
        name = str(d.get("name", make_task_name(b, s, n, m)))
        out.append({"name": name, "b": b, "s": s, "n": n, "m": m})
    return out


def default_risk_task_definitions() -> list[dict[str, int | str]]:
    # Existing active task set, expressed in (b,s,n,m).
    return normalize_task_definitions(
        [
            # Relaxed tasks
            {"name": "observed1_next2_gap1_binary", "b": 1, "s": 1, "n": 1, "m": 2},
            {"name": "observed2_next2_gap1_binary", "b": 2, "s": 2, "n": 1, "m": 2},
            {"name": "observed2_next3_gap1_binary", "b": 2, "s": 2, "n": 2, "m": 3},
            {"name": "observed3_next2_gap1_binary", "b": 3, "s": 3, "n": 1, "m": 2},
            {"name": "observed3_next3_gap1_binary", "b": 3, "s": 3, "n": 2, "m": 3},
            {"name": "observed3_next4_gap1_binary", "b": 3, "s": 3, "n": 3, "m": 4},
            # Flexible conditional tasks
            {"name": "cond3of4_to_next4_ge2_binary", "b": 3, "s": 4, "n": 2, "m": 4},
            {"name": "cond3of5_to_next4_ge2_binary", "b": 3, "s": 5, "n": 2, "m": 4},
            {"name": "cond3of5_to_next5_ge3_binary", "b": 3, "s": 5, "n": 3, "m": 5},
            # Strict tasks
            {"name": "observed1_next1_binary", "b": 1, "s": 1, "n": 1, "m": 1},
            {"name": "observed2_next1_binary", "b": 2, "s": 2, "n": 1, "m": 1},
            {"name": "observed2_next2_binary", "b": 2, "s": 2, "n": 2, "m": 2},
            {"name": "observed3_next2_binary", "b": 3, "s": 3, "n": 2, "m": 2},
            {"name": "observed3_next3_binary", "b": 3, "s": 3, "n": 3, "m": 3},
        ]
    )


@dataclass(frozen=True)
class PipelineConfig:
    seed: int = 42

    pair_dt_ratio_to_sched: float = 0.5
    pair_dt_min_sec: float = 60.0
    pair_dt_max_sec: float = 600.0
    pair_dt_fallback_sec: float = 240.0
    pair_dedup_bin_min: int = 2
    episode_max_gap_min: int = 25

    valid_split: pd.Timestamp = pd.Timestamp("2025-10-01")
    test_split: pd.Timestamp = pd.Timestamp("2025-11-16")

    filter_2024_pre_july: bool = True
    min_ts: pd.Timestamp = pd.Timestamp("2024-07-01 00:00:00")

    bunch_ratio_threshold: float = 0.5
    bunch_col: str = "bunched_half_sched_i"

    strict_cv_n_folds: int = 4
    strict_cv_valid_days: int = 21
    strict_cv_buffer_days: int = 1
    strict_cv_min_train_days: int = 120
    strict_cv_min_train_rows: int = 400
    strict_cv_min_valid_rows: int = 120

    threshold_policy_global_default: str = "f2"
    threshold_fpr_cap_global_default: float = 0.40

    xgb_params: list[dict] = field(
        default_factory=lambda: [
            dict(
                max_depth=6,
                learning_rate=0.05,
                n_estimators=900,
                min_child_weight=3,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=3.0,
                reg_alpha=0.0,
            ),
            dict(
                max_depth=8,
                learning_rate=0.03,
                n_estimators=1400,
                min_child_weight=5,
                subsample=0.9,
                colsample_bytree=0.85,
                reg_lambda=4.0,
                reg_alpha=0.2,
            ),
        ]
    )

    risk_task_definitions: list[dict[str, int | str]] = field(
        default_factory=default_risk_task_definitions
    )

    events_primary_candidates: tuple[str, ...] = (
        "df2025_all.csv",
        "toy model/df2025_all.csv",
    )
    events_extra_candidates: tuple[tuple[str, ...], ...] = (
        ("506-2024-1.csv", "toy model/506-2024-1.csv"),
        ("df_2024-2", "toy model/df_2024-2"),
        ("df_2026", "toy model/df_2026"),
    )
    split_fallback_candidates: tuple[tuple[str, ...], tuple[str, ...]] = (
        ("all506_df-new-1.csv", "toy model/all506_df-new-1.csv"),
        ("all506_df-new-2.csv", "toy model/all506_df-new-2.csv"),
    )
    prefer_chunked_events: bool = True
    chunk_manifest_candidates: tuple[str, ...] = (
        "data_files/manifest.json",
        "../data_files/manifest.json",
    )


def find_first_existing(candidates: tuple[str, ...] | list[str]) -> str | None:
    for c in candidates:
        p = Path(c)
        if p.exists():
            return str(p)
    return None
