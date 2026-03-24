from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import PipelineConfig

# Nanoseconds in a day
NS_DAY = 24 * 3600 * 1_000_000_000


@dataclass(frozen=True)
class FeatureBundle:
    incident_df: pd.DataFrame
    confirmed_df: pd.DataFrame
    incident_X: pd.DataFrame
    confirmed_X: pd.DataFrame
    incident_split: dict[str, pd.Series]
    confirmed_split: dict[str, pd.Series]
    base_num_feats: list[str]
    cat_feats: list[str]
    feature_cols: list[str]
    cb_cat_idx: list[int]


def add_prior_count_within_days(
    df_: pd.DataFrame,
    key_col: str,
    ts_col: str,
    days: int,
    out_col: str,
) -> None:
    out = np.zeros(len(df_), dtype="float32")
    win_ns = int(days) * NS_DAY
    groups = df_.groupby(key_col, sort=False).groups
    for _, idx in groups.items():
        idx = np.asarray(idx, dtype=int)
        t_ns_raw = df_.loc[idx, ts_col].values.astype("datetime64[ns]").astype("int64")
        order = np.argsort(t_ns_raw, kind="mergesort")
        idx_sorted = idx[order]
        t_ns = t_ns_raw[order]
        left = np.searchsorted(t_ns, t_ns - win_ns, side="left")
        counts = (np.arange(len(idx_sorted)) - left).astype("float32")
        out[idx_sorted] = counts
    df_[out_col] = out


def get_base_num_feats(include_external: bool = True) -> tuple[list[str], list[str]]:
    base_num_feats = [
        "start_ord",
        "hour",
        "dow",
        "month",
        "is_weekend",
        "start_pair_delay",
        "start_pair_sch",
        "start_dt_sec",
        "start_gap_ratio",
        "start_pair_delay_abs_diff",
        "start_pair_sch_abs_diff",
        "start_pair_delay_min",
        "start_pair_delay_max",
        "start_pair_sch_min",
        "start_pair_sch_max",
        "start_pair_veh_lag1_sch_mean",
        "start_pair_veh_lag2_sch_mean",
        "start_pair_veh_lag3_sch_mean",
        "start_pair_veh_lag123_sch_mean_mean",
        "start_pair_veh_lag1_bunched_mean",
        "start_pair_veh_lag2_bunched_mean",
        "start_pair_veh_lag3_bunched_mean",
        "start_pair_veh_lag123_bunched_sum_mean",
        "start_pair_veh_lag1_sch_abs_diff",
        "start_pair_veh_lag2_sch_abs_diff",
        "start_pair_veh_lag3_sch_abs_diff",
        "start_pair_veh_run_pos_mean",
        "start_pair_veh_elapsed_min_mean",
        "start_pair_veh_run_progress_mean",
        "start_pair_remaining_stops_to_terminal_mean",
        "start_pair_sch_trend_13",
        "start_pair_sch_trend_12",
        "start_pair_bunched_trend_13",
        "start_pair_bunched_trend_12",
        "start_fl_ratio_to_sched",
        "start_ll1_ratio_to_sched",
        "start_pair_fl_ll1_ratio_diff",
        "start_fl_gap_sec",
        "start_ll1_gap_sec",
        "start_fl_sched_headway_sec",
        "start_ll1_sched_headway_sec",
        "start_pair_sched_headway_sec_mean",
        "start_pair_sched_headway_min_mean",
        "start_pair_sched_headway_min_sq",
        "start_pair_gap_sec_mean",
        "start_pair_gap_minus_sched_sec",
        "start_stop_recent_bunch_n_30m",
        "start_stop_recent_bunch_rate_30m",
        "start_stop_recent_bunch_n_60m",
        "start_stop_recent_bunch_rate_60m",
        "pair_prev_1d_count",
        "veh_lo_prev_1d_count",
        "veh_hi_prev_1d_count",
    ]
    external_num_feats = [
        "stop_nearest_pedx_m",
        "stop_pedx_within_250m",
        "stop_pedx_within_500m",
        "stop_nearest_sig_m",
        "stop_sig_within_250m",
        "stop_sig_within_500m",
        "cum_stop_pedx_within_250m_to_ord",
        "cum_stop_pedx_within_500m_to_ord",
        "cum_stop_sig_within_250m_to_ord",
        "cum_stop_sig_within_500m_to_ord",
        "permit_active_city",
        "permit_active_route_hint",
        "permit_new_city_7d",
        "permit_new_route_7d",
        "permit_route_share",
        "temp_c_hr",
        "precip_hr_mm",
        "wind_kmh_hr",
        "humidity_pct",
        "is_raining",
        "is_snowing",
        "is_freezing",
    ]
    if include_external:
        for c in external_num_feats:
            if c not in base_num_feats:
                base_num_feats.append(c)
    return base_num_feats, external_num_feats


def _make_X(df: pd.DataFrame, feature_cols: list[str], base_num_feats: list[str], cat_feats: list[str]) -> pd.DataFrame:
    X = df[feature_cols].copy()
    for c in base_num_feats:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype("float32")
    for c in cat_feats:
        X[c] = X[c].astype("string").fillna("NA").astype(str)
    return X


def build_feature_bundle(
    incident_df: pd.DataFrame,
    confirmed_df: pd.DataFrame,
    cfg: PipelineConfig,
    include_external: bool = True,
) -> FeatureBundle:
    incident_df = incident_df.copy()
    confirmed_df = confirmed_df.copy()

    for df in [incident_df, confirmed_df]:
        add_prior_count_within_days(df, "pair_key", "start_ts", 1, "pair_prev_1d_count")
        add_prior_count_within_days(df, "veh_lo", "start_ts", 1, "veh_lo_prev_1d_count")
        add_prior_count_within_days(df, "veh_hi", "start_ts", 1, "veh_hi_prev_1d_count")

    base_num_feats, _ = get_base_num_feats(include_external=include_external)
    cat_feats = ["start_stopID", "bound"]
    feature_cols = base_num_feats + cat_feats

    for df in [incident_df, confirmed_df]:
        for c in base_num_feats:
            if c not in df.columns:
                df[c] = 0.0

    incident_X = _make_X(incident_df, feature_cols, base_num_feats, cat_feats)
    confirmed_X = _make_X(confirmed_df, feature_cols, base_num_feats, cat_feats)

    incident_split = {
        "train": incident_df["start_ts"] < cfg.valid_split,
        "valid": (incident_df["start_ts"] >= cfg.valid_split) & (incident_df["start_ts"] < cfg.test_split),
        "test": incident_df["start_ts"] >= cfg.test_split,
    }
    confirmed_split = {
        "train": confirmed_df["start_ts"] < cfg.valid_split,
        "valid": (confirmed_df["start_ts"] >= cfg.valid_split) & (confirmed_df["start_ts"] < cfg.test_split),
        "test": confirmed_df["start_ts"] >= cfg.test_split,
    }

    cb_cat_idx = [feature_cols.index(c) for c in cat_feats]

    return FeatureBundle(
        incident_df=incident_df,
        confirmed_df=confirmed_df,
        incident_X=incident_X,
        confirmed_X=confirmed_X,
        incident_split=incident_split,
        confirmed_split=confirmed_split,
        base_num_feats=base_num_feats,
        cat_feats=cat_feats,
        feature_cols=feature_cols,
        cb_cat_idx=cb_cat_idx,
    )
