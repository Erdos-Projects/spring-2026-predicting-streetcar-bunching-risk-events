from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PipelineConfig, normalize_task_definitions


def decision_col(task_name: str) -> str:
    return f"decision_ts_task__{task_name}"


def eligible_col(task_name: str) -> str:
    return f"eligible_task__{task_name}"


def add_observation_checkpoint_targets(
    incident_df: pd.DataFrame,
    confirmed_df: pd.DataFrame,
    contacts: pd.DataFrame,
    cfg: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    incident_df = incident_df.copy().reset_index(drop=True)
    confirmed_df = confirmed_df.copy().reset_index(drop=True)

    for c in ["persists_next1_stop", "persists_next2_stops", "persists_next3_stops"]:
        if c in incident_df.columns:
            incident_df[c] = pd.to_numeric(incident_df[c], errors="coerce").fillna(0).astype("int8")

    task_defs = normalize_task_definitions(list(cfg.risk_task_definitions))
    if len(task_defs) == 0:
        raise ValueError("No risk_task_definitions provided.")

    inc = incident_df[["pair_key", "day", "bound", "start_ord", "start_ts"]].copy().reset_index(drop=True)
    inc["_row_id"] = np.arange(len(inc), dtype="int64")
    cand = contacts[["pair_key", "day", "bound", "pair_ord", "pair_ts"]].copy()

    obs_ts_cache: dict[int, pd.Series] = {}
    future_hit_cache: dict[tuple[int, int], pd.Series] = {}
    rel_hit_cache: dict[tuple[int, int], pd.Series] = {}

    def obs_ts_from_start(obs_offset: int) -> pd.Series:
        k = int(obs_offset)
        if k in obs_ts_cache:
            return obs_ts_cache[k].copy()
        base = inc[["_row_id", "pair_key", "day", "bound", "start_ord"]].copy()
        base["obs_ord"] = base["start_ord"] + k
        j = base.merge(
            cand,
            left_on=["pair_key", "day", "bound", "obs_ord"],
            right_on=["pair_key", "day", "bound", "pair_ord"],
            how="left",
        )
        out = j.groupby("_row_id", sort=False)["pair_ts"].min().reindex(inc["_row_id"])
        obs_ts_cache[k] = out
        return out.copy()

    def has_hit_from_obs(obs_offset: int, future_offset: int) -> pd.Series:
        key = (int(obs_offset), int(future_offset))
        if key in future_hit_cache:
            return future_hit_cache[key].copy()

        base = inc[["_row_id", "pair_key", "day", "bound", "start_ord"]].copy()
        base["obs_ord"] = base["start_ord"] + int(obs_offset)
        base["target_ord"] = base["obs_ord"] + int(future_offset)
        obs_ts = obs_ts_from_start(obs_offset)
        base = base.merge(obs_ts.rename("obs_ts"), left_on="_row_id", right_index=True, how="left")

        j = base.merge(
            cand,
            left_on=["pair_key", "day", "bound", "target_ord"],
            right_on=["pair_key", "day", "bound", "pair_ord"],
            how="left",
        )
        dt_min = (j["pair_ts"] - j["obs_ts"]).dt.total_seconds().div(60.0)
        max_dt_min = float(cfg.episode_max_gap_min) * float(max(1, int(future_offset)))
        j["is_hit"] = j["obs_ts"].notna() & j["pair_ts"].notna() & (dt_min > 0.0) & (dt_min <= max_dt_min)
        out = (
            j.groupby("_row_id", sort=False)["is_hit"]
            .max()
            .reindex(inc["_row_id"], fill_value=False)
            .astype("int8")
        )
        future_hit_cache[key] = out
        return out.copy()

    def has_hit_relative_to_obs(obs_offset: int, rel_offset: int) -> pd.Series:
        key = (int(obs_offset), int(rel_offset))
        if key in rel_hit_cache:
            return rel_hit_cache[key].copy()

        rel = int(rel_offset)
        if rel > 0:
            out = has_hit_from_obs(obs_offset, rel)
        elif rel == 0:
            out = obs_ts_from_start(obs_offset).notna().astype("int8")
        else:
            obs_ts = obs_ts_from_start(obs_offset)
            past_ts = obs_ts_from_start(obs_offset + rel)
            dt_min = (obs_ts - past_ts).dt.total_seconds().div(60.0)
            max_dt_min = float(cfg.episode_max_gap_min) * float(abs(rel))
            out = (
                obs_ts.notna()
                & past_ts.notna()
                & (dt_min > 0.0)
                & (dt_min <= max_dt_min)
            ).astype("int8")

        rel_hit_cache[key] = out
        return out.copy()

    def count_hits_relative_to_obs(obs_offset: int, rel_offsets: tuple[int, ...]) -> pd.Series:
        cnt = pd.Series(0, index=inc["_row_id"], dtype="int16")
        for off in rel_offsets:
            cnt = cnt + has_hit_relative_to_obs(obs_offset, int(off))
        return cnt.astype("int16")

    # Create shared checkpoint columns by s.
    unique_s = sorted({int(d["s"]) for d in task_defs})
    for s in unique_s:
        obs_offset = s - 1
        dcol = f"decision_ts_observed{s}"
        ecol = f"eligible_observed{s}"
        incident_df[dcol] = obs_ts_from_start(obs_offset).to_numpy()
        incident_df[ecol] = incident_df[dcol].notna().astype("int8")

    # Build targets + eligibility from (b,s,n,m).
    for d in task_defs:
        name = str(d["name"])
        b = int(d["b"])
        s = int(d["s"])
        n = int(d["n"])
        m = int(d["m"])

        obs_offset = s - 1
        lookback_offsets = tuple(range(-(s - 1), 1))
        future_offsets = tuple(range(1, m + 1))

        dts = obs_ts_from_start(obs_offset)
        cond_hits = count_hits_relative_to_obs(obs_offset, lookback_offsets)
        fut_hits = count_hits_relative_to_obs(obs_offset, future_offsets)

        incident_df[name] = (fut_hits >= n).astype("int8").to_numpy()
        incident_df[decision_col(name)] = dts.to_numpy()
        incident_df[eligible_col(name)] = (dts.notna() & (cond_hits >= b)).astype("int8").to_numpy()

    # # Compatibility aliases for legacy cells.
    # alias_pairs = [
    #     ("observed1_next12_any_binary", "observed1_next2_gap1_binary"),
    #     ("observed2_next12_any_binary", "observed2_next2_gap1_binary"),
    #     ("observed3_next12_any_binary", "observed3_next2_gap1_binary"),
    #     ("observed1_next_binary", "observed1_next2_gap1_binary"),
    #     ("incident_next1_binary_strict", "observed1_next1_binary"),
    #     ("incident_next12_any_binary", "observed1_next2_gap1_binary"),
    #     ("incident_next1_binary_relaxed", "observed1_next2_gap1_binary"),
    #     ("incident_next1_binary", "observed1_next2_gap1_binary"),
    #     ("obs2_next12_any_binary", "observed2_next2_gap1_binary"),
    #     ("obs3_next12_any_binary", "observed3_next2_gap1_binary"),
    #     ("obs2_next3_binary", "observed2_next3_gap1_binary"),
    #     ("obs3_next2_binary", "observed3_next2_gap1_binary"),
    #     ("obs3_next3_binary", "observed3_next3_gap1_binary"),
    #     ("observed1_next1_strict_binary", "observed1_next1_binary"),
    # ]
    # for out_col, src_col in alias_pairs:
    #     if src_col in incident_df.columns:
    #         incident_df[out_col] = incident_df[src_col].astype("int8")

    # # Compatibility map for confirmed2 aliases.
    # if {"observed3_next1_binary", "observed3_next2_binary"}.issubset(incident_df.columns):
    #     map_df = incident_df[
    #         ["episode_id", "observed3_next1_binary", "observed3_next2_binary"]
    #     ].copy()
    #     for c in ["observed3_next1_binary", "observed3_next2_binary"]:
    #         if c in confirmed_df.columns:
    #             confirmed_df = confirmed_df.drop(columns=[c])
    #     confirmed_df = confirmed_df.merge(map_df, on="episode_id", how="left")
    #     confirmed_df["observed3_next1_strict_binary"] = pd.to_numeric(
    #         confirmed_df["observed3_next1_binary"], errors="coerce"
    #     ).fillna(0).astype("int8")
    #     confirmed_df["observed3_next2_strict_binary"] = pd.to_numeric(
    #         confirmed_df["observed3_next2_binary"], errors="coerce"
    #     ).fillna(0).astype("int8")
    #     confirmed_df["confirmed2_next1_binary"] = confirmed_df["observed3_next1_strict_binary"].astype("int8")
    #     confirmed_df["confirmed2_next2_binary"] = confirmed_df["observed3_next2_strict_binary"].astype("int8")

    # Keep task defs attached for downstream visibility.
    incident_df.attrs["risk_task_definitions"] = task_defs
    return incident_df, confirmed_df

