from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .raw_events import load_raw_events, resolve_csv_path

from .config import PipelineConfig


@dataclass(frozen=True)
class ContextColumns:
    lag_cols: list[str]
    run_ctx_cols: list[str]
    stop_ctx_cols: list[str]


def _dedupe_keep_order(paths: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for p in paths:
        pp = str(Path(p))
        key = pp.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(pp)
    return out


def _manifest_relpath_to_path(base_dir: Path, rel_path: str) -> Path:
    parts = [x for x in rel_path.replace("\\", "/").split("/") if x]
    return base_dir.joinpath(*parts)


def _load_chunk_manifest_map(cfg: PipelineConfig) -> dict[str, list[str]]:
    for cand in cfg.chunk_manifest_candidates:
        manifest_path = Path(cand)
        if not manifest_path.exists():
            continue

        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        files = payload.get("files", [])
        if not isinstance(files, list):
            continue

        out: dict[str, list[str]] = {}
        for item in files:
            if not isinstance(item, dict):
                continue

            name = str(item.get("name", "")).strip()
            rel_parts = item.get("parts", [])
            if (not name) or (not isinstance(rel_parts, list)) or len(rel_parts) == 0:
                continue

            abs_parts = [
                _manifest_relpath_to_path(manifest_path.parent, str(x))
                for x in rel_parts
            ]
            if not all(p.exists() for p in abs_parts):
                continue
            out[name] = [str(p) for p in abs_parts]

        if len(out) > 0:
            return out

    return {}


def _resolve_candidate_group(
    candidates: tuple[str, ...] | list[str],
    chunk_map: dict[str, list[str]],
    prefer_chunks: bool,
) -> list[str] | None:
    valid = [str(c).strip() for c in candidates if str(c).strip()]
    if len(valid) == 0:
        return None

    probe_order = ("chunk", "file") if prefer_chunks else ("file", "chunk")
    for mode in probe_order:
        for cand in valid:
            p = Path(cand)
            if mode == "file":
                if p.exists():
                    return [str(p)]
                continue

            chunk_parts = chunk_map.get(p.name)
            if chunk_parts:
                return list(chunk_parts)

    return None


def load_events_table(
    cfg: PipelineConfig,
    csv_path: str | None = None,
    extra_csv_paths: list[str] | None = None,
    max_rows: int | None = None,
    max_delay_minutes: float | None = 120.0,
) -> pd.DataFrame:
    chunk_map = _load_chunk_manifest_map(cfg)
    prefer_chunks_primary = cfg.prefer_chunked_events if csv_path is None else False

    if csv_path is None:
        primary_candidates = list(cfg.events_primary_candidates)
        try:
            resolved_primary = resolve_csv_path(None)
            if resolved_primary:
                primary_candidates = [resolved_primary] + primary_candidates
        except Exception:
            pass
    else:
        primary_candidates = [csv_path]

    primary_paths = _resolve_candidate_group(
        primary_candidates,
        chunk_map=chunk_map,
        prefer_chunks=prefer_chunks_primary,
    )
    if primary_paths is None:
        fallback_primary = csv_path if csv_path is not None else resolve_csv_path(None)
        primary_paths = [fallback_primary]

    if extra_csv_paths is None:
        resolved_extras: list[str] = []
        for cands in cfg.events_extra_candidates:
            paths = _resolve_candidate_group(
                cands,
                chunk_map=chunk_map,
                prefer_chunks=cfg.prefer_chunked_events,
            )
            if paths:
                resolved_extras.extend(paths)
        extra_paths = resolved_extras
    else:
        extra_paths = []
        for p in extra_csv_paths:
            resolved = _resolve_candidate_group(
                [p],
                chunk_map=chunk_map,
                prefer_chunks=False,
            )
            if resolved:
                extra_paths.extend(resolved)
            else:
                extra_paths.append(p)

    # Preserve original semantics: max_rows limits primary read only.
    # When the primary source is chunked, keep only the first chunk under max_rows.
    if max_rows is not None and len(primary_paths) > 1:
        primary_for_load = primary_paths[:1]
    else:
        primary_for_load = primary_paths

    all_paths = _dedupe_keep_order(primary_for_load + extra_paths)
    csv_path = all_paths[0]
    extra_csv_paths = all_paths[1:] or None

    try:
        return load_raw_events(
            csv_path,
            max_rows=max_rows,
            max_delay_minutes=max_delay_minutes,
            extra_csv_paths=extra_csv_paths,
        )
    except Exception as exc:
        msg = str(exc).lower()
        if "out of memory" not in msg and "parsererror" not in exc.__class__.__name__.lower():
            raise

        split_main = _resolve_candidate_group(
            cfg.split_fallback_candidates[0],
            chunk_map=chunk_map,
            prefer_chunks=cfg.prefer_chunked_events,
        )
        split_part2 = _resolve_candidate_group(
            cfg.split_fallback_candidates[1],
            chunk_map=chunk_map,
            prefer_chunks=cfg.prefer_chunked_events,
        )
        if split_main is None or split_part2 is None:
            raise RuntimeError("Split fallback files not found: all506_df-new-1/2.csv") from exc

        fallback_all_paths = _dedupe_keep_order(split_main + split_part2 + (extra_csv_paths or []))
        return load_raw_events(
            fallback_all_paths[0],
            max_rows=max_rows,
            max_delay_minutes=max_delay_minutes,
            extra_csv_paths=fallback_all_paths[1:] or None,
        )


def build_event_frame(events: pd.DataFrame, cfg: PipelineConfig) -> tuple[pd.DataFrame, ContextColumns]:
    ev = events.sort_values("ts").reset_index(drop=True).copy()
    for c in [
        "sch_adherence_f",
        "mins_delayed_f",
        "gap_sec",
        "headway_sec",
        "gap_headway_ratio",
        "bound_ordinal_f",
        "schedule_offset_sec",
        "bunched_i",
    ]:
        ev[c] = pd.to_numeric(ev[c], errors="coerce")

    ev["Vehicle_num"] = pd.to_numeric(ev["Vehicle"], errors="coerce").round().astype("Int64")
    ev["day"] = ev["ts"].dt.floor("D")
    ev["hour"] = ev["ts"].dt.hour.astype("int8")
    ev["dow"] = ev["ts"].dt.dayofweek.astype("int8")
    ev["month"] = ev["ts"].dt.month.astype("int8")
    ev["is_weekend"] = (ev["dow"] >= 5).astype("int8")
    ev["delay_pos"] = ev["sch_adherence_f"].clip(lower=0)

    ratio_from_raw = ev["gap_sec"] / ev["headway_sec"].replace(0, np.nan)
    ev["gap_headway_ratio"] = (
        pd.to_numeric(ev["gap_headway_ratio"], errors="coerce")
        .fillna(ratio_from_raw)
        .astype("float32")
    )
    ev[cfg.bunch_col] = (
        (ev["gap_headway_ratio"] < cfg.bunch_ratio_threshold)
        & ev["gap_headway_ratio"].notna()
    ).astype("int8")

    # Vehicle-specific lag features, strictly prior within vehicle-day-bound.
    tmp = ev[
        ["Vehicle_num", "day", "bound", "ts", "bound_ordinal_f", "sch_adherence_f", cfg.bunch_col]
    ].copy()
    tmp = tmp.sort_values(["Vehicle_num", "day", "bound", "ts"])
    grp = tmp.groupby(["Vehicle_num", "day", "bound"], observed=True, sort=False)

    for k in [1, 2, 3]:
        tmp[f"veh_lag{k}_sch"] = grp["sch_adherence_f"].shift(k)
        tmp[f"veh_lag{k}_bunched"] = grp[cfg.bunch_col].shift(k)

    lag_sch_cols = [f"veh_lag{k}_sch" for k in [1, 2, 3]]
    lag_b_cols = [f"veh_lag{k}_bunched" for k in [1, 2, 3]]
    tmp["veh_lag123_sch_mean"] = tmp[lag_sch_cols].mean(axis=1)
    tmp["veh_lag123_bunched_sum"] = tmp[lag_b_cols].fillna(0).sum(axis=1)

    for c in lag_sch_cols + ["veh_lag123_sch_mean"]:
        ev[c] = pd.to_numeric(tmp[c], errors="coerce").astype("float32")
    for c in lag_b_cols + ["veh_lag123_bunched_sum"]:
        ev[c] = pd.to_numeric(tmp[c], errors="coerce").fillna(0).astype("float32")

    tmp["veh_run_pos"] = grp.cumcount().astype("float32")
    tmp["veh_elapsed_min"] = (
        (tmp["ts"] - grp["ts"].transform("min")).dt.total_seconds().div(60.0)
    ).astype("float32")

    bound_max_by_bound = pd.to_numeric(
        ev.groupby("bound", observed=True)["bound_ordinal_f"].max(),
        errors="coerce",
    )
    tmp["veh_run_progress"] = (
        pd.to_numeric(tmp["bound_ordinal_f"], errors="coerce")
        / tmp["bound"].map(bound_max_by_bound).replace(0, np.nan)
    ).fillna(0.0).astype("float32")

    for c in ["veh_run_pos", "veh_elapsed_min", "veh_run_progress"]:
        ev[c] = pd.to_numeric(tmp[c], errors="coerce").fillna(0.0).astype("float32")

    bound_max_ord = pd.to_numeric(
        ev.groupby("bound")["bound_ordinal_f"].transform("max"),
        errors="coerce",
    )
    ev["remaining_stops_to_terminal"] = (
        (bound_max_ord - pd.to_numeric(ev["bound_ordinal_f"], errors="coerce"))
        .clip(lower=0)
        .fillna(0.0)
        .astype("float32")
    )
    del tmp, grp, bound_max_by_bound

    # Stop-level lagged bunch context from prior bins only.
    ev["bin5"] = ev["ts"].dt.floor("5min")
    stop_bin = (
        ev.groupby(["stopID", "bound", "bin5"], observed=True)
        .agg(sb_n=(cfg.bunch_col, "size"), sb_k=(cfg.bunch_col, "sum"))
        .reset_index()
        .sort_values(["stopID", "bound", "bin5"])
    )
    sbg = stop_bin.groupby(["stopID", "bound"], observed=True, sort=False)
    stop_bin["sb_prev_k_30m"] = sbg["sb_k"].transform(lambda s: s.shift(1).rolling(6, min_periods=1).sum())
    stop_bin["sb_prev_n_30m"] = sbg["sb_n"].transform(lambda s: s.shift(1).rolling(6, min_periods=1).sum())
    stop_bin["sb_prev_k_60m"] = sbg["sb_k"].transform(lambda s: s.shift(1).rolling(12, min_periods=1).sum())
    stop_bin["sb_prev_n_60m"] = sbg["sb_n"].transform(lambda s: s.shift(1).rolling(12, min_periods=1).sum())
    stop_bin["stop_recent_bunch_n_30m"] = stop_bin["sb_prev_k_30m"].fillna(0.0)
    stop_bin["stop_recent_bunch_rate_30m"] = (
        stop_bin["sb_prev_k_30m"] / stop_bin["sb_prev_n_30m"].replace(0, np.nan)
    ).fillna(0.0)
    stop_bin["stop_recent_bunch_n_60m"] = stop_bin["sb_prev_k_60m"].fillna(0.0)
    stop_bin["stop_recent_bunch_rate_60m"] = (
        stop_bin["sb_prev_k_60m"] / stop_bin["sb_prev_n_60m"].replace(0, np.nan)
    ).fillna(0.0)

    keep_stop_ctx = [
        "stopID",
        "bound",
        "bin5",
        "stop_recent_bunch_n_30m",
        "stop_recent_bunch_rate_30m",
        "stop_recent_bunch_n_60m",
        "stop_recent_bunch_rate_60m",
    ]
    ev = ev.merge(stop_bin[keep_stop_ctx], on=["stopID", "bound", "bin5"], how="left")
    for c in [
        "stop_recent_bunch_n_30m",
        "stop_recent_bunch_rate_30m",
        "stop_recent_bunch_n_60m",
        "stop_recent_bunch_rate_60m",
    ]:
        ev[c] = pd.to_numeric(ev[c], errors="coerce").fillna(0.0).astype("float32")

    lag_cols = [
        "veh_lag1_sch",
        "veh_lag2_sch",
        "veh_lag3_sch",
        "veh_lag123_sch_mean",
        "veh_lag1_bunched",
        "veh_lag2_bunched",
        "veh_lag3_bunched",
        "veh_lag123_bunched_sum",
    ]
    stop_ctx_cols = [
        "stop_recent_bunch_n_30m",
        "stop_recent_bunch_rate_30m",
        "stop_recent_bunch_n_60m",
        "stop_recent_bunch_rate_60m",
    ]
    run_ctx_cols = [
        "veh_run_pos",
        "veh_elapsed_min",
        "veh_run_progress",
        "remaining_stops_to_terminal",
    ]

    return ev, ContextColumns(lag_cols=lag_cols, run_ctx_cols=run_ctx_cols, stop_ctx_cols=stop_ctx_cols)


def _hw_state(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return np.select([x < 0.75, x > 1.25], ["short", "long"], default="ontime")


def build_contacts(ev: pd.DataFrame, cfg: PipelineConfig, ctx: ContextColumns) -> pd.DataFrame:
    bx = ev.loc[
        (ev[cfg.bunch_col] > 0) & ev["Vehicle_num"].notna(),
        [
            "ts",
            "day",
            "stopID",
            "bound",
            "bound_ordinal_f",
            "Vehicle_num",
            "sch_adherence_f",
            "delay_pos",
            "gap_headway_ratio",
            "gap_sec",
            "headway_sec",
            "schedule_offset_sec",
        ]
        + ctx.lag_cols
        + ctx.run_ctx_cols
        + ctx.stop_ctx_cols,
    ].copy()
    bx = bx.sort_values(["day", "bound", "stopID", "ts"]).reset_index(drop=True)

    g = bx.groupby(["day", "bound", "stopID"], observed=True, sort=False)
    bx["prev_ts"] = g["ts"].shift(1)
    bx["prev_v"] = g["Vehicle_num"].shift(1)
    bx["prev_ord"] = g["bound_ordinal_f"].shift(1)
    bx["prev_sch"] = g["sch_adherence_f"].shift(1)
    bx["prev_delay"] = g["delay_pos"].shift(1)
    bx["prev_ratio"] = g["gap_headway_ratio"].shift(1)
    bx["prev_gap_sec"] = g["gap_sec"].shift(1)
    bx["prev_headway_sec"] = g["headway_sec"].shift(1)
    bx["prev_sched_offset"] = g["schedule_offset_sec"].shift(1)
    for c in ctx.lag_cols + ctx.run_ctx_cols:
        bx[f"prev_{c}"] = g[c].shift(1)
    bx["dt_sec"] = (bx["ts"] - bx["prev_ts"]).dt.total_seconds()

    hw_cur = pd.to_numeric(bx["headway_sec"], errors="coerce")
    hw_prev = pd.to_numeric(bx["prev_headway_sec"], errors="coerce")
    bx["pair_dt_limit_sec"] = (
        pd.concat([hw_cur, hw_prev], axis=1)
        .min(axis=1)
        .mul(cfg.pair_dt_ratio_to_sched)
        .clip(lower=cfg.pair_dt_min_sec, upper=cfg.pair_dt_max_sec)
        .fillna(cfg.pair_dt_fallback_sec)
    )

    mask = (
        bx["prev_v"].notna()
        & (bx["Vehicle_num"] != bx["prev_v"])
        & (bx["dt_sec"] >= 0)
        & (bx["dt_sec"] <= bx["pair_dt_limit_sec"])
    )
    contacts = bx.loc[mask].copy()
    if len(contacts) == 0:
        raise RuntimeError("No pair contacts found with current thresholds.")

    veh_lo = np.minimum(
        contacts["Vehicle_num"].astype("int64").to_numpy(),
        contacts["prev_v"].astype("int64").to_numpy(),
    )
    veh_hi = np.maximum(
        contacts["Vehicle_num"].astype("int64").to_numpy(),
        contacts["prev_v"].astype("int64").to_numpy(),
    )
    contacts["veh_lo"] = veh_lo
    contacts["veh_hi"] = veh_hi
    contacts["pair_key"] = contacts["veh_lo"].astype(str) + "_" + contacts["veh_hi"].astype(str)
    contacts["pair_ts"] = contacts["prev_ts"] + (contacts["ts"] - contacts["prev_ts"]) / 2
    contacts["pair_ord"] = contacts[["bound_ordinal_f", "prev_ord"]].max(axis=1).fillna(0).astype(int)
    contacts["pair_sch_mean"] = contacts[["sch_adherence_f", "prev_sch"]].mean(axis=1)
    contacts["pair_delay_mean"] = contacts[["delay_pos", "prev_delay"]].mean(axis=1)
    contacts["pair_gap_ratio_mean"] = contacts[["gap_headway_ratio", "prev_ratio"]].mean(axis=1)
    contacts["pair_delay_abs_diff"] = (contacts["delay_pos"] - contacts["prev_delay"]).abs()
    contacts["pair_sch_abs_diff"] = (contacts["sch_adherence_f"] - contacts["prev_sch"]).abs()
    contacts["pair_delay_min"] = contacts[["delay_pos", "prev_delay"]].min(axis=1)
    contacts["pair_delay_max"] = contacts[["delay_pos", "prev_delay"]].max(axis=1)
    contacts["pair_sch_min"] = contacts[["sch_adherence_f", "prev_sch"]].min(axis=1)
    contacts["pair_sch_max"] = contacts[["sch_adherence_f", "prev_sch"]].max(axis=1)

    contacts["fl_ratio_to_sched"] = pd.to_numeric(contacts["gap_headway_ratio"], errors="coerce")
    contacts["ll1_ratio_to_sched"] = pd.to_numeric(contacts["prev_ratio"], errors="coerce")
    contacts["fl_gap_sec"] = pd.to_numeric(contacts["gap_sec"], errors="coerce")
    contacts["ll1_gap_sec"] = pd.to_numeric(contacts["prev_gap_sec"], errors="coerce")
    contacts["fl_sched_headway_sec"] = pd.to_numeric(contacts["headway_sec"], errors="coerce")
    contacts["ll1_sched_headway_sec"] = pd.to_numeric(contacts["prev_headway_sec"], errors="coerce")
    contacts["pair_sched_headway_sec_mean"] = contacts[["fl_sched_headway_sec", "ll1_sched_headway_sec"]].mean(axis=1)
    contacts["pair_sched_headway_min_mean"] = contacts["pair_sched_headway_sec_mean"] / 60.0
    contacts["pair_sched_headway_min_sq"] = contacts["pair_sched_headway_min_mean"] ** 2
    contacts["pair_gap_sec_mean"] = contacts[["fl_gap_sec", "ll1_gap_sec"]].mean(axis=1)
    contacts["pair_gap_minus_sched_sec"] = contacts["pair_gap_sec_mean"] - contacts["pair_sched_headway_sec_mean"]
    contacts["pair_fl_ll1_ratio_diff"] = contacts["fl_ratio_to_sched"] - contacts["ll1_ratio_to_sched"]
    contacts["fl_hw_state"] = _hw_state(contacts["fl_ratio_to_sched"])
    contacts["ll1_hw_state"] = _hw_state(contacts["ll1_ratio_to_sched"])
    contacts["fl_ll1_hw_state"] = contacts["fl_hw_state"].astype(str) + "__" + contacts["ll1_hw_state"].astype(str)

    for c in ctx.lag_cols:
        contacts[f"pair_{c}_mean"] = contacts[[c, f"prev_{c}"]].mean(axis=1)
    for c in ["veh_lag1_sch", "veh_lag2_sch", "veh_lag3_sch"]:
        contacts[f"pair_{c}_abs_diff"] = (contacts[c] - contacts[f"prev_{c}"]).abs()
    for c in ctx.run_ctx_cols:
        contacts[f"pair_{c}_mean"] = contacts[[c, f"prev_{c}"]].mean(axis=1)

    contacts["pair_sch_trend_13"] = contacts["pair_veh_lag1_sch_mean"] - contacts["pair_veh_lag3_sch_mean"]
    contacts["pair_sch_trend_12"] = contacts["pair_veh_lag1_sch_mean"] - contacts["pair_veh_lag2_sch_mean"]
    contacts["pair_bunched_trend_13"] = contacts["pair_veh_lag1_bunched_mean"] - contacts["pair_veh_lag3_bunched_mean"]
    contacts["pair_bunched_trend_12"] = contacts["pair_veh_lag1_bunched_mean"] - contacts["pair_veh_lag2_bunched_mean"]

    contacts["hour"] = contacts["pair_ts"].dt.hour.astype("int8")
    contacts["dow"] = contacts["pair_ts"].dt.dayofweek.astype("int8")
    contacts["month"] = contacts["pair_ts"].dt.month.astype("int8")
    contacts["is_weekend"] = (contacts["dow"] >= 5).astype("int8")

    contacts["pair_bin"] = contacts["pair_ts"].dt.floor(f"{cfg.pair_dedup_bin_min}min")
    contacts = (
        contacts.sort_values(["pair_key", "day", "bound", "stopID", "pair_bin", "dt_sec"])
        .drop_duplicates(["pair_key", "day", "bound", "stopID", "pair_bin"], keep="first")
        .reset_index(drop=True)
    )

    pair_lag_cols = [
        "pair_veh_lag1_sch_mean",
        "pair_veh_lag2_sch_mean",
        "pair_veh_lag3_sch_mean",
        "pair_veh_lag123_sch_mean_mean",
        "pair_veh_lag1_bunched_mean",
        "pair_veh_lag2_bunched_mean",
        "pair_veh_lag3_bunched_mean",
        "pair_veh_lag123_bunched_sum_mean",
        "pair_veh_lag1_sch_abs_diff",
        "pair_veh_lag2_sch_abs_diff",
        "pair_veh_lag3_sch_abs_diff",
        "pair_veh_run_pos_mean",
        "pair_veh_elapsed_min_mean",
        "pair_veh_run_progress_mean",
        "pair_remaining_stops_to_terminal_mean",
        "pair_sch_trend_13",
        "pair_sch_trend_12",
        "pair_bunched_trend_13",
        "pair_bunched_trend_12",
    ]
    paper_pair_cols = [
        "fl_ratio_to_sched",
        "ll1_ratio_to_sched",
        "pair_fl_ll1_ratio_diff",
        "fl_gap_sec",
        "ll1_gap_sec",
        "fl_sched_headway_sec",
        "ll1_sched_headway_sec",
        "pair_sched_headway_sec_mean",
        "pair_sched_headway_min_mean",
        "pair_sched_headway_min_sq",
        "pair_gap_sec_mean",
        "pair_gap_minus_sched_sec",
        "fl_ll1_hw_state",
    ]
    contact_cols = [
        "pair_key",
        "veh_lo",
        "veh_hi",
        "pair_ts",
        "day",
        "bound",
        "stopID",
        "pair_ord",
        "dt_sec",
        "pair_sch_mean",
        "pair_delay_mean",
        "pair_gap_ratio_mean",
        "pair_delay_abs_diff",
        "pair_sch_abs_diff",
        "pair_delay_min",
        "pair_delay_max",
        "pair_sch_min",
        "pair_sch_max",
    ] + pair_lag_cols + paper_pair_cols + ctx.stop_ctx_cols
    return contacts[contact_cols].copy()


def build_episode_tables(contacts: pd.DataFrame, cfg: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    contacts = contacts.sort_values(["pair_key", "day", "bound", "pair_ts"]).reset_index(drop=True)
    g2 = contacts.groupby(["pair_key", "day", "bound"], observed=True, sort=False)
    contacts["prev_pair_ts"] = g2["pair_ts"].shift(1)
    contacts["prev_pair_ord"] = g2["pair_ord"].shift(1)
    contacts["gap_min"] = (contacts["pair_ts"] - contacts["prev_pair_ts"]).dt.total_seconds().div(60.0)
    contacts["ord_step"] = contacts["pair_ord"] - contacts["prev_pair_ord"]
    contacts["is_strict_link"] = (
        contacts["prev_pair_ts"].notna()
        & (contacts["gap_min"] <= cfg.episode_max_gap_min)
        & (contacts["ord_step"] == 1)
    )
    contacts["new_episode"] = (~contacts["is_strict_link"]).astype("int8")
    contacts["episode_num"] = g2["new_episode"].cumsum().astype("int32")
    contacts["episode_id"] = (
        contacts["pair_key"]
        + "|"
        + contacts["day"].dt.strftime("%Y-%m-%d")
        + "|"
        + contacts["bound"].astype(str)
        + "|"
        + contacts["episode_num"].astype(str)
    )

    episodes = (
        contacts.groupby("episode_id", observed=True)
        .agg(
            pair_key=("pair_key", "first"),
            day=("day", "first"),
            bound=("bound", "first"),
            veh_lo=("veh_lo", "first"),
            veh_hi=("veh_hi", "first"),
            start_ts=("pair_ts", "min"),
            end_ts=("pair_ts", "max"),
            start_stopID=("stopID", "first"),
            start_ord=("pair_ord", "first"),
            n_contacts=("pair_ts", "size"),
            n_unique_stops=("stopID", "nunique"),
            ord_min=("pair_ord", "min"),
            ord_max=("pair_ord", "max"),
            pair_delay_mean_avg=("pair_delay_mean", "mean"),
            pair_delay_sum=("pair_delay_mean", "sum"),
            start_pair_delay=("pair_delay_mean", "first"),
            start_pair_sch=("pair_sch_mean", "first"),
            start_pair_delay_abs_diff=("pair_delay_abs_diff", "first"),
            start_pair_sch_abs_diff=("pair_sch_abs_diff", "first"),
            start_pair_delay_min=("pair_delay_min", "first"),
            start_pair_delay_max=("pair_delay_max", "first"),
            start_pair_sch_min=("pair_sch_min", "first"),
            start_pair_sch_max=("pair_sch_max", "first"),
            start_dt_sec=("dt_sec", "first"),
            start_gap_ratio=("pair_gap_ratio_mean", "first"),
        )
        .reset_index()
    )

    start_extra_cols = [
        "pair_veh_lag1_sch_mean",
        "pair_veh_lag2_sch_mean",
        "pair_veh_lag3_sch_mean",
        "pair_veh_lag123_sch_mean_mean",
        "pair_veh_lag1_bunched_mean",
        "pair_veh_lag2_bunched_mean",
        "pair_veh_lag3_bunched_mean",
        "pair_veh_lag123_bunched_sum_mean",
        "pair_veh_lag1_sch_abs_diff",
        "pair_veh_lag2_sch_abs_diff",
        "pair_veh_lag3_sch_abs_diff",
        "pair_veh_run_pos_mean",
        "pair_veh_elapsed_min_mean",
        "pair_veh_run_progress_mean",
        "pair_remaining_stops_to_terminal_mean",
        "pair_sch_trend_13",
        "pair_sch_trend_12",
        "pair_bunched_trend_13",
        "pair_bunched_trend_12",
        "fl_ratio_to_sched",
        "ll1_ratio_to_sched",
        "pair_fl_ll1_ratio_diff",
        "fl_gap_sec",
        "ll1_gap_sec",
        "fl_sched_headway_sec",
        "ll1_sched_headway_sec",
        "pair_sched_headway_sec_mean",
        "pair_sched_headway_min_mean",
        "pair_sched_headway_min_sq",
        "pair_gap_sec_mean",
        "pair_gap_minus_sched_sec",
        "fl_ll1_hw_state",
        "stop_recent_bunch_n_30m",
        "stop_recent_bunch_rate_30m",
        "stop_recent_bunch_n_60m",
        "stop_recent_bunch_rate_60m",
    ]
    start_extra = (
        contacts.sort_values("pair_ts")
        .groupby("episode_id", observed=True)[start_extra_cols]
        .first()
        .add_prefix("start_")
        .reset_index()
    )
    episodes = episodes.merge(start_extra, on="episode_id", how="left")
    episodes["duration_min"] = (episodes["end_ts"] - episodes["start_ts"]).dt.total_seconds().div(60.0)
    episodes["ord_span"] = (episodes["ord_max"] - episodes["ord_min"]).astype(int)
    episodes["persists_next1_stop"] = (episodes["n_unique_stops"] >= 2).astype("int8")
    episodes["persists_next2_stops"] = (episodes["n_unique_stops"] >= 3).astype("int8")
    episodes["persists_next3_stops"] = (episodes["n_unique_stops"] >= 4).astype("int8")
    episodes["future_additional_stops"] = (episodes["n_unique_stops"] - 1).clip(lower=0).astype("int16")

    episodes_all = episodes.copy()
    episodes = episodes_all.copy().sort_values("start_ts").reset_index(drop=True)
    episodes["pair_prev_episode_count"] = episodes.groupby("pair_key", observed=True).cumcount().astype("int32")
    episodes["veh_lo_prev_episode_count"] = episodes.groupby("veh_lo", observed=True).cumcount().astype("int32")
    episodes["veh_hi_prev_episode_count"] = episodes.groupby("veh_hi", observed=True).cumcount().astype("int32")
    return episodes, episodes_all


def build_incident_tables(
    episodes: pd.DataFrame,
    episodes_all: pd.DataFrame,
    cfg: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    incident_df = episodes.copy()
    confirmed_df = episodes_all.loc[episodes_all["n_unique_stops"] >= 3].copy()

    if cfg.filter_2024_pre_july:
        incident_df = incident_df.loc[incident_df["start_ts"] >= cfg.min_ts].copy()

    for df in [incident_df, confirmed_df]:
        df["veh_lo"] = pd.to_numeric(df["veh_lo"], errors="coerce").fillna(-1).astype("int64")
        df["veh_hi"] = pd.to_numeric(df["veh_hi"], errors="coerce").fillna(-1).astype("int64")
        if "hour" not in df.columns:
            df["hour"] = df["start_ts"].dt.hour.astype("int8")
        if "dow" not in df.columns:
            df["dow"] = df["start_ts"].dt.dayofweek.astype("int8")
        if "month" not in df.columns:
            df["month"] = df["start_ts"].dt.month.astype("int8")
        if "is_weekend" not in df.columns:
            df["is_weekend"] = (df["dow"] >= 5).astype("int8")

    incident_df = incident_df.sort_values(["start_ts", "episode_id"]).reset_index(drop=True)
    confirmed_df = confirmed_df.sort_values(["start_ts", "episode_id"]).reset_index(drop=True)
    return incident_df, confirmed_df
