from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


def parse_mmss(value: object) -> float | None:
    if pd.isna(value):
        return None
    s = str(value).strip()
    m = re.match(r"^(\d{1,3}):(\d{2})$", s)
    if not m:
        return None
    return float(int(m.group(1)) * 60 + int(m.group(2)))


def parse_schedule_offset(value: object) -> float | None:
    if pd.isna(value):
        return None
    s = str(value)
    m = re.search(r"(\d{1,3}:\d{2})\s+(ahead|behind)", s)
    if not m:
        return None
    mm, ss = m.group(1).split(":")
    sign = -1 if m.group(2) == "ahead" else 1
    return float(sign * (int(mm) * 60 + int(ss)))


def resolve_csv_path(explicit_path: str | Path | None = None) -> Path:
    if explicit_path:
        p = Path(explicit_path)
        if p.exists():
            return p
    for p in (Path("df2025_all.csv"), Path("toy model/df2025_all.csv")):
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find df2025_all.csv. Pass csv_path or run from a folder where it exists."
    )


def _is_modern_schema(columns: pd.Index) -> bool:
    cols = {str(c) for c in columns}
    required = {"datetime", "stopID", "bound"}
    return required.issubset(cols)


def _extract_stop_map_part(df: pd.DataFrame) -> pd.DataFrame:
    needed = {"stopID", "bound", "bound_ordinal"}
    if not needed.issubset(set(df.columns)):
        return pd.DataFrame(columns=["stopID", "bound", "bound_ordinal"])

    out = df.loc[:, ["stopID", "bound", "bound_ordinal"]].copy()
    out["stopID"] = out["stopID"].astype(str)
    out["bound"] = out["bound"].astype(str)
    out["bound_ordinal"] = pd.to_numeric(out["bound_ordinal"], errors="coerce")
    out = out.dropna(subset=["stopID", "bound", "bound_ordinal"])
    out = out.drop_duplicates(subset=["stopID", "bound", "bound_ordinal"])
    return out


def _compose_stop_map(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if len(parts) == 0:
        return pd.DataFrame(columns=["stopID", "bound", "bound_ordinal"])
    merged = pd.concat(parts, ignore_index=True)
    merged = (
        merged.groupby(["stopID", "bound"], observed=True)["bound_ordinal"]
        .median()
        .reset_index()
    )
    return merged


def _load_default_stop_map_from_disk() -> pd.DataFrame:
    for p in (Path("df2025_all.csv"), Path("toy model/df2025_all.csv")):
        if not p.exists():
            continue
        try:
            src = pd.read_csv(
                p,
                usecols=["stopID", "bound", "bound_ordinal"],
                low_memory=False,
            )
        except Exception:
            continue
        part = _extract_stop_map_part(src)
        if len(part) == 0:
            continue
        return _compose_stop_map([part])
    return pd.DataFrame(columns=["stopID", "bound", "bound_ordinal"])


def _derive_bound(destination: pd.Series) -> pd.Series:
    east_kws = ["east", "main street", "main st", "coxwell"]
    west_kws = ["west", "high park", "roncesvalles", "dufferin"]
    dest_lower = destination.astype(str).str.lower().fillna("")
    bound = pd.Series(np.nan, index=destination.index, dtype=object)
    bound[dest_lower.str.contains("|".join(east_kws), regex=True)] = "E"
    bound[dest_lower.str.contains("|".join(west_kws), regex=True)] = "W"
    return bound


def normalize_legacy_raw(df: pd.DataFrame, stop_map: pd.DataFrame | None = None) -> pd.DataFrame:
    required = ["Schedule", "Time", "day", "Destination", "Gap", "stopID"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Legacy CSV is missing required columns: {missing}")

    out = df.copy()

    is_sched = out["Schedule"].astype(str).str.contains("Scheduled at", na=False)
    out = out[~is_sched].reset_index(drop=True)

    def _parse_time(t: object):
        s = str(t).split("-")[0].replace("(sched)", "").strip()
        for fmt in ("%I:%M:%S%p", "%I:%M%p", "%H:%M:%S", "%H:%M"):
            try:
                return pd.to_datetime(s, format=fmt).time()
            except Exception:
                pass
        return None

    out["_time_obj"] = out["Time"].map(_parse_time)
    out["_day_dt"] = pd.to_datetime(out["day"], errors="coerce")
    valid = out["_time_obj"].notna() & out["_day_dt"].notna()
    out["datetime"] = pd.NaT
    out.loc[valid, "datetime"] = out.loc[valid].apply(
        lambda r: pd.Timestamp.combine(r["_day_dt"].date(), r["_time_obj"]),
        axis=1,
    )
    out = out.drop(columns=["_time_obj", "_day_dt"])

    def _parse_gap_value(g: object) -> float:
        if pd.isna(g):
            return np.nan
        s = str(g).strip()
        if "days" in s or s.count(":") >= 2:
            try:
                return float(pd.to_timedelta(s).total_seconds())
            except Exception:
                pass
        mmss = parse_mmss(s)
        return float(mmss) if mmss is not None else np.nan

    out["gap_sec"] = out["Gap"].map(_parse_gap_value)
    out["bunched"] = (out["gap_sec"] < 120).astype("Int8")
    out["gapped"] = (out["gap_sec"] >= 1140).astype("Int8")
    out["bunched"] = out["bunched"].where(out["gap_sec"].notna(), other=pd.NA)
    out["gapped"] = out["gapped"].where(out["gap_sec"].notna(), other=pd.NA)
    out["sch_adherence"] = out["Schedule"].map(parse_schedule_offset).astype("float32") / 60.0
    out["mins delayed"] = out["sch_adherence"].clip(lower=0)
    out["bound"] = _derive_bound(out["Destination"])

    if stop_map is None:
        stop_map = _load_default_stop_map_from_disk()
    if len(stop_map) > 0:
        out["stopID"] = out["stopID"].astype(str)
        out = out.merge(stop_map, on=["stopID", "bound"], how="left")
    else:
        out["bound_ordinal"] = np.nan

    for col in ["upstream_bunched_5stops_3h_source", "cond_sum", "prev3hr_delays"]:
        if col not in out.columns:
            out[col] = np.nan

    return out


def _normalize_outlier_policy(policy: str | None) -> str:
    mode = "clip" if policy is None else str(policy).strip().lower()
    aliases = {
        "cap": "clip",
        "capped": "clip",
        "remove": "drop",
        "filter": "drop",
        "off": "none",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"clip", "drop", "none"}:
        raise ValueError(
            f"Unknown outlier_policy={policy!r}. Use one of: 'clip', 'drop', 'none'."
        )
    return mode


def _coerce_event_fields(
    df: pd.DataFrame,
    max_delay_minutes: float | None,
    max_gap_minutes: float | None,
    outlier_policy: str = "clip",
) -> pd.DataFrame:
    out = df.copy()
    desired_cols = [
        "datetime",
        "stopID",
        "bound",
        "bound_ordinal",
        "Gap",
        "Headway",
        "Schedule",
        "upstream_bunched_5stops_3h_source",
        "cond_sum",
        "prev3hr_delays",
        "bunched",
        "gapped",
        "mins delayed",
        "sch_adherence",
        "Vehicle",
    ]
    for c in desired_cols:
        if c not in out.columns:
            out[c] = np.nan

    out["ts"] = pd.to_datetime(out["datetime"], errors="coerce")
    out["stopID"] = out["stopID"].astype("string")
    out["bound"] = out["bound"].astype("string")
    out = out[out["ts"].notna() & out["stopID"].notna() & out["bound"].notna()].copy()

    if "gap_sec" in out.columns:
        gap_sec = pd.to_numeric(out["gap_sec"], errors="coerce")
    else:
        gap_sec = pd.Series(np.nan, index=out.index, dtype="float64")
    parsed_td = pd.to_timedelta(out["Gap"], errors="coerce").dt.total_seconds()
    parsed_mmss = pd.to_numeric(out["Gap"].map(parse_mmss), errors="coerce")
    out["gap_sec"] = gap_sec.fillna(parsed_td).fillna(parsed_mmss)
    gap_sec2 = pd.to_numeric(out["gap_sec"], errors="coerce")
    gap_sec2 = gap_sec2.where(gap_sec2 >= 0)

    headway_from_mmss = out["Headway"].map(parse_mmss)
    out["headway_sec"] = pd.to_numeric(headway_from_mmss, errors="coerce")
    out["schedule_offset_sec"] = out["Schedule"].map(parse_schedule_offset)

    out["bunched_i"] = (
        pd.to_numeric(out["bunched"], errors="coerce").fillna(0).astype(float) > 0
    ).astype("int8")
    out["gapped_i"] = (
        pd.to_numeric(out["gapped"], errors="coerce").fillna(0).astype(float) > 0
    ).astype("int8")
    out["incident"] = ((out["bunched_i"] + out["gapped_i"]) > 0).astype("int8")

    source_col = "upstream_bunched_5stops_3h_source"
    if source_col not in out.columns or out[source_col].isna().all():
        source_col = "cond_sum"

    policy = _normalize_outlier_policy(outlier_policy)
    delay_raw = pd.to_numeric(out["mins delayed"], errors="coerce").clip(lower=0)

    if policy == "drop":
        keep = pd.Series(True, index=out.index, dtype=bool)
        if max_delay_minutes is not None:
            delay_cap = float(max_delay_minutes)
            keep &= delay_raw.isna() | (delay_raw <= delay_cap)
        if max_gap_minutes is not None:
            gap_cap_sec = float(max_gap_minutes) * 60.0
            keep &= gap_sec2.isna() | (gap_sec2 <= gap_cap_sec)
        dropped = int((~keep).sum())
        if dropped > 0:
            print(f"[load_raw_events] dropped {dropped:,} outlier rows (delay/gap cap).")
        out = out.loc[keep].copy()
        delay_raw = delay_raw.loc[out.index]
        gap_sec2 = gap_sec2.loc[out.index]
    elif policy == "clip":
        if max_delay_minutes is not None:
            delay_raw = delay_raw.clip(upper=float(max_delay_minutes))
        if max_gap_minutes is not None:
            gap_sec2 = gap_sec2.clip(upper=float(max_gap_minutes) * 60.0)

    out["gap_sec"] = gap_sec2.astype("float32")
    out["mins_delayed_f"] = delay_raw.astype("float32")

    out["upstream_bunched_5stops_3h_source_f"] = pd.to_numeric(
        out[source_col], errors="coerce"
    ).fillna(0.0)
    out["bound_ordinal_f"] = pd.to_numeric(out["bound_ordinal"], errors="coerce").fillna(0.0)
    out["prev3hr_delays_f"] = pd.to_numeric(out["prev3hr_delays"], errors="coerce").fillna(0.0)
    out["sch_adherence_f"] = pd.to_numeric(out["sch_adherence"], errors="coerce").astype("float32")

    headway_sec = pd.to_numeric(out["headway_sec"], errors="coerce")
    out["gap_excess_sec"] = (gap_sec2 - headway_sec).clip(lower=0).astype("float32")
    out["gap_headway_ratio"] = (gap_sec2 / headway_sec.replace(0, np.nan)).astype("float32")

    return out


def load_raw_events(
    csv_path: str | Path,
    max_rows: int | None = None,
    max_delay_minutes: float | None = 120.0,
    max_gap_minutes: float | None = 120.0,
    outlier_policy: str = "clip",
    extra_csv_paths: list[str | Path] | None = None,
) -> pd.DataFrame:
    all_paths = [Path(csv_path)] + [Path(p) for p in (extra_csv_paths or [])]
    if len(all_paths) == 0:
        raise ValueError("No CSV paths provided.")

    frames: list[pd.DataFrame] = []
    stop_map_parts: list[pd.DataFrame] = []
    cached_stop_map: pd.DataFrame | None = None

    for i, path in enumerate(all_paths):
        if not path.exists():
            print(f"[load_raw_events] path not found, skipping: {path}")
            continue

        row_cap = max_rows if i == 0 else None
        cols = pd.read_csv(path, nrows=0).columns
        is_modern = _is_modern_schema(cols)
        raw = pd.read_csv(path, low_memory=False, nrows=row_cap)

        if is_modern:
            print(f"[load_raw_events] loading modern CSV: {path}")
            norm_base = raw
            part = _extract_stop_map_part(raw)
            if len(part) > 0:
                stop_map_parts.append(part)
                cached_stop_map = None
        else:
            print(f"[load_raw_events] loading legacy CSV: {path}")
            if cached_stop_map is None:
                cached_stop_map = _compose_stop_map(stop_map_parts)
                if len(cached_stop_map) == 0:
                    cached_stop_map = _load_default_stop_map_from_disk()
            norm_base = normalize_legacy_raw(raw, stop_map=cached_stop_map)

        events = _coerce_event_fields(
            norm_base,
            max_delay_minutes=max_delay_minutes,
            max_gap_minutes=max_gap_minutes,
            outlier_policy=outlier_policy,
        )
        if len(events) > 0:
            dt_min = events["ts"].min().date()
            dt_max = events["ts"].max().date()
            print(f"  loaded {len(events):,} events  ({dt_min} to {dt_max})")
        else:
            print("  loaded 0 events")
        frames.append(events)

    if len(frames) == 0:
        raise FileNotFoundError("No event files could be loaded.")

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values("ts").reset_index(drop=True)
    if len(out) > 0:
        print(
            f"[load_raw_events] combined total: {len(out):,} rows  "
            f"({out['ts'].min().date()} to {out['ts'].max().date()})"
        )
    return out
