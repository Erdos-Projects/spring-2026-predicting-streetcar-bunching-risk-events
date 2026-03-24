from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PipelineConfig


def _find_existing(paths: list[str] | tuple[str, ...]) -> Path | None:
    for p in paths:
        pp = Path(p)
        if pp.exists():
            return pp
    return None


def _extract_lon_lat(geom_text: str) -> tuple[float, float]:
    if pd.isna(geom_text):
        return (np.nan, np.nan)
    try:
        obj = json.loads(geom_text)
        typ = str(obj.get("type", ""))
        coords = obj.get("coordinates", None)
        if typ == "Point" and isinstance(coords, list) and len(coords) >= 2:
            return (float(coords[0]), float(coords[1]))
        if typ == "MultiPoint" and isinstance(coords, list) and len(coords) > 0 and len(coords[0]) >= 2:
            return (float(coords[0][0]), float(coords[0][1]))
    except Exception:
        pass
    return (np.nan, np.nan)


def _haversine_m(lat1: float, lon1: float, lats2: np.ndarray, lons2: np.ndarray) -> np.ndarray:
    r = 6371000.0
    lat1r = np.radians(lat1)
    lon1r = np.radians(lon1)
    lats2r = np.radians(lats2)
    lons2r = np.radians(lons2)
    dlat = lats2r - lat1r
    dlon = lons2r - lon1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lats2r) * (np.sin(dlon / 2.0) ** 2)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1e-12, 1.0 - a)))
    return r * c


def _compute_stop_poi_features(
    stop_geo: pd.DataFrame,
    path_candidates: list[str],
    prefix: str,
) -> tuple[pd.DataFrame, Path | None]:
    poi_path = _find_existing(path_candidates)
    feat = stop_geo[["stop_id", "stop_lat", "stop_lon"]].copy()
    feat[f"stop_nearest_{prefix}_m"] = np.nan
    feat[f"stop_{prefix}_within_250m"] = 0.0
    feat[f"stop_{prefix}_within_500m"] = 0.0

    if poi_path is not None:
        poi = pd.read_csv(poi_path, usecols=["geometry"], low_memory=False)
        ll = poi["geometry"].map(_extract_lon_lat)
        poi["poi_lon"] = [x[0] for x in ll]
        poi["poi_lat"] = [x[1] for x in ll]
        poi = poi.dropna(subset=["poi_lat", "poi_lon"]).reset_index(drop=True)

        if len(poi) > 0:
            poi_lats = poi["poi_lat"].to_numpy(dtype="float64")
            poi_lons = poi["poi_lon"].to_numpy(dtype="float64")
            nearest = []
            c250 = []
            c500 = []
            for _, row in feat.iterrows():
                if pd.isna(row["stop_lat"]) or pd.isna(row["stop_lon"]):
                    nearest.append(np.nan)
                    c250.append(0.0)
                    c500.append(0.0)
                    continue
                d = _haversine_m(float(row["stop_lat"]), float(row["stop_lon"]), poi_lats, poi_lons)
                nearest.append(float(np.min(d)))
                c250.append(float((d <= 250.0).sum()))
                c500.append(float((d <= 500.0).sum()))
            feat[f"stop_nearest_{prefix}_m"] = nearest
            feat[f"stop_{prefix}_within_250m"] = c250
            feat[f"stop_{prefix}_within_500m"] = c500

    feat = feat.rename(columns={"stop_id": "start_stopID_i"})
    cols = [
        "start_stopID_i",
        f"stop_nearest_{prefix}_m",
        f"stop_{prefix}_within_250m",
        f"stop_{prefix}_within_500m",
    ]
    return feat[cols], poi_path


def _daily_active(series_start: pd.Series, series_end: pd.Series, index_days: pd.DatetimeIndex) -> pd.Series:
    if len(index_days) == 0:
        return pd.Series(dtype="float64")
    d0_local = index_days[0]
    d1_local = index_days[-1]
    starts = pd.to_datetime(series_start, errors="coerce").dt.floor("D")
    ends = pd.to_datetime(series_end, errors="coerce").dt.floor("D")
    mask = starts.notna() & ends.notna() & (ends >= d0_local) & (starts <= d1_local)
    starts = starts[mask].clip(lower=d0_local, upper=d1_local)
    ends = ends[mask].clip(lower=d0_local, upper=d1_local)
    arr = np.zeros(len(index_days) + 1, dtype="int64")
    s_idx = (starts - d0_local).dt.days.to_numpy(dtype="int64")
    e_idx = (ends - d0_local).dt.days.to_numpy(dtype="int64") + 1
    np.add.at(arr, s_idx, 1)
    np.add.at(arr, e_idx, -1)
    return pd.Series(np.cumsum(arr[:-1]), index=index_days, dtype="float64")


def add_external_features(
    incident_df: pd.DataFrame,
    confirmed_df: pd.DataFrame,
    ev: pd.DataFrame,
    cfg: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, str | None]]:
    incident_df = incident_df.copy()
    confirmed_df = confirmed_df.copy()

    stops_path = _find_existing(["stops.csv", "./data_files/stops.csv"])
    if stops_path is None:
        raise FileNotFoundError("Could not find stops.csv for stop geospatial joins.")

    stops = pd.read_csv(stops_path, usecols=["stop_id", "stop_lat", "stop_lon"], low_memory=False)
    stops["stop_id"] = pd.to_numeric(stops["stop_id"], errors="coerce")
    stops["stop_lat"] = pd.to_numeric(stops["stop_lat"], errors="coerce")
    stops["stop_lon"] = pd.to_numeric(stops["stop_lon"], errors="coerce")
    stop_geo = stops.dropna(subset=["stop_id", "stop_lat", "stop_lon"]).copy()
    stop_geo["stop_id"] = stop_geo["stop_id"].astype(int)
    stop_geo = stop_geo.drop_duplicates(subset=["stop_id"])

    for df in [incident_df, confirmed_df]:
        df["start_stopID_i"] = pd.to_numeric(df["start_stopID"], errors="coerce").fillna(-1).astype(int)

    stop_geo_keyed = stop_geo.rename(columns={"stop_id": "start_stopID_i"})
    incident_df = incident_df.merge(stop_geo_keyed, on="start_stopID_i", how="left")
    confirmed_df = confirmed_df.merge(stop_geo_keyed, on="start_stopID_i", how="left")

    ped_feats, ped_path = _compute_stop_poi_features(
        stop_geo,
        ["Pedestrian Crossover - 4326.csv", "./data_files/Pedestrian Crossover - 4326.csv"],
        "pedx",
    )
    sig_feats, sig_path = _compute_stop_poi_features(
        stop_geo,
        ["Traffic Signal - 4326.csv", "./data_files/Traffic Signal - 4326.csv"],
        "sig",
    )
    incident_df = incident_df.merge(ped_feats, on="start_stopID_i", how="left")
    incident_df = incident_df.merge(sig_feats, on="start_stopID_i", how="left")
    confirmed_df = confirmed_df.merge(ped_feats, on="start_stopID_i", how="left")
    confirmed_df = confirmed_df.merge(sig_feats, on="start_stopID_i", how="left")

    ord_map = ev[["stopID", "bound", "bound_ordinal_f"]].copy()
    ord_map["start_stopID_i"] = pd.to_numeric(ord_map["stopID"], errors="coerce")
    ord_map = ord_map.dropna(subset=["start_stopID_i", "bound", "bound_ordinal_f"]).copy()
    ord_map["start_stopID_i"] = ord_map["start_stopID_i"].astype(int)
    ord_map = (
        ord_map.groupby(["start_stopID_i", "bound"], observed=True)["bound_ordinal_f"]
        .median()
        .reset_index()
    )
    route_ctx = ord_map.merge(
        ped_feats[["start_stopID_i", "stop_pedx_within_250m", "stop_pedx_within_500m"]],
        on="start_stopID_i",
        how="left",
    )
    route_ctx = route_ctx.merge(
        sig_feats[["start_stopID_i", "stop_sig_within_250m", "stop_sig_within_500m"]],
        on="start_stopID_i",
        how="left",
    )
    route_count_cols = [
        "stop_pedx_within_250m",
        "stop_pedx_within_500m",
        "stop_sig_within_250m",
        "stop_sig_within_500m",
    ]
    for c in route_count_cols:
        route_ctx[c] = pd.to_numeric(route_ctx[c], errors="coerce").fillna(0.0)
    route_ctx = route_ctx.sort_values(["bound", "bound_ordinal_f", "start_stopID_i"]).reset_index(drop=True)
    cum_cols = []
    for c in route_count_cols:
        out_c = f"cum_{c}_to_ord"
        route_ctx[out_c] = route_ctx.groupby("bound", observed=True)[c].cumsum()
        cum_cols.append(out_c)
    route_ctx_small = route_ctx[["start_stopID_i", "bound"] + cum_cols].drop_duplicates(["start_stopID_i", "bound"])
    incident_df = incident_df.merge(route_ctx_small, on=["start_stopID_i", "bound"], how="left")
    confirmed_df = confirmed_df.merge(route_ctx_small, on=["start_stopID_i", "bound"], how="left")

    # Permit intensity.
    perm_path = _find_existing(["Utility Cut Permits Data.csv", "./data_files/Utility Cut Permits Data.csv"])
    permit_active_city = pd.Series(dtype="float64")
    permit_active_route = pd.Series(dtype="float64")
    permit_starts_city_7d = pd.Series(dtype="float64")
    permit_starts_route_7d = pd.Series(dtype="float64")
    all_ts = pd.concat([incident_df["start_ts"], confirmed_df["start_ts"]], axis=0).dropna()
    if len(all_ts) > 0:
        d0 = all_ts.min().floor("D")
        d1 = all_ts.max().floor("D")
        day_index = pd.date_range(d0, d1, freq="D")
    else:
        day_index = pd.DatetimeIndex([])

    if perm_path is not None and len(day_index) > 0:
        permit_cols = ["PROPOSED_FROM_DATE", "PROPOSED_TO_DATE", "DISPLAY_DESC", "PERMIT_STATUS"]
        perm = pd.read_csv(perm_path, usecols=permit_cols, low_memory=False)
        perm["permit_start"] = pd.to_datetime(
            perm["PROPOSED_FROM_DATE"].astype(str).str.slice(0, 10),
            format="%Y-%m-%d",
            errors="coerce",
        ).dt.floor("D")
        perm["permit_end"] = pd.to_datetime(
            perm["PROPOSED_TO_DATE"].astype(str).str.slice(0, 10),
            format="%Y-%m-%d",
            errors="coerce",
        ).dt.floor("D")
        perm = perm.dropna(subset=["permit_start", "permit_end"]).copy()
        perm = perm[
            perm["permit_start"].dt.year.between(2010, 2035)
            & perm["permit_end"].dt.year.between(2010, 2035)
        ].copy()
        status = perm["PERMIT_STATUS"].fillna("").astype(str).str.upper()
        perm = perm[~status.str.contains("CANCEL", na=False)].copy()
        bad = perm["permit_end"] < perm["permit_start"]
        if bad.any():
            tmp = perm.loc[bad, "permit_start"].copy()
            perm.loc[bad, "permit_start"] = perm.loc[bad, "permit_end"]
            perm.loc[bad, "permit_end"] = tmp
        route_pat = r"\b(?:COLLEGE|CARLTON|GERRARD|MAIN|BROADVIEW|PARLIAMENT|BAY|SPADINA|BATHURST|OSSINGTON|DUFFERIN|RONCESVALLES|HIGH PARK|COXWELL|GREENWOOD)\b"
        desc_up = perm["DISPLAY_DESC"].fillna("").astype(str).str.upper()
        perm["is_route_hint"] = desc_up.str.contains(route_pat, regex=True, na=False)
        permit_active_city = _daily_active(perm["permit_start"], perm["permit_end"], day_index)
        perm_route = perm[perm["is_route_hint"]].copy()
        permit_active_route = _daily_active(perm_route["permit_start"], perm_route["permit_end"], day_index)
        starts_city = perm["permit_start"].value_counts().reindex(day_index, fill_value=0).sort_index()
        starts_route = perm_route["permit_start"].value_counts().reindex(day_index, fill_value=0).sort_index()
        permit_starts_city_7d = starts_city.rolling(7, min_periods=1).sum().astype("float64")
        permit_starts_route_7d = starts_route.rolling(7, min_periods=1).sum().astype("float64")

    for df in [incident_df, confirmed_df]:
        df["start_date"] = df["start_ts"].dt.floor("D")
        if len(day_index) > 0 and len(permit_active_city) > 0:
            df["permit_active_city"] = df["start_date"].map(permit_active_city).fillna(0.0)
            df["permit_active_route_hint"] = df["start_date"].map(permit_active_route).fillna(0.0)
            df["permit_new_city_7d"] = df["start_date"].map(permit_starts_city_7d).fillna(0.0)
            df["permit_new_route_7d"] = df["start_date"].map(permit_starts_route_7d).fillna(0.0)
        else:
            df["permit_active_city"] = 0.0
            df["permit_active_route_hint"] = 0.0
            df["permit_new_city_7d"] = 0.0
            df["permit_new_route_7d"] = 0.0
        df["permit_route_share"] = (
            df["permit_active_route_hint"] / df["permit_active_city"].replace(0, np.nan)
        ).fillna(0.0)

    # Hourly weather.
    weather_path = _find_existing(["hourly_weather_2024_2026.csv", "./data_files/hourly_weather_2024_2026.csv"])
    weather_cols = [
        "datetime_hour",
        "temp_c_hr",
        "precip_hr_mm",
        "wind_kmh_hr",
        "humidity_pct",
        "is_raining",
        "is_snowing",
        "is_freezing",
    ]
    if weather_path is not None:
        wx = pd.read_csv(weather_path, usecols=weather_cols, low_memory=False)
        wx["datetime_hour"] = pd.to_datetime(wx["datetime_hour"], errors="coerce").dt.floor("h")
        wx = wx.dropna(subset=["datetime_hour"]).drop_duplicates(subset=["datetime_hour"])

        def merge_wx(df: pd.DataFrame) -> pd.DataFrame:
            out = df.copy()
            out["start_hour"] = out["start_ts"].dt.floor("h")
            out = out.merge(wx, left_on="start_hour", right_on="datetime_hour", how="left")
            if "datetime_hour" in out.columns:
                out = out.drop(columns=["datetime_hour"])
            return out

        incident_df = merge_wx(incident_df)
        confirmed_df = merge_wx(confirmed_df)
    else:
        for df in [incident_df, confirmed_df]:
            df["temp_c_hr"] = 0.0
            df["precip_hr_mm"] = 0.0
            df["wind_kmh_hr"] = 0.0
            df["humidity_pct"] = 0.0
            df["is_raining"] = 0.0
            df["is_snowing"] = 0.0
            df["is_freezing"] = 0.0

    external_cols = [
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
    for df in [incident_df, confirmed_df]:
        for c in external_cols:
            if c not in df.columns:
                df[c] = 0.0
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype("float32")

    external_sources = {
        "stops": str(stops_path),
        "pedestrian_crossover": str(ped_path) if ped_path is not None else None,
        "traffic_signal": str(sig_path) if sig_path is not None else None,
        "road_permits": str(perm_path) if perm_path is not None else None,
        "weather": str(weather_path) if weather_path is not None else None,
    }
    return incident_df, confirmed_df, external_cols, external_sources

