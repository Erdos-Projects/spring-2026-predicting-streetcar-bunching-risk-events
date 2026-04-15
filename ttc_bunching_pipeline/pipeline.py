from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import PipelineConfig
from .data import (
    ContextColumns,
    build_contacts,
    build_episode_tables,
    build_event_frame,
    build_incident_tables,
    load_events_table,
)
from .external import add_external_features
from .features import FeatureBundle, build_feature_bundle
from .targets import add_observation_checkpoint_targets


@dataclass(frozen=True)
class TrainingDataBundle:
    cfg: PipelineConfig
    events: pd.DataFrame
    ev: pd.DataFrame
    ctx: ContextColumns
    contacts: pd.DataFrame
    episodes: pd.DataFrame
    episodes_all: pd.DataFrame
    incident_df: pd.DataFrame
    confirmed_df: pd.DataFrame
    feature_bundle: FeatureBundle
    external_cols: list[str]
    external_sources: dict[str, str | None]



    @property
    def source_info(self) -> dict[str, str | None]:
        return self.external_sources


@dataclass(frozen=True)
class InferenceDataBundle:
    cfg: PipelineConfig
    events: pd.DataFrame
    ev: pd.DataFrame
    ctx: ContextColumns
    contacts: pd.DataFrame
    episodes: pd.DataFrame
    episodes_all: pd.DataFrame
    incident_df: pd.DataFrame
    confirmed_df: pd.DataFrame
    feature_bundle: FeatureBundle
    external_cols: list[str]
    external_sources: dict[str, str | None]



    @property
    def source_info(self) -> dict[str, str | None]:
        return self.external_sources


def build_training_data_bundle(
    cfg: PipelineConfig,
    csv_path: str | None = None,
    extra_csv_paths: list[str] | None = None,
    add_external: bool = True,
    max_rows: int | None = None,
) -> TrainingDataBundle:
    events = load_events_table(
        cfg=cfg,
        csv_path=csv_path,
        extra_csv_paths=extra_csv_paths,
        max_rows=max_rows,
    )
    ev, ctx = build_event_frame(events, cfg)
    contacts = build_contacts(ev, cfg, ctx)
    episodes, episodes_all = build_episode_tables(contacts, cfg)
    incident_df, confirmed_df = build_incident_tables(episodes, episodes_all, cfg)


    if add_external:
        incident_df, confirmed_df, external_cols, external_sources = add_external_features(
            incident_df, confirmed_df, ev, cfg
        )
    else:
        external_cols, external_sources = [], {}

    incident_df, confirmed_df = add_observation_checkpoint_targets(
        incident_df=incident_df,
        confirmed_df=confirmed_df,
        contacts=contacts,
        cfg=cfg,
    )

    feature_bundle = build_feature_bundle(
        incident_df=incident_df,
        confirmed_df=confirmed_df,
        cfg=cfg,
        include_external=add_external,
    )

    return TrainingDataBundle(
        cfg=cfg,
        events=events,
        ev=ev,
        ctx=ctx,
        contacts=contacts,
        episodes=episodes,
        episodes_all=episodes_all,
        incident_df=feature_bundle.incident_df,
        confirmed_df=feature_bundle.confirmed_df,
        feature_bundle=feature_bundle,
        external_cols=external_cols,
        external_sources=external_sources,
    )


def build_inference_data_bundle(
    cfg: PipelineConfig,
    csv_path: str | None = None,
    extra_csv_paths: list[str] | None = None,
    add_external: bool = True,
    max_rows: int | None = None,
) -> InferenceDataBundle:
    events = load_events_table(
        cfg=cfg,
        csv_path=csv_path,
        extra_csv_paths=extra_csv_paths,
        max_rows=max_rows,
    )
    ev, ctx = build_event_frame(events, cfg)
    contacts = build_contacts(ev, cfg, ctx)
    episodes, episodes_all = build_episode_tables(contacts, cfg)
    incident_df, confirmed_df = build_incident_tables(episodes, episodes_all, cfg)


    if add_external:
        incident_df, confirmed_df, external_cols, external_sources = add_external_features(
            incident_df, confirmed_df, ev, cfg
        )
    else:
        external_cols, external_sources = [], {}

    feature_bundle = build_feature_bundle(
        incident_df=incident_df,
        confirmed_df=confirmed_df,
        cfg=cfg,
        include_external=add_external,
    )

    return InferenceDataBundle(
        cfg=cfg,
        events=events,
        ev=ev,
        ctx=ctx,
        contacts=contacts,
        episodes=episodes,
        episodes_all=episodes_all,
        incident_df=feature_bundle.incident_df,
        confirmed_df=feature_bundle.confirmed_df,
        feature_bundle=feature_bundle,
        external_cols=external_cols,
        external_sources=external_sources,
    )
