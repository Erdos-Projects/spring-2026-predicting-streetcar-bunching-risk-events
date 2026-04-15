from __future__ import annotations

import json
from dataclasses import dataclass
import copy
from itertools import product

import numpy as np
import pandas as pd
import xgboost as xgb
try:
    import lightgbm as lgb
except Exception:  # pragma: no cover - optional dependency
    lgb = None

from bayes_opt import BayesianOptimization
# except Exception:  # pragma: no cover - optional dependency
#     BayesianOptimization = None

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import PipelineConfig
from .config import normalize_task_definitions
from .targets import decision_col, eligible_col


@dataclass(frozen=True)
class TaskSpec:
    task: str
    X: pd.DataFrame
    y: pd.Series
    ts: pd.Series
    split: dict[str, pd.Series]


def default_task_threshold_policy() -> dict[str, str]:


    return {
    "observed1_next2_gap1_binary": {'name': "f2", 'fpr_cap': 40},
    "observed2_next2_gap1_binary": {'name': "f2",'fpr_cap': 40},
    
    "observed2_next3_gap1_binary": {"name": "f2", 'fpr_cap': 40},
    "observed3_next2_gap1_binary": {"name": "f2", 'fpr_cap': 40},
    "observed3_next3_gap1_binary": {'name': "f2", 'fpr_cap': 40},
    "observed3_next4_gap1_binary": {'name': "f2", 'fpr_cap': 40},
    "observed1_next1_binary": {'name':"f2"},
    "observed2_next1_binary": {'name':"f2"},
    "observed2_next2_binary": {'name':"f2"},
    "observed3_next2_binary": {'name':"f2"},
    "observed3_next3_binary": {'name':"f3", 'fpr_cap': 40},
    "cond3of4_to_next4_ge2_binary": {'name': "f2", 'fpr_cap': 40},
    "cond3of5_to_next4_ge2_binary": {'name': "f2", 'fpr_cap': 40},
    "cond3of5_to_next5_ge3_binary": {'name': "f2", 'fpr_cap': 40},
}




def threshold_stats(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tnr = float(tn / max(tn + fp, 1))
    tpr = float(tp / max(tp + fn, 1))
    fpr = float(fp / max(tn + fp, 1))
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2,zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "tnr": tnr,
        "tpr": tpr,
        "fpr": fpr,
        "alert_rate": float(y_pred.mean()),
    }


def pick_threshold(
    y_true: np.ndarray,
    p_hat: np.ndarray,
    policy: dict,
    # default_fpr_cap: bool 
) -> tuple[float, dict[str, float]]:
    grid = np.linspace(0.05, 0.95, 91)
    rows = []
    for t in grid:
        y_pred = (p_hat >= t).astype("int8")
        st = threshold_stats(y_true, y_pred)
        st["threshold"] = float(t)
        rows.append(st)
    cand = pd.DataFrame(rows)

    policy_name = policy.get( "name" )
    # if policy.get( "name" ) else default_policy[0]
    policy_fpr_cap = policy.get("fpr_cap") 
    
    # if policy.get("fpr_cap") else default_policy[1]   
     
    row = cand.sort_values( [ policy_name, "f1" ], ascending=False).iloc[0] 
    
        # Support either fraction (0.30) or percent-style (30) caps.
    # cap = float(policy["fpr_cap"])
    if policy_fpr_cap:
        if policy_fpr_cap> 1.0:
            policy_fpr_cap = policy_fpr_cap / 100.0
        policy_fpr_cap = min(max(policy_fpr_cap, 0.0), 1.0)

        below_cap = cand[cand["fpr"] <= policy_fpr_cap]
        if len(below_cap):
            row = below_cap.sort_values([policy["name"], "f1"], ascending=False).iloc[0]
        else:
            row = cand.sort_values(["fpr", policy["name"]], ascending=[True, False]).iloc[0]
        
    # if policy == "f1":
    #     row = cand.sort_values(["f1", "balanced_accuracy"], ascending=False).iloc[0]
    # elif policy == "balacc":
    #     row = cand.sort_values(["balanced_accuracy", "f1"], ascending=False).iloc[0]
    # elif policy == "f2_fprcap30":
    #     ok = cand[cand["fpr"] <= 30]
    #     if len(ok):
    #         row = ok.sort_values(["f1", "balanced_accuracy"], ascending=False).iloc[0]
    #     else:
    #         row = cand.sort_values(["fpr", "f1"], ascending=[True, False]).iloc[0]
    # else:
    #     raise ValueError(f"Unknown threshold policy: {policy}")

    return float(row["threshold"]), {
        policy_name: float(row[ policy_name ]),
        "f1": float(row["f1"]),
        "fpr": float(row["fpr"]),
        "tnr": float(row["tnr"]),
        "alert_rate": float(row["alert_rate"]),
    }

### build train/val/test splits out of data 
def build_walkforward_folds(
    ts: pd.Series,
    pretest_mask: pd.Series,
    cfg: PipelineConfig,
) -> list[dict[str, pd.Series | pd.Timestamp]]:
    
    ts = pd.to_datetime(ts)
    pretest_mask = pd.Series(pretest_mask, index=ts.index).astype(bool)
    ts_pre = ts.loc[pretest_mask].sort_values()
    if ts_pre.empty:
        return []

    valid_td = pd.Timedelta(days=cfg.strict_cv_valid_days)
    buffer_td = pd.Timedelta(days=cfg.strict_cv_buffer_days)
    min_train_td = pd.Timedelta(days=cfg.strict_cv_min_train_days)

    cur_end = ts_pre.max().normalize() + pd.Timedelta(days=1)  ## current-day end: set timestamp to midnight, add a day. So, 3:14 3/14/25 -> 00:00 3/14/25 -> 00:00 3/15/25  
    min_ts = ts_pre.min( )

    folds: list[dict[str, pd.Series | pd.Timestamp]] = []
    attempts = 0
    
    while len(folds) < int(cfg.strict_cv_n_folds) and attempts < int(cfg.strict_cv_n_folds) * 8:
        attempts += 1
        va_end = cur_end
        va_start = va_end - valid_td
        tr_end = va_start - buffer_td

        train_mask = pretest_mask & (ts < tr_end)
        valid_mask = pretest_mask & (ts >= va_start) & (ts < va_end)
        
        enough_rows = (
            int(train_mask.sum()) >= cfg.strict_cv_min_train_rows
            and int(valid_mask.sum()) >= cfg.strict_cv_min_valid_rows
        )
        enough_span = False
        
        if int(train_mask.sum()) > 0:
            tr_ts = ts.loc[train_mask]
            enough_span = (tr_ts.max() - tr_ts.min()) >= min_train_td

        if enough_rows and enough_span:
            folds.append(
                {
                    "train": train_mask,
                    "valid": valid_mask,
                    "valid_start": va_start,
                    "valid_end": va_end,
                }
            )

        cur_end = va_start
        if cur_end <= (min_ts + valid_td):
            break

    return list(reversed(folds))

### determine if we need to scale positive weights for imbalanced datasets
def xgb_scale_pos_weight(y_arr: np.ndarray) -> float:
    """
    Return class-ratio scaling for positive class.

    For standard imbalanced tasks (positives are minority), this is > 1.
    For high-positive-rate tasks (positives are majority), this is < 1, which
    effectively upweights the rarer negative class relative to positives.
    """
    pos = float(np.sum(y_arr == 1))
    neg = float(np.sum(y_arr == 0))
    if pos <= 0 or neg <= 0:
        return 1.0

    # Keep weighting bounded in either direction: [1/8, 8].
    ratio = neg / pos
    return float(min(8.0, max(0.125, ratio)))


def _resolve_threshold_policy_for_task(
    y: pd.Series,
    cfg: PipelineConfig,
    policy: dict | None,
) -> dict:
    """
    Resolve a per-task threshold policy and adapt defaults when positives are
    very common.
    """
    out = dict(policy) if policy else {
        "name": cfg.threshold_policy_global_default,
        "fpr_cap": cfg.threshold_fpr_cap_global_default,
    }

    if "name" not in out or out["name"] is None:
        out["name"] = cfg.threshold_policy_global_default
    if "fpr_cap" not in out or out["fpr_cap"] is None:
        out["fpr_cap"] = cfg.threshold_fpr_cap_global_default

    y_arr = pd.to_numeric(y, errors="coerce").fillna(0).astype("int8").to_numpy()
    if y_arr.size == 0:
        return out

    pos_rate = float(np.mean(y_arr))
    default_name = str(cfg.threshold_policy_global_default).lower()
    policy_name = str(out.get("name", cfg.threshold_policy_global_default)).lower()

    # If a task is heavily positive and still using the default metric, switch
    # to balanced accuracy so thresholding emphasizes both classes.
    if pos_rate >= 0.80 and policy_name == default_name:
        out["name"] = "balanced_accuracy"
        cap = out.get("fpr_cap", 0.20)
        cap = 0.20 if cap is None else float(cap)
        if cap > 1.0:
            cap = cap / 100.0
        out["fpr_cap"] = min(max(cap, 0.0), 0.20)

    return out


# def _require_lightgbm() -> None:
#     if lgb is None:
#         raise ModuleNotFoundError(
#             "lightgbm is not installed. Install it with `pip install lightgbm` to run LGBM benchmarks."
#         )


def _require_bayes_opt() -> None:
    if BayesianOptimization is None:
        raise ModuleNotFoundError(
            "bayesian-optimization is not installed. Install it with "
            "`pip install bayesian-optimization` to run Bayesian XGBoost tuning."
        )


# def _fit_lgbm_with_early_stopping(
#     model,
#     Xtr: pd.DataFrame,
#     ytr: np.ndarray,
#     Xva: pd.DataFrame,
#     yva: np.ndarray,
# ) -> None:
#     _require_lightgbm()
#     model.fit(
#         Xtr,
#         ytr,
#         eval_set=[(Xva, yva)],
#         eval_metric="auc",
#         callbacks=[lgb.early_stopping(stopping_rounds=80, verbose=False)],
#     )


def encode_xgb(train_df: pd.DataFrame, other_df: pd.DataFrame, cat_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    cat_cols = [c for c in cat_cols if c in train_df.columns]
    tr = pd.get_dummies(train_df.copy(), columns=cat_cols, dummy_na=True)
    ot = pd.get_dummies(other_df.copy(), columns=cat_cols, dummy_na=True)
    ot = ot.reindex(columns=tr.columns, fill_value=0)
    tr = tr.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype("float32")
    ot = ot.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype("float32")
    return tr, ot


# def encode_xgb_for_model(
#     model: xgb.XGBClassifier,
#     X_live: pd.DataFrame,
#     cat_cols: list[str],
# ) -> pd.DataFrame:
#     cat_cols = [c for c in cat_cols if c in X_live.columns]
#     Xenc = pd.get_dummies(X_live.copy(), columns=cat_cols, dummy_na=True)

#     feat_names = list(getattr(model, "feature_names_in_", []))
#     if len(feat_names) == 0:
#         booster_names = model.get_booster().feature_names
#         if booster_names is not None:
#             feat_names = list(booster_names)

#     if len(feat_names) == 0:
#         raise ValueError("Model has no feature names; cannot align live features safely.")

#     Xenc = Xenc.reindex(columns=feat_names, fill_value=0)
#     Xenc = Xenc.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype("float32")
#     return Xenc


# def _threshold_map(
#     thresholds: dict[str, float] | pd.DataFrame | None,
# ) -> dict[str, float]:
#     if thresholds is None:
#         return {}

#     if isinstance(thresholds, dict):
#         out: dict[str, float] = {}
#         for k, v in thresholds.items():
#             out[str(k)] = float(v)
#         return out

#     if isinstance(thresholds, pd.DataFrame):
#         req = {"task", "best_threshold"}
#         if not req.issubset(set(thresholds.columns)):
#             raise ValueError(
#                 "Threshold DataFrame must contain columns: 'task' and 'best_threshold'."
#             )
#         out = {}
#         for _, row in thresholds[["task", "best_threshold"]].dropna().iterrows():
#             out[str(row["task"])] = float(row["best_threshold"])
#         return out

#     raise TypeError(
#         "thresholds must be a dict[str, float], a DataFrame with "
#         "['task','best_threshold'], or None."
#     )


# def predict_live(
#     X_live: pd.DataFrame,
#     models: dict[str, xgb.XGBClassifier],
#     cat_feats: list[str],
#     thresholds: dict[str, float] | pd.DataFrame | None = None,
#     eligible_by_task: dict[str, pd.Series] | None = None,
#     default_threshold: float = 0.5,
# ) -> dict[str, pd.DataFrame]:
#     threshold_by_task = _threshold_map(thresholds)
#     out: dict[str, pd.DataFrame] = {}

#     for task, model in models.items():
#         Xenc = encode_xgb_for_model(model=model, X_live=X_live, cat_cols=cat_feats)
#         proba = model.predict_proba(Xenc)[:, 1].astype("float64")
#         thr = float(threshold_by_task.get(task, default_threshold))
#         pred = (proba >= thr).astype("int8")

#         pred_df = pd.DataFrame(
#             {
#                 "proba": proba,
#                 "threshold": float(thr),
#                 "pred": pred,
#             },
#             index=X_live.index,
#         )

#         if eligible_by_task is not None and task in eligible_by_task:
#             eligible = pd.Series(eligible_by_task[task], index=X_live.index).fillna(False).astype(bool)
#             pred_df["eligible"] = eligible.astype("int8")
#             pred_df["pred_live"] = (pred_df["pred"].astype("int8") & pred_df["eligible"].astype("int8")).astype("int8")
#         else:
#             pred_df["eligible"] = np.int8(1)
#             pred_df["pred_live"] = pred_df["pred"].astype("int8")

#         out[task] = pred_df

#     return out


def run_binary_classifier_xgb(
    task: str,
    X: pd.DataFrame,
    y: pd.Series,
    ts: pd.Series,
    split: dict[str, pd.Series],
    cat_feats: list[str],
    cfg: PipelineConfig,
    threshold_policy: dict,
    pre_tuned_params: dict | None = None  
) -> tuple[dict[str, object], xgb.XGBClassifier, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    if pre_tuned_params:
        xgb_params = pre_tuned_params[task]
    else: 
        xgb_params = cfg.xgb_params


    y = pd.to_numeric(y, errors="coerce").fillna(0).astype("int8")
    ts = pd.to_datetime(ts)
    
    pretest_mask = pd.Series(split["train"] | split["valid"], index=X.index).astype(bool)
    test_mask = pd.Series(split["test"], index=X.index).astype(bool)
    
    folds = build_walkforward_folds(ts=ts, pretest_mask=pretest_mask, cfg=cfg)
    
    
    if len(folds) < 2:
        
        ## default fold splitting ... not enough rows to produce specified CV split with n folds per 
        folds = [
            {
                "train": pd.Series(split["train"], index=X.index).astype(bool),
                "valid": pd.Series(split["valid"], index=X.index).astype(bool),
                "valid_start": pd.Timestamp(cfg.valid_split),
                "valid_end": pd.Timestamp(cfg.test_split),
            }
        ]

    cfg_rows = []
    cfg_oof_cache: dict[tuple, pd.DataFrame] = {}
    for xcfg in xgb_params:
        fold_ap = []
        used_folds = 0
        oof_rows_cfg = []
        for fold_id, fd in enumerate(folds):
            tr_mask = fd["train"]
            va_mask = fd["valid"]
            ytr = y.loc[tr_mask].to_numpy(dtype="int8")
            yva = y.loc[va_mask].to_numpy(dtype="int8")
            if len(np.unique(ytr)) < 2 or len(np.unique(yva)) < 2:
                continue
            Xtr, Xva = encode_xgb(X.loc[tr_mask], X.loc[va_mask], cat_feats)
            
            spw = xgb_scale_pos_weight(ytr)
            
            model = xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=cfg.seed,
                n_jobs=-1,
                verbosity=0,
                early_stopping_rounds=80,
                scale_pos_weight=spw,
                **xcfg,
            )
            model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            pva = model.predict_proba(Xva)[:, 1]
            fold_ap.append(float(average_precision_score(yva, pva)))
            used_folds += 1
            oof_rows_cfg.append(
                pd.DataFrame(
                    {
                        "idx": X.index[va_mask],
                        "fold_id": int(fold_id),
                        "y_true": yva.astype("int8"),
                        "proba": pva.astype("float64"),
                    }
                )
            )
        if len(fold_ap) > 0:
            cfg_rows.append(
                {
                    "task": task,
                    "cfg": json.dumps(xcfg),
                    "cv_ap_mean": float(np.mean(fold_ap)),
                    "cv_ap_std": float(np.std(fold_ap)),
                    "cv_used_folds": int(used_folds),
                }
            )
            if len(oof_rows_cfg) > 0:
                cfg_oof_cache[_cfg_key(xcfg)] = pd.concat(oof_rows_cfg, ignore_index=True)
    if len(cfg_rows) == 0:
        raise RuntimeError(f"No valid XGBoost CV folds for task={task}.")
    cfg_scores_df = pd.DataFrame(cfg_rows).sort_values(["cv_ap_mean", "cv_ap_std"], ascending=[False, True]).reset_index(drop=True)
    best_cfg = json.loads(cfg_scores_df.iloc[0]["cfg"])
    best_cv_ap = float(cfg_scores_df.iloc[0]["cv_ap_mean"])
    best_cv_ap_std = float(cfg_scores_df.iloc[0]["cv_ap_std"])

    
    

    ## OOF threshold tuning (reuse cached OOF from cfg-scoring loop to avoid retraining)
    oof_df = cfg_oof_cache.get(_cfg_key(best_cfg))
    
    
    # if oof_df is None or len(oof_df) == 0:
    #     # Fallback: rebuild OOF for best_cfg if cache is unavailable.
    #     oof_rows = []
    #     for fold_id, fd in enumerate(folds):
    #         tr_mask = fd["train"]
    #         va_mask = fd["valid"]
    #         ytr = y.loc[tr_mask].to_numpy(dtype="int8")
    #         yva = y.loc[va_mask].to_numpy(dtype="int8")
    #         if len(np.unique(ytr)) < 2 or len(np.unique(yva)) < 2:
    #             continue
    #         Xtr, Xva = encode_xgb(X.loc[tr_mask], X.loc[va_mask], cat_feats)
    #         spw = xgb_scale_pos_weight(ytr)
    #         model = xgb.XGBClassifier(
    #             objective="binary:logistic",
    #             eval_metric="logloss",
    #             tree_method="hist",
    #             random_state=cfg.seed,
    #             n_jobs=-1,
    #             verbosity=0,
    #             early_stopping_rounds=80,
    #             scale_pos_weight=spw,
    #             **best_cfg,
    #         )
    #         model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    #         pva = model.predict_proba(Xva)[:, 1]
    #         oof_rows.append(
    #             pd.DataFrame(
    #                 {
    #                     "idx": X.index[va_mask],
    #                     "fold_id": int(fold_id),
    #                     "y_true": yva.astype("int8"),
    #                     "proba": pva.astype("float64"),
    #                 }
    #             )
    #         )
    #     if len(oof_rows) == 0:
    #         raise RuntimeError(f"Unable to build XGBoost OOF predictions for task={task}.")
    #     oof_df = pd.concat(oof_rows, ignore_index=True)
    
    
    y_oof = oof_df["y_true"].to_numpy(dtype="int8")
    p_oof = oof_df["proba"].to_numpy(dtype="float64")
    
    t_star, threshold_attrs = pick_threshold(
        y_oof, p_oof, policy=threshold_policy
    )

    y_pre = y.loc[pretest_mask].to_numpy(dtype="int8")
    y_te = y.loc[test_mask].to_numpy(dtype="int8")
    Xpre, Xte = encode_xgb(X.loc[pretest_mask], X.loc[test_mask], cat_feats)
    spw_final = xgb_scale_pos_weight(y_pre)
    final_model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=cfg.seed,
        n_jobs=-1,
        verbosity=0,
        scale_pos_weight=spw_final,
        **best_cfg,
    )
    final_model.fit(Xpre, y_pre, verbose=False)
    
    ## get predictions from X_test 
    pte = final_model.predict_proba(Xte)[:, 1]
    
    
    
    yhat = (pte >= t_star).astype("int8")

    metrics = {
        "task": task,
        "type": "binary_classifier_timecv_xgboost",
        "cv_n_folds": int(len(folds)),
        "cv_mean_ap": float(best_cv_ap),
        "cv_std_ap": float(best_cv_ap_std),
        "threshold_policy": threshold_policy['name'],
        "threshold_fpr_cap": float(threshold_policy.get("fpr_cap")) if threshold_policy.get("fpr_cap") is not None else np.nan,
        f"threshold_selection_metric_oof_{threshold_policy['name']}": float(threshold_attrs[threshold_policy['name']]),
        "threshold_selection_metric_oof_f1": float(threshold_attrs["f1"]),
        "threshold_selection_metric_oof_fpr": float(threshold_attrs["fpr"]),
        "base_rate_train_pretest": float(y_pre.mean()),
        "train_scale_pos_weight": float(spw_final),
        "base_rate_test": float(y_te.mean()),
        "ap_oof": float(average_precision_score(y_oof, p_oof)),
        "ap_test": float(average_precision_score(y_te, pte)),
        "auc_test": float(roc_auc_score(y_te, pte)),
        "f1_test": float(f1_score(y_te, yhat, zero_division=0)),
        "f2_test": float(fbeta_score(y_te,yhat,beta=2, zero_division=0)),
        "precision_test": float(precision_score(y_te, yhat, zero_division=0)),
        "recall_test": float(recall_score(y_te, yhat, zero_division=0)),
        "accuracy_test": float(accuracy_score(y_te, yhat)),
        "balanced_accuracy_test": float(balanced_accuracy_score(y_te, yhat)),
        "best_threshold": float(t_star),
        "best_cfg": json.dumps(best_cfg),
    }
    pred_df = pd.DataFrame({"proba": pte, "pred": yhat, "y_true": y_te})
    return metrics, final_model, pred_df, oof_df, cfg_scores_df


def build_binary_tasks(incident_df: pd.DataFrame, incident_X: pd.DataFrame, cfg: PipelineConfig) -> list[TaskSpec]:
    task_defs = normalize_task_definitions(list(cfg.risk_task_definitions))
    out: list[TaskSpec] = []
    for d in task_defs:
        task = str(d["name"])
        e_col = eligible_col(task)
        d_col = decision_col(task)
        if task not in incident_df.columns:
            raise KeyError(f"Task target column missing: {task}")
        if e_col not in incident_df.columns:
            raise KeyError(f"Task eligible column missing: {e_col}")
        if d_col not in incident_df.columns:
            raise KeyError(f"Task decision timestamp column missing: {d_col}")

        mask = incident_df[e_col].astype(bool)
        task_df = incident_df.loc[mask].copy()
        task_X = incident_X.loc[mask].copy()
        ts = pd.to_datetime(task_df[d_col], errors="coerce")
        split = {
            "train": ts < cfg.valid_split,
            "valid": (ts >= cfg.valid_split) & (ts < cfg.test_split),
            "test": ts >= cfg.test_split,
        }
        out.append(TaskSpec(task=task, X=task_X, y=task_df[task], ts=ts, split=split))
    return out




def train_xgb_binary_tasks(
    task_specs: list[TaskSpec],
    cat_feats: list[str],
    cfg: PipelineConfig,
    task_threshold_policy: dict[str, dict] | None = None,
    tuning_flag: bool | None = None,
    tuning_mode: str = "bayes_opt",
    tuning_top_k_per_task: int = 4,
    tuning_random_search_iters: int = 30,
) -> tuple[pd.DataFrame, dict[str, xgb.XGBClassifier], dict[str, pd.DataFrame], dict[str, dict[str, pd.DataFrame]]]:
    
    
    
    task_threshold_policy = task_threshold_policy or default_task_threshold_policy()
    rows = []
    models: dict[str, xgb.XGBClassifier] = {}
    predictions: dict[str, pd.DataFrame] = {}
    diagnostics: dict[str, dict[str, pd.DataFrame]] = {}
    
    pre_tuned_params = (
        tuning_cfg_for_xgb_binary_tasks(
            task_specs,
            cat_feats,
            cfg,
            search_mode=tuning_mode,
            top_k_per_task=tuning_top_k_per_task,
            random_search_iters=tuning_random_search_iters,
        )
        if tuning_flag
        else None
    )

    for spec in task_specs:
        policy = _resolve_threshold_policy_for_task(
            y=spec.y,
            cfg=cfg,
            policy=task_threshold_policy.get(spec.task),
        )
        
        metrics, model, pred_df, oof_df, cfg_df = run_binary_classifier_xgb(
            task=spec.task,
            X=spec.X,
            y=spec.y,
            ts=spec.ts,
            split=spec.split,
            cat_feats=cat_feats,
            cfg=cfg,
            threshold_policy=policy,
            pre_tuned_params=pre_tuned_params
        )
        
        rows.append(metrics)
        models[spec.task] = model
        predictions[spec.task] = pred_df
        diagnostics[spec.task] = {"oof_predictions": oof_df, "cfg_scores": cfg_df}

    metrics_df = pd.DataFrame(rows).sort_values("task").reset_index(drop=True)
    return metrics_df, models, predictions, diagnostics




# def _default_lgbm_seed_params() -> list[dict]:
#     return [
#         dict(
#             num_leaves=31,
#             learning_rate=0.05,
#             n_estimators=900,
#             min_child_samples=25,
#             subsample=0.9,
#             colsample_bytree=0.9,
#             reg_lambda=3.0,
#             reg_alpha=0.0,
#         ),
#         dict(
#             num_leaves=63,
#             learning_rate=0.03,
#             n_estimators=1400,
#             min_child_samples=40,
#             subsample=0.9,
#             colsample_bytree=0.85,
#             reg_lambda=4.0,
#             reg_alpha=0.2,
#         ),
#     ]


# def _lgbm_small_grid_candidates(base_cfgs: list[dict]) -> list[dict]:
#     if len(base_cfgs) == 0:
#         base_cfgs = _default_lgbm_seed_params()

#     out: list[dict] = []
#     seen: set[tuple] = set()
#     for base in base_cfgs:
#         cfg0 = copy.deepcopy(base)
#         leaves0 = int(cfg0.get("num_leaves", 31))
#         lr0 = float(cfg0.get("learning_rate", 0.05))
#         n0 = int(cfg0.get("n_estimators", 900))

#         leaves_opts = sorted({max(15, leaves0 - 8), leaves0, min(255, leaves0 + 8)})
#         lr_opts = sorted(
#             {
#                 round(max(0.01, lr0 * 0.85), 6),
#                 round(max(0.01, lr0), 6),
#                 round(min(0.30, lr0 * 1.15), 6),
#             }
#         )
#         n_opts = sorted(
#             {
#                 max(200, int(round((n0 * 0.85) / 25.0) * 25)),
#                 max(200, int(round(n0 / 25.0) * 25)),
#                 max(200, int(round((n0 * 1.15) / 25.0) * 25)),
#             }
#         )

#         for leaves, lr, n_est in product(leaves_opts, lr_opts, n_opts):
#             cfg = copy.deepcopy(cfg0)
#             cfg["num_leaves"] = int(leaves)
#             cfg["learning_rate"] = float(lr)
#             cfg["n_estimators"] = int(n_est)
#             key = _cfg_key(cfg)
#             if key in seen:
#                 continue
#             seen.add(key)
#             out.append(cfg)
#     return out


# def _sample_lgbm_cfg(rng) -> dict:
#     return {
#         "num_leaves": int(rng.integers(15, 128)),
#         "learning_rate": float(np.exp(rng.uniform(np.log(0.01), np.log(0.20)))),
#         "n_estimators": int(rng.integers(400, 2201)),
#         "min_child_samples": int(rng.integers(10, 121)),
#         "subsample": float(rng.uniform(0.65, 1.0)),
#         "colsample_bytree": float(rng.uniform(0.60, 1.0)),
#         "reg_lambda": float(np.exp(rng.uniform(np.log(0.1), np.log(20.0)))),
#         "reg_alpha": float(np.exp(rng.uniform(np.log(1e-4), np.log(5.0)))),
#     }


# def _cv_ap_for_lgbm_cfg(
#     task_spec: TaskSpec,
#     cat_feats: list[str],
#     lgbm_cfg: dict,
#     cfg: PipelineConfig,
# ) -> tuple[float, float, int]:
#     _require_lightgbm()

#     X = task_spec.X
#     y = pd.to_numeric(task_spec.y, errors="coerce").fillna(0).astype("int8")
#     ts = pd.to_datetime(task_spec.ts)
#     split = task_spec.split

#     pretest_mask = pd.Series(split["train"] | split["valid"], index=X.index).astype(bool)
#     folds = build_walkforward_folds(ts=ts, pretest_mask=pretest_mask, cfg=cfg)
#     if len(folds) < 2:
#         folds = [
#             {
#                 "train": pd.Series(split["train"], index=X.index).astype(bool),
#                 "valid": pd.Series(split["valid"], index=X.index).astype(bool),
#                 "valid_start": pd.Timestamp(cfg.valid_split),
#                 "valid_end": pd.Timestamp(cfg.test_split),
#             }
#         ]

#     ap_vals: list[float] = []
#     used = 0
#     for fd in folds:
#         tr_mask = fd["train"]
#         va_mask = fd["valid"]
#         ytr = y.loc[tr_mask].to_numpy(dtype="int8")
#         yva = y.loc[va_mask].to_numpy(dtype="int8")
#         if len(np.unique(ytr)) < 2 or len(np.unique(yva)) < 2:
#             continue

#         Xtr, Xva = encode_xgb(X.loc[tr_mask], X.loc[va_mask], cat_feats)
#         spw = xgb_scale_pos_weight(ytr)

#         m = lgb.LGBMClassifier(
#             objective="binary",
#             random_state=cfg.seed,
#             n_jobs=-1,
#             verbosity=-1,
#             scale_pos_weight=spw,
#             **lgbm_cfg,
#         )
#         _fit_lgbm_with_early_stopping(m, Xtr, ytr, Xva, yva)
#         pva = m.predict_proba(Xva)[:, 1]
#         ap_vals.append(float(average_precision_score(yva, pva)))
#         used += 1

#     if len(ap_vals) == 0:
#         return np.nan, np.nan, 0
#     return float(np.mean(ap_vals)), float(np.std(ap_vals)), int(used)


# def tuning_cfg_for_lgbm_binary_tasks(
#     task_specs: list[TaskSpec],
#     cat_feats: list[str],
#     cfg: PipelineConfig,
#     search_mode: str = "grid_small",
#     top_k_per_task: int = 4,
#     random_search_iters: int = 30,
# ) -> dict:
#     _require_lightgbm()

#     seed_params = _default_lgbm_seed_params()
#     rng = np.random.default_rng(int(cfg.seed) + 4603)
#     rows: list[dict] = []
#     out_by_task: dict[str, list[dict]] = {}

#     for spec in task_specs:
#         if str(search_mode).lower() in {"grid", "grid_small", "small_grid"}:
#             candidates = _lgbm_small_grid_candidates(seed_params)
#         elif str(search_mode).lower() in {"random", "random_search"}:
#             candidates = []
#             seen = set()
#             for base in seed_params:
#                 key = _cfg_key(base)
#                 if key in seen:
#                     continue
#                 seen.add(key)
#                 candidates.append(copy.deepcopy(base))
#             while len(candidates) < (len(seed_params) + int(random_search_iters)):
#                 cand = _sample_lgbm_cfg(rng)
#                 key = _cfg_key(cand)
#                 if key in seen:
#                     continue
#                 seen.add(key)
#                 candidates.append(cand)
#         else:
#             raise ValueError(
#                 f"Unknown search_mode={search_mode!r}. Use 'grid_small' or 'random'."
#             )

#         for cfg_id, lcfg in enumerate(candidates):
#             ap_mean, ap_std, used_folds = _cv_ap_for_lgbm_cfg(spec, cat_feats, lcfg, cfg)
#             if not np.isfinite(ap_mean):
#                 continue
#             rows.append(
#                 {
#                     "task": spec.task,
#                     "cfg_id": int(cfg_id),
#                     "cv_ap_mean": float(ap_mean),
#                     "cv_ap_std": float(ap_std),
#                     "cv_used_folds": int(used_folds),
#                     "cfg": json.dumps(lcfg),
#                 }
#             )

#     if len(rows) == 0:
#         raise RuntimeError("No valid tuning results were produced for LightGBM.")

#     tuning_df = (
#         pd.DataFrame(rows)
#         .sort_values(["task", "cv_ap_mean", "cv_ap_std"], ascending=[True, False, True])
#         .reset_index(drop=True)
#     )
#     for task in tuning_df["task"].unique():
#         topk = tuning_df.loc[tuning_df["task"] == task].head(int(top_k_per_task))
#         out_by_task[task] = [json.loads(s) for s in topk["cfg"].tolist()]
#     return out_by_task


# def run_binary_classifier_lgbm(
#     task: str,
#     X: pd.DataFrame,
#     y: pd.Series,
#     ts: pd.Series,
#     split: dict[str, pd.Series],
#     cat_feats: list[str],
#     cfg: PipelineConfig,
#     threshold_policy: dict,
#     pre_tuned_params: dict | None = None,
# ) -> tuple[dict[str, object], object, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
#     _require_lightgbm()

#     if pre_tuned_params and task in pre_tuned_params:
#         lgbm_params = pre_tuned_params[task]
#     else:
#         lgbm_params = _default_lgbm_seed_params()

#     y = pd.to_numeric(y, errors="coerce").fillna(0).astype("int8")
#     ts = pd.to_datetime(ts)
#     pretest_mask = pd.Series(split["train"] | split["valid"], index=X.index).astype(bool)
#     test_mask = pd.Series(split["test"], index=X.index).astype(bool)

#     folds = build_walkforward_folds(ts=ts, pretest_mask=pretest_mask, cfg=cfg)
#     if len(folds) < 2:
#         folds = [
#             {
#                 "train": pd.Series(split["train"], index=X.index).astype(bool),
#                 "valid": pd.Series(split["valid"], index=X.index).astype(bool),
#                 "valid_start": pd.Timestamp(cfg.valid_split),
#                 "valid_end": pd.Timestamp(cfg.test_split),
#             }
#         ]

#     cfg_rows = []
#     for lcfg in lgbm_params:
#         fold_ap = []
#         used_folds = 0
#         for fd in folds:
#             tr_mask = fd["train"]
#             va_mask = fd["valid"]
#             ytr = y.loc[tr_mask].to_numpy(dtype="int8")
#             yva = y.loc[va_mask].to_numpy(dtype="int8")
#             if len(np.unique(ytr)) < 2 or len(np.unique(yva)) < 2:
#                 continue

#             Xtr, Xva = encode_xgb(X.loc[tr_mask], X.loc[va_mask], cat_feats)
#             spw = xgb_scale_pos_weight(ytr)
#             model = lgb.LGBMClassifier(
#                 objective="binary",
#                 random_state=cfg.seed,
#                 n_jobs=-1,
#                 verbosity=-1,
#                 scale_pos_weight=spw,
#                 **lcfg,
#             )
#             _fit_lgbm_with_early_stopping(model, Xtr, ytr, Xva, yva)
#             pva = model.predict_proba(Xva)[:, 1]
#             fold_ap.append(float(average_precision_score(yva, pva)))
#             used_folds += 1

#         if len(fold_ap) > 0:
#             cfg_rows.append(
#                 {
#                     "task": task,
#                     "cfg": json.dumps(lcfg),
#                     "cv_ap_mean": float(np.mean(fold_ap)),
#                     "cv_ap_std": float(np.std(fold_ap)),
#                     "cv_used_folds": int(used_folds),
#                 }
#             )

#     if len(cfg_rows) == 0:
#         raise RuntimeError(f"No valid LightGBM CV folds for task={task}.")

#     cfg_scores_df = (
#         pd.DataFrame(cfg_rows)
#         .sort_values(["cv_ap_mean", "cv_ap_std"], ascending=[False, True])
#         .reset_index(drop=True)
#     )
#     best_cfg = json.loads(cfg_scores_df.iloc[0]["cfg"])
#     best_cv_ap = float(cfg_scores_df.iloc[0]["cv_ap_mean"])
#     best_cv_ap_std = float(cfg_scores_df.iloc[0]["cv_ap_std"])

#     oof_rows = []
#     for fold_id, fd in enumerate(folds):
#         tr_mask = fd["train"]
#         va_mask = fd["valid"]
#         ytr = y.loc[tr_mask].to_numpy(dtype="int8")
#         yva = y.loc[va_mask].to_numpy(dtype="int8")
#         if len(np.unique(ytr)) < 2 or len(np.unique(yva)) < 2:
#             continue

#         Xtr, Xva = encode_xgb(X.loc[tr_mask], X.loc[va_mask], cat_feats)
#         spw = xgb_scale_pos_weight(ytr)
#         model = lgb.LGBMClassifier(
#             objective="binary",
#             random_state=cfg.seed,
#             n_jobs=-1,
#             verbosity=-1,
#             scale_pos_weight=spw,
#             **best_cfg,
#         )
#         _fit_lgbm_with_early_stopping(model, Xtr, ytr, Xva, yva)
#         pva = model.predict_proba(Xva)[:, 1]
#         oof_rows.append(
#             pd.DataFrame(
#                 {
#                     "idx": X.index[va_mask],
#                     "fold_id": int(fold_id),
#                     "y_true": yva.astype("int8"),
#                     "proba": pva.astype("float64"),
#                 }
#             )
#         )

#     if len(oof_rows) == 0:
#         raise RuntimeError(f"Unable to build LightGBM OOF predictions for task={task}.")

#     oof_df = pd.concat(oof_rows, ignore_index=True)
#     y_oof = oof_df["y_true"].to_numpy(dtype="int8")
#     p_oof = oof_df["proba"].to_numpy(dtype="float64")
#     t_star, threshold_attrs = pick_threshold(y_oof, p_oof, policy=threshold_policy)

#     y_pre = y.loc[pretest_mask].to_numpy(dtype="int8")
#     y_te = y.loc[test_mask].to_numpy(dtype="int8")
#     Xpre, Xte = encode_xgb(X.loc[pretest_mask], X.loc[test_mask], cat_feats)
#     spw_final = xgb_scale_pos_weight(y_pre)
#     final_model = lgb.LGBMClassifier(
#         objective="binary",
#         random_state=cfg.seed,
#         n_jobs=-1,
#         verbosity=-1,
#         scale_pos_weight=spw_final,
#         **best_cfg,
#     )
#     final_model.fit(Xpre, y_pre)
#     pte = final_model.predict_proba(Xte)[:, 1]
#     yhat = (pte >= t_star).astype("int8")

#     metrics = {
#         "task": task,
#         "type": "binary_classifier_timecv_lightgbm",
#         "cv_n_folds": int(len(folds)),
#         "cv_mean_ap": float(best_cv_ap),
#         "cv_std_ap": float(best_cv_ap_std),
#         "threshold_policy": threshold_policy["name"],
#         "threshold_fpr_cap": float(threshold_policy.get("fpr_cap")) if threshold_policy.get("fpr_cap") is not None else np.nan,
#         f"threshold_selection_metric_oof_{threshold_policy['name']}": float(
#             threshold_attrs[threshold_policy["name"]]
#         ),
#         "threshold_selection_metric_oof_f1": float(threshold_attrs["f1"]),
#         "threshold_selection_metric_oof_fpr": float(threshold_attrs["fpr"]),
#         "base_rate_train_pretest": float(y_pre.mean()),
#         "train_scale_pos_weight": float(spw_final),
#         "base_rate_test": float(y_te.mean()),
#         "ap_oof": float(average_precision_score(y_oof, p_oof)),
#         "ap_test": float(average_precision_score(y_te, pte)),
#         "auc_test": float(roc_auc_score(y_te, pte)),
#         "f1_test": float(f1_score(y_te, yhat, zero_division=0)),
#         "f2_test": float(fbeta_score(y_te, yhat, beta=2, zero_division=0)),
#         "precision_test": float(precision_score(y_te, yhat, zero_division=0)),
#         "recall_test": float(recall_score(y_te, yhat, zero_division=0)),
#         "accuracy_test": float(accuracy_score(y_te, yhat)),
#         "balanced_accuracy_test": float(balanced_accuracy_score(y_te, yhat)),
#         "best_threshold": float(t_star),
#         "best_cfg": json.dumps(best_cfg),
#     }
#     pred_df = pd.DataFrame({"proba": pte, "pred": yhat, "y_true": y_te})
#     return metrics, final_model, pred_df, oof_df, cfg_scores_df


# def train_lgbm_binary_tasks(
#     task_specs: list[TaskSpec],
#     cat_feats: list[str],
#     cfg: PipelineConfig,
#     task_threshold_policy: dict[str, dict] | None = None,
#     tuning_flag: bool | None = None,
#     tuning_mode: str = "grid_small",
#     tuning_top_k_per_task: int = 4,
#     tuning_random_search_iters: int = 30,
# ) -> tuple[pd.DataFrame, dict[str, object], dict[str, pd.DataFrame], dict[str, dict[str, pd.DataFrame]]]:
#     _require_lightgbm()

#     task_threshold_policy = task_threshold_policy or default_task_threshold_policy()
#     rows = []
#     models: dict[str, object] = {}
#     predictions: dict[str, pd.DataFrame] = {}
#     diagnostics: dict[str, dict[str, pd.DataFrame]] = {}

#     pre_tuned_params = (
#         tuning_cfg_for_lgbm_binary_tasks(
#             task_specs,
#             cat_feats,
#             cfg,
#             search_mode=tuning_mode,
#             top_k_per_task=tuning_top_k_per_task,
#             random_search_iters=tuning_random_search_iters,
#         )
#         if tuning_flag
#         else None
#     )

#     for spec in task_specs:
#         policy = _resolve_threshold_policy_for_task(
#             y=spec.y,
#             cfg=cfg,
#             policy=task_threshold_policy.get(spec.task),
#         )

#         metrics, model, pred_df, oof_df, cfg_df = run_binary_classifier_lgbm(
#             task=spec.task,
#             X=spec.X,
#             y=spec.y,
#             ts=spec.ts,
#             split=spec.split,
#             cat_feats=cat_feats,
#             cfg=cfg,
#             threshold_policy=policy,
#             pre_tuned_params=pre_tuned_params,
#         )
#         rows.append(metrics)
#         models[spec.task] = model
#         predictions[spec.task] = pred_df
#         diagnostics[spec.task] = {"oof_predictions": oof_df, "cfg_scores": cfg_df}

#     metrics_df = pd.DataFrame(rows).sort_values("task").reset_index(drop=True)
#     return metrics_df, models, predictions, diagnostics


def tuning_cfg_for_xgb_binary_tasks(
    task_specs: list[TaskSpec],
    cat_feats: list[str],
    cfg: PipelineConfig,
    search_mode: str = "bayes_opt",
    top_k_per_task: int = 4,
    random_search_iters: int = 30,
) -> dict :
    XGB_TUNING_SEED = int(cfg.seed ) + 2603
    rng = np.random.default_rng(XGB_TUNING_SEED)
    xgb_tuning_rows = []
    xgb_params_by_task = {}

    for task_idx, spec in enumerate(task_specs):
        task = spec.task
        mode = str(search_mode).lower()

        if mode in {"grid", "grid_small", "small_grid"}:
            candidate_cfgs = _xgb_small_grid_candidates(base_cfgs=cfg.xgb_params)
        elif mode in {"random", "random_search"}:
            # Start from existing hand-picked configs + random proposals.
            candidate_cfgs = []
            seen = set()
            for xgb_cfg in cfg.xgb_params:
                key = _cfg_key(xgb_cfg)
                if key not in seen:
                    seen.add(key)
                    candidate_cfgs.append(copy.deepcopy(xgb_cfg))

            while len(candidate_cfgs) < (len(cfg.xgb_params) + int(random_search_iters)):
                xgb_cfg = _sample_xgb_cfg(rng)
                key = _cfg_key(xgb_cfg)
                if key in seen:
                    continue
                seen.add(key)
                candidate_cfgs.append(xgb_cfg)
        elif mode in {"bayes", "bayes_opt", "bayesian", "bayesian_optimization"}:
            bayes_rows = _bayes_opt_xgb_candidates_for_task(
                task_spec=spec,
                cat_feats=cat_feats,
                cfg=cfg,
                random_state=(XGB_TUNING_SEED + 97 * int(task_idx + 1)),
                n_iter=max(8, int(random_search_iters)),
                warm_start_cfgs=cfg.xgb_params,
            )
            for xgb_cfg_id, row in enumerate(bayes_rows):
                xgb_tuning_rows.append(
                    {
                        "task": task,
                        "cfg_id": int(xgb_cfg_id),
                        "cv_ap_mean": float(row["cv_ap_mean"]),
                        "cv_ap_std": float(row["cv_ap_std"]),
                        "cv_used_folds": int(row["cv_used_folds"]),
                        "cfg": json.dumps(row["cfg"]),
                    }
                )
            continue
        else:
            raise ValueError(
                f"Unknown search_mode={search_mode!r}. "
                "Use 'bayes_opt', 'random', or 'grid_small'."
            )

        for xgb_cfg_id, xgb_cfg in enumerate(candidate_cfgs):
            ap_mean, ap_std, used_folds = _cv_ap_for_cfg(spec, cat_feats, xgb_cfg, cfg)
            if not np.isfinite(ap_mean):
                continue

            xgb_tuning_rows.append({
                "task": task,
                "cfg_id": int(xgb_cfg_id),
                "cv_ap_mean": float(ap_mean),
                "cv_ap_std": float(ap_std),
                "cv_used_folds": int(used_folds),
                "cfg": json.dumps(xgb_cfg),
            })

    if len(xgb_tuning_rows) == 0:
        raise RuntimeError("No valid tuning results were produced for XGBoost.")

    xgb_tuning_df = (
        pd.DataFrame(xgb_tuning_rows)
        .sort_values(["task", "cv_ap_mean", "cv_ap_std"], ascending=[True, False, True])
        .reset_index(drop=True)
    )


    for task in xgb_tuning_df["task"].unique():
        topk = xgb_tuning_df.loc[xgb_tuning_df["task"] == task].head(int(top_k_per_task))
        xgb_params_by_task[task] = [json.loads(s) for s in topk["cfg"].tolist()]
    
    return xgb_params_by_task



def _sample_xgb_cfg(rng):
    return {
        "max_depth": int(rng.integers(4, 11)),
        "learning_rate": float(np.exp(rng.uniform(np.log(0.015), np.log(0.18)))),
        "n_estimators": int(rng.integers(600, 2201)),
        "min_child_weight": int(rng.integers(1, 13)),
        "subsample": float(rng.uniform(0.65, 1.0)),
        "colsample_bytree": float(rng.uniform(0.60, 1.0)),
        "gamma": float(rng.uniform(0.0, 3.0)),
        "reg_lambda": float(np.exp(rng.uniform(np.log(0.1), np.log(20.0)))),
        "reg_alpha": float(np.exp(rng.uniform(np.log(1e-4), np.log(5.0)))),
    }


def _xgb_cfg_from_bayes_params(params: dict[str, float]) -> dict:
    # BayesianOptimization works over continuous ranges; project to valid XGB types/ranges.
    return {
        "max_depth": int(np.clip(np.round(params["max_depth"]), 3, 14)),
        "learning_rate": float(
            np.clip(np.exp(params["learning_rate_log"]), 0.01, 0.30)
        ),
        "n_estimators": int(
            np.clip(np.round(params["n_estimators"] / 25.0) * 25, 200, 2500)
        ),
        "min_child_weight": int(np.clip(np.round(params["min_child_weight"]), 1, 20)),
        "subsample": float(np.clip(params["subsample"], 0.60, 1.00)),
        "colsample_bytree": float(np.clip(params["colsample_bytree"], 0.60, 1.00)),
        "gamma": float(np.clip(params["gamma"], 0.0, 5.0)),
        "reg_lambda": float(np.clip(np.exp(params["reg_lambda_log"]), 0.05, 50.0)),
        "reg_alpha": float(np.clip(np.exp(params["reg_alpha_log"]), 1e-6, 10.0)),
    }


def _bayes_opt_xgb_candidates_for_task(
    task_spec: TaskSpec,
    cat_feats: list[str],
    cfg: PipelineConfig,
    random_state: int,
    n_iter: int,
    warm_start_cfgs: list[dict] | None = None,
) -> list[dict]:
    _require_bayes_opt()

    warm_start_cfgs = warm_start_cfgs or []
    evaluated: dict[tuple, dict] = {}

    def _record_eval(xgb_cfg: dict, ap_mean: float, ap_std: float, used_folds: int) -> None:
        key = _cfg_key(xgb_cfg)
        prev = evaluated.get(key)
        row = {
            "cfg": copy.deepcopy(xgb_cfg),
            "cv_ap_mean": float(ap_mean),
            "cv_ap_std": float(ap_std),
            "cv_used_folds": int(used_folds),
        }
        if prev is None:
            evaluated[key] = row
            return
        if (row["cv_ap_mean"], -row["cv_ap_std"]) > (prev["cv_ap_mean"], -prev["cv_ap_std"]):
            evaluated[key] = row

    # Warm start with existing hand-picked configs from cfg.xgb_params.
    for base_cfg in warm_start_cfgs:
        xgb_cfg = copy.deepcopy(base_cfg)
        ap_mean, ap_std, used_folds = _cv_ap_for_cfg(task_spec, cat_feats, xgb_cfg, cfg)
        if np.isfinite(ap_mean):
            _record_eval(xgb_cfg, ap_mean, ap_std, used_folds)

    pbounds = {
        "max_depth": (4.0, 10.0),
        "learning_rate_log": (float(np.log(0.015)), float(np.log(0.18))),
        "n_estimators": (600.0, 2200.0),
        "min_child_weight": (1.0, 12.0),
        "subsample": (0.65, 1.00),
        "colsample_bytree": (0.60, 1.00),
        "gamma": (0.0, 3.0),
        "reg_lambda_log": (float(np.log(0.1)), float(np.log(20.0))),
        "reg_alpha_log": (float(np.log(1e-4)), float(np.log(5.0))),
    }

    def objective(
        max_depth,
        learning_rate_log,
        n_estimators,
        min_child_weight,
        subsample,
        colsample_bytree,
        gamma,
        reg_lambda_log,
        reg_alpha_log,
    ):
        params = {
            "max_depth": float(max_depth),
            "learning_rate_log": float(learning_rate_log),
            "n_estimators": float(n_estimators),
            "min_child_weight": float(min_child_weight),
            "subsample": float(subsample),
            "colsample_bytree": float(colsample_bytree),
            "gamma": float(gamma),
            "reg_lambda_log": float(reg_lambda_log),
            "reg_alpha_log": float(reg_alpha_log),
        }
        xgb_cfg = _xgb_cfg_from_bayes_params(params)
        key = _cfg_key(xgb_cfg)

        if key in evaluated:
            return float(evaluated[key]["cv_ap_mean"])

        ap_mean, ap_std, used_folds = _cv_ap_for_cfg(task_spec, cat_feats, xgb_cfg, cfg)
        if not np.isfinite(ap_mean):
            return 0.0
        _record_eval(xgb_cfg, ap_mean, ap_std, used_folds)
        return float(ap_mean)

    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        random_state=int(random_state),
        verbose=0,
    )
    optimizer.maximize(init_points=max(5, len(warm_start_cfgs)), n_iter=max(1, int(n_iter)))

    out = list(evaluated.values())
    out.sort(key=lambda r: (-float(r["cv_ap_mean"]), float(r["cv_ap_std"])))
    return out


def _xgb_small_grid_candidates(base_cfgs: list[dict]) -> list[dict]:
    # Small local grid around known-good params from cfg.xgb_params.
    if len(base_cfgs) == 0:
        base_cfgs = [dict(max_depth=6, learning_rate=0.05, n_estimators=900)]

    out: list[dict] = []
    seen: set[tuple] = set()

    for base in base_cfgs:
        cfg0 = copy.deepcopy(base)
        d0 = int(cfg0.get("max_depth", 6))
        lr0 = float(cfg0.get("learning_rate", 0.05))
        n0 = int(cfg0.get("n_estimators", 900))

        depth_opts = sorted({max(3, d0 - 1), d0, min(14, d0 + 1)})
        lr_opts = sorted(
            {
                round(max(0.01, lr0 * 0.85), 6),
                round(max(0.01, lr0), 6),
                round(min(0.30, lr0 * 1.15), 6),
            }
        )
        n_opts = sorted(
            {
                max(200, int(round((n0 * 0.85) / 25.0) * 25)),
                max(200, int(round(n0 / 25.0) * 25)),
                max(200, int(round((n0 * 1.15) / 25.0) * 25)),
            }
        )

        for d, lr, n_est in product(depth_opts, lr_opts, n_opts):
            cfg = copy.deepcopy(cfg0)
            cfg["max_depth"] = int(d)
            cfg["learning_rate"] = float(lr)
            cfg["n_estimators"] = int(n_est)
            key = _cfg_key(cfg)
            if key in seen:
                continue
            seen.add(key)
            out.append(cfg)

    return out


def _cfg_key(cfg):
    norm = []
    for k in sorted(cfg.keys()):
        v = cfg[k]
        if isinstance(v, float):
            v = round(v, 8)
        norm.append((k, v))
    return tuple(norm)


def _cv_ap_for_cfg(
    task_spec,
    cat_feats,
    xgb_cfg,
    cfg: PipelineConfig,
                   ):
    
    X = task_spec.X
    y = task_spec.y
    ts = task_spec.ts
    split = task_spec.split
    
    y = pd.to_numeric(y, errors="coerce").fillna(0).astype("int8")
    ts = pd.to_datetime(ts)

    pretest_mask = pd.Series(split["train"] | split["valid"], index=X.index).astype(bool)
    folds = build_walkforward_folds(ts=ts, pretest_mask=pretest_mask, cfg=cfg)

    if len(folds) < 2:
        folds = [
            {
                "train": pd.Series(split["train"], index=X.index).astype(bool),
                "valid": pd.Series(split["valid"], index=X.index).astype(bool),
                "valid_start": pd.Timestamp(cfg.valid_split),
                "valid_end": pd.Timestamp(cfg.test_split),
            }
        ]

    ap_vals = []
    used = 0

    for fd in folds:
        tr_mask = fd["train"]
        va_mask = fd["valid"]

        ytr = y.loc[tr_mask].to_numpy(dtype="int8")
        yva = y.loc[va_mask].to_numpy(dtype="int8")
        if len(np.unique(ytr)) < 2 or len(np.unique(yva)) < 2:
            continue

        Xtr, Xva = encode_xgb(X.loc[tr_mask], X.loc[va_mask], cat_feats)
        spw = xgb_scale_pos_weight(ytr)

        m = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state= cfg.seed,
            n_jobs=-1,
            verbosity=0,
            early_stopping_rounds=80,
            scale_pos_weight=spw,
            **xgb_cfg,
        )
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        pva = m.predict_proba(Xva)[:, 1]
        ap_vals.append(float(average_precision_score(yva, pva)))
        used += 1

    if len(ap_vals) == 0:
        return np.nan, np.nan, 0

    return float(np.mean(ap_vals)), float(np.std(ap_vals)), int(used)
