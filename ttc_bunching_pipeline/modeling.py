from __future__ import annotations

import json
from dataclasses import dataclass
import copy

import numpy as np
import pandas as pd
import xgboost as xgb

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
    
#     return {
#     "observed1_next2_gap1_binary": "f2_fprcap30",
#     "observed2_next2_gap1_binary": "f2_fprcap30",
#     "observed2_next3_gap1_binary": "f2_fprcap30",
#     "observed3_next2_gap1_binary": "f2_fprcap30",
#     "observed3_next3_gap1_binary": "f2_fprcap30",
#     "observed3_next4_gap1_binary": "f2_fprcap30",
#     "observed1_next1_binary": "f2",
#     "observed2_next1_binary": "f2",
#     "observed2_next2_binary": "f2",
#     "observed3_next2_binary": "f2",
#     "observed3_next3_binary": "f3_fprcap30",
    
#     "cond3of4_to_next4_ge2_binary": "f2_fprcap30",
#     "cond3of5_to_next4_ge2_binary": "f2_fprcap30",
#     "cond3of5_to_next5_ge3_binary": "f2_fprcap30",
# }








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





# {
#         "observed1_next2_gap1_binary": "f1_fprcap40",
#         "observed2_next2_gap1_binary": "f1_fprcap40",
#         "observed2_next3_gap1_binary": "f1_fprcap40",
#         "observed3_next2_gap1_binary": "f1_fprcap40",
#         "observed3_next3_gap1_binary": "f1_fprcap40",
#         "observed3_next4_gap1_binary": "f1_fprcap40",
#         "cond3of4_to_next4_ge2_binary": "f1_fprcap40",
#         "cond3of5_to_next4_ge2_binary": "f1_fprcap40",
#         "cond3of5_to_next5_ge3_binary": "f1_fprcap40",
#         "observed1_next1_binary": "balacc",
#         "observed2_next1_binary": "balacc",
#         "observed2_next2_binary": "f1",
#         "observed3_next2_binary": "f1",
#         "observed3_next3_binary": "f1_fprcap40",
#     }


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

### threshold tuning does not require retraining.. 
### 
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
    pos = float(np.sum(y_arr == 1))
    neg = float(np.sum(y_arr == 0))
    if pos <= 0:
        return 1.0
    
    ### upper bound on scale factor? 8 means 80% negative class, 20% positive class 
    return float(min(8.0, max(1.0, neg / pos)))


def encode_xgb(train_df: pd.DataFrame, other_df: pd.DataFrame, cat_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    cat_cols = [c for c in cat_cols if c in train_df.columns]
    tr = pd.get_dummies(train_df.copy(), columns=cat_cols, dummy_na=True)
    ot = pd.get_dummies(other_df.copy(), columns=cat_cols, dummy_na=True)
    ot = ot.reindex(columns=tr.columns, fill_value=0)
    tr = tr.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype("float32")
    ot = ot.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype("float32")
    return tr, ot


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
    for xcfg in xgb_params:
        fold_ap = []
        used_folds = 0
        for fd in folds:
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
                eval_metric="auc",
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
    if len(cfg_rows) == 0:
        raise RuntimeError(f"No valid XGBoost CV folds for task={task}.")
    cfg_scores_df = pd.DataFrame(cfg_rows).sort_values(["cv_ap_mean", "cv_ap_std"], ascending=[False, True]).reset_index(drop=True)
    best_cfg = json.loads(cfg_scores_df.iloc[0]["cfg"])
    best_cv_ap = float(cfg_scores_df.iloc[0]["cv_ap_mean"])
    best_cv_ap_std = float(cfg_scores_df.iloc[0]["cv_ap_std"])

    
    

    ## OOF threshold tuning      
    oof_rows = []
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
            eval_metric="auc",
            tree_method="hist",
            random_state=cfg.seed,
            n_jobs=-1,
            verbosity=0,
            early_stopping_rounds=80,
            scale_pos_weight=spw,
            **best_cfg,
        )
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        pva = model.predict_proba(Xva)[:, 1]
        
        oof_rows.append(
            pd.DataFrame(
                {
                    "idx": X.index[va_mask],
                    "fold_id": int(fold_id),
                    "y_true": yva.astype("int8"),
                    "proba": pva.astype("float64"),
                }
            )
        )
    if len(oof_rows) == 0:
        raise RuntimeError(f"Unable to build XGBoost OOF predictions for task={task}.")
    oof_df = pd.concat(oof_rows, ignore_index=True)
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
        eval_metric="auc",
        tree_method="hist",
        random_state=cfg.seed,
        n_jobs=-1,
        verbosity=0,
        scale_pos_weight=spw_final,
        **best_cfg,
    )
    final_model.fit(Xpre, y_pre, verbose=False)
    pte = final_model.predict_proba(Xte)[:, 1]
    yhat = (pte >= t_star).astype("int8")

    metrics = {
        "task": task,
        "type": "binary_classifier_timecv_xgboost",
        "cv_n_folds": int(len(folds)),
        "cv_mean_ap": float(best_cv_ap),
        "cv_std_ap": float(best_cv_ap_std),
        "threshold_policy": threshold_policy['name'],
        f"threshold_selection_metric_oof_{threshold_policy['name']}": float(threshold_attrs[threshold_policy['name']]),
        "threshold_selection_metric_oof_f1": float(threshold_attrs["f1"]),
        "threshold_selection_metric_oof_fpr": float(threshold_attrs["fpr"]),
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



### make a version of this with hyperparam tuning?

def train_xgb_binary_tasks(
    task_specs: list[TaskSpec],
    cat_feats: list[str],
    cfg: PipelineConfig,
    task_threshold_policy: dict[str, str] | None = None,
    tuning_flag: bool | None = None
) -> tuple[pd.DataFrame, dict[str, xgb.XGBClassifier], dict[str, pd.DataFrame], dict[str, dict[str, pd.DataFrame]]]:
    
    
    
    task_threshold_policy = task_threshold_policy or default_task_threshold_policy()
    rows = []
    models: dict[str, xgb.XGBClassifier] = {}
    predictions: dict[str, pd.DataFrame] = {}
    diagnostics: dict[str, dict[str, pd.DataFrame]] = {}
    
    pre_tuned_params = tuning_cfg_for_xgb_binary_tasks( task_specs,cat_feats, cfg ) if tuning_flag else None 

    for spec in task_specs:
        
        policy = task_threshold_policy.get(spec.task)
        
        
        ## roll-back to default policy if no policy is specified
        if not policy:
            policy =  {'name': cfg.threshold_policy_global_default, 'fpr_cap': cfg.threshold_fpr_cap_global_default } 
        
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




def tuning_cfg_for_xgb_binary_tasks(
    task_specs: list[TaskSpec],
    cat_feats: list[str],
    cfg: PipelineConfig,
) -> dict :
   
   
    XGB_RANDOM_SEARCH_ITERS = 30
    XGB_TOP_K_PER_TASK = 4
    XGB_TUNING_SEED = int(cfg.seed ) + 2603
    rng = np.random.default_rng(XGB_TUNING_SEED)
    xgb_tuning_rows = []
    xgb_params_by_task = {}

    for spec in task_specs:
        task = spec.task
        # Start from existing hand-picked configs + random proposals.
        candidate_cfgs = []
        seen = set()

        for xgb_cfg in cfg.xgb_params:
            key = _cfg_key(xgb_cfg)
            if key not in seen:
                seen.add(key)
                candidate_cfgs.append(copy.deepcopy(xgb_cfg))

        while len(candidate_cfgs) < (len(cfg.xgb_params) + int(XGB_RANDOM_SEARCH_ITERS)):
            xgb_cfg = _sample_xgb_cfg(rng)
            key = _cfg_key(xgb_cfg)
            if key in seen:
                continue
            seen.add(key)
            candidate_cfgs.append(xgb_cfg)

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
        topk = xgb_tuning_df.loc[xgb_tuning_df["task"] == task].head(int(XGB_TOP_K_PER_TASK))
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
            eval_metric="auc",
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
