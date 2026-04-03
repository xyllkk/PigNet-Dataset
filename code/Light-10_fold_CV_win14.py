# -*- coding: utf-8 -*-
"""
LightGBM（天粒度，固定滑窗 W=7 → H=7，多输出 MIMO）
流程：先全局 5 折（按猪 GroupKFold）调参选最优；再用最优参数做 10 折（按猪 GroupKFold）交叉验证评估。
口径：逐猪等权 —— 先猪内 7 天（h1..h7）RMSE 均值，再在猪之间做宏平均；不按窗口/样本量加权。

输出 Excel（仅 Global）：
  - 每折工作表：Fold{i}（逐猪 Train/Val 明细，含 RMSE_h1..h7、RMSE=7天均值、R2=等权均值）
  - Fold{i}__summary_train / Fold{i}__summary_val / Fold{i}__val_means
  - （新增）Fold{i}__shap_val_hwise / Fold{i}__shap_val_family（SHAP 解释）
  - CV10_ALL（合并全部折的逐猪明细）
  - MacroOverall_Val / MacroHorizon_Val（基于全部折的 Val 逐猪明细）
  - STABILITY_Val（逐折 Val 宏平均 RMSE 列表 + 折间 mean/std/95%CI/CV）
  - splits_info（每折的猪ID划分）
  - tuning_info（5折调参日志 + 最优参数）
  - （新增）GLOBAL_MEANS（Train/Val 的“所有均值（总的）”汇总）

依赖：
  pip install lightgbm scikit-learn pandas numpy xlsxwriter shap
"""

from pathlib import Path
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor
import shap  # SHAP 解释

# =========================
# 手动参数区（只改这里）
# =========================
DATA_PATH = r"E:\文献\课题\AI\猪场实验数据\输入_采食加环境加日龄_按天"
OUT_EXCEL = r"E:\文献\课题\AI\猪场实验数据\正式试验结果\最终结果\win14\LightGBM_10_fold_CV_win14"  # 生成 _W7_H7.xlsx（沿用原命名）

DATE_COL    = "日期"
PIG_COL     = "耳缺号"
WEIGHT_COL  = "体重"
BREED_COL   = None
STATION_COL = None

# 固定窗口：7 -> 7
WINDOW  = 14
HORIZON = 7

# 可选：剔除特征关键字（保持空表示不剔除）
EXCLUDE_FEATURE_KEYWORDS = []

# ======== LightGBM 基础与候选（5折调参用） ========
USE_GPU = False  # 若 LightGBM 为 GPU 版本可设 True（将置 device="gpu"）
BASE_LGB_PARAMS = dict(
    objective="regression",
    random_state=42,
    n_estimators=1000, learning_rate=0.05, max_depth=6, num_leaves=64,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, min_child_samples=1,
    n_jobs=-1, verbose=-1
)
# 将你原 XGBoost 的 4 组候选网格做“语义映射”：
# depth 4/6/8 → num_leaves ≈ 2^depth（16/64/256）
PARAM_CANDIDATES: List[dict] = [
    dict(n_estimators=800,  learning_rate=0.05, max_depth=4, num_leaves=16,  subsample=0.9, colsample_bytree=0.9, min_child_samples=1, reg_lambda=1.0),
    dict(n_estimators=1000, learning_rate=0.05, max_depth=6, num_leaves=64,  subsample=0.8, colsample_bytree=0.8, min_child_samples=1, reg_lambda=1.0),
    dict(n_estimators=1200, learning_rate=0.03, max_depth=6, num_leaves=64,  subsample=0.9, colsample_bytree=0.9, min_child_samples=1, reg_lambda=1.0),
    dict(n_estimators=800,  learning_rate=0.05, max_depth=8, num_leaves=256, subsample=0.8, colsample_bytree=0.8, min_child_samples=3, reg_lambda=1.5),
]
INNER_FOLDS_TUNE = 5
OUTER_FOLDS_EVAL = 10
SEED = 42

# ======== 宏平均统计（95%CI + CV） ========
BOOTSTRAP_B = 2000
CI_ALPHA = 0.05
_EPS = 1e-12

# ======== SHAP 分析设置（可选） ========
DO_SHAP = True           # 是否在每折对 Val 计算 SHAP
SHAP_MAX_PER_PIG = 20    # 每头猪最多取多少个窗口样本做 SHAP（控制计算量）

# ========== I/O ==========
def load_daily_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.is_dir():
        parts = []
        for p in sorted(list(path.glob("*.xlsx")) + list(path.glob("*.xls")) + list(path.glob("*.csv"))):
            try:
                df = pd.read_excel(p) if p.suffix.lower() in [".xls", ".xlsx"] else pd.read_csv(p)
            except Exception:
                continue
            df["__source__"] = p.name
            parts.append(df)
        if not parts:
            raise RuntimeError("目录内没有可读的 xlsx/csv 文件。")
        df = pd.concat(parts, ignore_index=True)
    else:
        df = pd.read_excel(path) if str(path).lower().endswith((".xls", ".xlsx")) else pd.read_csv(path)
        df["__source__"] = Path(path).name
    return df

# ========== 预处理 ==========
def normalize_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, str, str]:
    for c, nm in [(DATE_COL, "日期"), (PIG_COL, "猪ID"), (WEIGHT_COL, "体重")]:
        if c not in df.columns:
            raise ValueError(f"缺少 {nm} 列：{c}")

    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df[PIG_COL] = df[PIG_COL].astype(str)
    if "日龄" in df.columns:
        df["日龄"] = pd.to_numeric(df["日龄"], errors="coerce")

    # 采食时长转秒
    for tcol in ["采食时间", "采食时长", "总采食时长"]:
        if tcol in df.columns:
            s = df[tcol]
            if pd.api.types.is_numeric_dtype(s):
                df[tcol + "_s"] = s.astype(float) * 86400.0
            elif np.issubdtype(s.dtype, np.datetime64):
                ss = pd.to_datetime(s, errors="coerce")
                df[tcol + "_s"] = (ss.dt.hour * 3600 + ss.dt.minute * 60 + ss.dt.second).astype(float)
            else:
                ss = s.astype(str).str.replace("：", ":", regex=False).str.strip()
                td = pd.to_timedelta(ss, errors="coerce")
                df[tcol + "_s"] = td.dt.total_seconds()

    # 品种列
    bcol = BREED_COL or "_breed_"
    if BREED_COL and BREED_COL in df.columns:
        df[bcol] = df[BREED_COL].astype(str)
    else:
        def derive_breed(x: str) -> str:
            m = re.match(r'^[A-Za-z\u4e00-\u9fa5]+', str(x))
            return m.group(0) if m else "UNK"
        df[bcol] = df[PIG_COL].map(derive_breed)

    # 测定站列
    def station_from_pid(pid: str) -> str | None:
        m = re.search(r'(\d+)[\-_]\d+', str(pid))
        return f"S{m.group(1)}" if m else None
    def station_from_source(src: str) -> str | None:
        s = str(src)
        m = re.search(r'(\d+)[\-_]\d+', s)
        if m: return f"S{m.group(1)}"
        ss = s.lower()
        m = re.search(r'(站|room|pen|house|测定站)\s*([0-9一二三四五六七八九]+)', ss)
        return f"S{m.group(2)}" if m else None

    scol = STATION_COL or "_station_"
    if STATION_COL and STATION_COL in df.columns:
        df[scol] = df[STATION_COL].astype(str)
    else:
        st_pid = df[PIG_COL].map(station_from_pid)
        st_src = df["__source__"].map(station_from_source) if "__source__" in df.columns else None
        df[scol] = st_pid.fillna(st_src) if st_src is not None else st_pid
        df[scol] = df[scol].fillna("S?")

    df = (df.drop_duplicates(subset=[DATE_COL, PIG_COL])
            .sort_values([PIG_COL, DATE_COL])
            .reset_index(drop=True))
    return df, bcol, scol

# ========== 特征 ==========
def pick_features(df: pd.DataFrame, bcol: str, scol: str) -> List[str]:
    drop = {DATE_COL, PIG_COL, WEIGHT_COL, bcol, scol, "__source__"}
    cand = [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]

    def _deny(cname: str) -> bool:
        name_l = str(cname).lower()
        return any(k.lower() in name_l for k in EXCLUDE_FEATURE_KEYWORDS)
    cand = [c for c in cand if not _deny(c)]
    if not cand:
        raise ValueError("未找到可用的数值特征列。")

    pri_order = [
        "采食量","采食量(kg)","总采食量","采食次数",
        "采食时间_s","采食时长_s","总采食时长_s",
        "采食时间","采食时长","总采食时长",
        "日龄"
    ]
    pri = [c for c in pri_order if c in cand]
    rest = [c for c in cand if c not in pri]
    return pri + rest

# ---------- 造样本（多输出 H=1..7） ----------
def make_samples_one_pig_sliding_multiH(g: pd.DataFrame, feat_cols: List[str], window: int, horizon: int):
    """
    返回：
      X: pd.DataFrame (n_win, F*window + 1)  # +1=init_weight
      Y: pd.DataFrame (n_win, H)             # 列 y_h1..y_h7
    """
    Xs, Ys = [], []
    if len(g) < window + horizon:
        return pd.DataFrame(), pd.DataFrame()

    init_series = g[WEIGHT_COL].dropna()
    if init_series.empty:
        return pd.DataFrame(), pd.DataFrame()
    init_w = float(init_series.iloc[0])

    for t in range(window - 1, len(g) - horizon):
        w = g.iloc[t - window + 1:t + 1]
        row = {}
        for c in feat_cols:
            s = pd.to_numeric(w[c], errors="coerce")
            for lag in range(1, window + 1):
                row[f"{c}_lag{lag}"] = float(s.iloc[-lag]) if pd.notnull(s.iloc[-lag]) else np.nan
        row["init_weight"] = init_w
        Xs.append(row)
        Ys.append([float(g[WEIGHT_COL].iloc[t + h]) for h in range(1, horizon + 1)])

    X = pd.DataFrame(Xs)
    Y = pd.DataFrame(Ys, columns=[f"y_h{h}" for h in range(1, horizon + 1)])
    return X, Y

def build_all_samples(df: pd.DataFrame, bcol: str, scol: str, feat_cols: List[str], window: int, horizon: int):
    out = []
    for pid, g in df.groupby(PIG_COL, sort=False):
        g = g[[DATE_COL] + feat_cols + [WEIGHT_COL, bcol, scol]].sort_values(DATE_COL).reset_index(drop=True)
        X, Y = make_samples_one_pig_sliding_multiH(g, feat_cols, window, horizon)
        if len(X):
            out.append(dict(pid=str(pid), breed=str(g[bcol].iloc[0]), station=str(g[scol].iloc[0]), X=X, Y=Y))
    return out

def build_Xy_groups(samples_list: List[dict]):
    X_parts, Y_parts, groups = [], [], []
    for s in samples_list:
        if len(s["X"]) == 0:
            continue
        X_parts.append(s["X"])
        Y_parts.append(s["Y"])
        groups.extend([s["pid"]] * len(s["X"]))
    if not X_parts:
        return pd.DataFrame(), pd.DataFrame(), np.array([])
    X = pd.concat(X_parts, ignore_index=True)
    Y = pd.concat(Y_parts, ignore_index=True)
    return X, Y, np.array(groups, dtype=object)

# ---------- 基础工具 ----------
def rmse_safe(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def _bootstrap_ci_mean(values, B=BOOTSTRAP_B, alpha=CI_ALPHA, seed=1234):
    vals = np.asarray(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = []
    n = len(vals)
    for _ in range(max(50, B)):
        idx = rng.integers(0, n, size=n)
        boots.append(np.mean(vals[idx]))
    lo = float(np.percentile(boots, 100*alpha/2))
    hi = float(np.percentile(boots, 100*(1-alpha/2)))
    return float(np.mean(vals)), float(np.std(vals, ddof=1)), lo, hi

def _cv_ratio(mean_val, std_val):
    if mean_val is None or np.isnan(mean_val) or abs(mean_val) < _EPS:
        return np.nan
    return float(std_val / (mean_val + _EPS))

# ---------- 逐猪评估（等权口径） ----------
def eval_by_pig_named(samples_list: List[dict], yhat_list: List[np.ndarray], set_name: str, horizon: int) -> pd.DataFrame:
    rows = []
    for s, Yhat in zip(samples_list, yhat_list):
        Y = s["Y"].to_numpy()  # (n_win, H)
        if Y.size == 0:
            continue
        y_pred = np.asarray(Yhat)
        n_win = Y.shape[0]

        rmse_h_list, r2_h_list = [], []
        rec = dict(集合=set_name, 猪ID=s["pid"], 品种=s["breed"], 测定站=s["station"],
                   n_win=int(n_win), n=int(n_win * horizon))

        for h in range(1, horizon + 1):
            yt = Y[:, h-1]
            yp = y_pred[:, h-1]
            rmse_h = rmse_safe(yt, yp)
            r2_h   = r2_score(yt, yp) if len(np.unique(yt)) > 1 else np.nan
            rec[f"RMSE_h{h}"] = rmse_h
            rec[f"R2_h{h}"]   = r2_h
            rmse_h_list.append(rmse_h); r2_h_list.append(r2_h)

        rec["RMSE"] = float(np.nanmean(rmse_h_list)) if rmse_h_list else np.nan
        rec["R2"]   = float(np.nanmean([v for v in r2_h_list if pd.notnull(v)])) if r2_h_list else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)

# ---------- 宏平均报告（等权口径：mean/std/95%CI/CV） ----------
def build_macro_tables(df_per_pig: pd.DataFrame, target_set: str, horizon: int):
    df = df_per_pig[df_per_pig["集合"] == target_set].copy()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # overall：直接用逐猪 RMSE 列（=7天均值）
    m, sd, lo, hi = _bootstrap_ci_mean(df["RMSE"].values)
    cv = _cv_ratio(m, sd)
    overall = pd.DataFrame([{
        "集合": target_set,
        "N_pigs": int(df.shape[0]),
        "Metric": "RMSE_macro_over_pigs",
        "Mean": m, "Std": sd, "CI95_L": lo, "CI95_H": hi,
        "CV": cv, "CV_%": (cv * 100.0 if not np.isnan(cv) else np.nan)
    }])

    # per-horizon + h_avg（先猪内均值，再跨猪宏平均）
    rows = []
    rmse_cols = [f"RMSE_h{h}" for h in range(1, horizon+1)]
    mat = df[rmse_cols].apply(pd.to_numeric, errors="coerce")
    per_pig_havg = mat.mean(axis=1).values
    m, sd, lo, hi = _bootstrap_ci_mean(per_pig_havg)
    cv = _cv_ratio(m, sd)
    rows.append({"水平":"h_avg","N_pigs":int(df.shape[0]),"RMSE_Mean":m,"RMSE_Std":sd,"CI95_L":lo,"CI95_H":hi,
                 "RMSE_CV":cv,"RMSE_CV_%":(cv*100.0 if not np.isnan(cv) else np.nan)})
    for h in range(1, horizon+1):
        m, sd, lo, hi = _bootstrap_ci_mean(mat[f"RMSE_h{h}"].values)
        cv = _cv_ratio(m, sd)
        rows.append({"水平":f"h{h}","N_pigs":int(df.shape[0]),"RMSE_Mean":m,"RMSE_Std":sd,"CI95_L":lo,"CI95_H":hi,
                     "RMSE_CV":cv,"RMSE_CV_%":(cv*100.0 if not np.isnan(cv) else np.nan)})
    horizon_tbl = pd.DataFrame(rows, columns=["水平","N_pigs","RMSE_Mean","RMSE_Std","CI95_L","CI95_H","RMSE_CV","RMSE_CV_%"])
    return overall, horizon_tbl

def add_macro_reports(writer, base_sheet_name: str, df_per_pig: pd.DataFrame, horizon: int, target_set: str):
    df_over, df_hor = build_macro_tables(df_per_pig, target_set, horizon)
    if df_over is not None and not df_over.empty:
        name1 = make_safe_sheetname(base_sheet_name, f"macro_summary_{target_set.lower()}")
        df_over.to_excel(writer, sheet_name=name1, index=False)
    if df_hor is not None and not df_hor.empty:
        name2 = make_safe_sheetname(base_sheet_name, f"macro_horizon_{target_set.lower()}")
        df_hor.to_excel(writer, sheet_name=name2, index=False)

# ---------- 选参目标：逐猪等权宏平均 RMSE ----------
def _macro_rmse_over_pigs_equal_weight(Y_val: pd.DataFrame, Yhat_val: np.ndarray, groups_val: np.ndarray, horizon: int) -> float:
    pigs = np.unique(groups_val)
    per_pig = []
    Yv = Y_val.to_numpy() if isinstance(Y_val, pd.DataFrame) else np.asarray(Y_val)
    Yh = np.asarray(Yhat_val)
    for p in pigs:
        m = (groups_val == p)
        if not np.any(m):
            continue
        y_p = Yv[m]       # (n_win_p, H)
        yh_p = Yh[m]      # (n_win_p, H)
        rmse_h = [rmse_safe(y_p[:, h], yh_p[:, h]) for h in range(horizon)]
        per_pig.append(float(np.nanmean(rmse_h)))
    return float(np.nanmean(per_pig)) if per_pig else np.nan

def tune_params_global_5fold(X: pd.DataFrame, Y: pd.DataFrame, groups: np.ndarray,
                             base_params: dict, candidates: List[dict],
                             horizon: int = 7, inner_folds: int = 5):
    """
    全数据上（按猪分组）做 GroupKFold(inner_folds) 选参一次，返回 (best_params, logs)。
    """
    uniq = np.unique(groups)
    n_splits = max(2, min(inner_folds, len(uniq)))
    gkf = GroupKFold(n_splits=n_splits)
    logs = []
    best_params = dict(base_params); best_score = float("inf")

    for cand in (candidates or [dict()]):
        params = dict(base_params); params.update(cand)
        # GPU/CPU 切换
        params = dict(params, device=("gpu" if USE_GPU else "cpu"))
        scores = []
        for tr_idx, va_idx in gkf.split(X, Y, groups):
            est = LGBMRegressor(**params)
            model = MultiOutputRegressor(est, n_jobs=-1)
            model.fit(X.iloc[tr_idx], Y.iloc[tr_idx])
            Yhat = model.predict(X.iloc[va_idx])
            score = _macro_rmse_over_pigs_equal_weight(
                Y.iloc[va_idx], np.asarray(Yhat), groups[va_idx], horizon
            )
            scores.append(score)
        mean_score = float(np.nanmean(scores)) if scores else np.inf
        log_row = dict(inner_folds=n_splits, mean_macro_RMSE=mean_score)
        log_row.update(cand)
        logs.append(log_row)
        if mean_score < best_score:
            best_score = mean_score
            best_params = params

    return best_params, logs

# ---------- 预测 & 汇总 ----------
def predict_samples(model, samples: List[dict], horizon: int) -> List[np.ndarray]:
    return [model.predict(s["X"]) if len(s["X"]) else np.empty((0, horizon)) for s in samples]

def add_summary_sheet(writer, df_per_pig: pd.DataFrame, sheet_name: str, set_filter: str | None = None):
    if df_per_pig.empty: return
    df = df_per_pig if set_filter is None else df_per_pig[df_per_pig["集合"] == set_filter]
    if df.empty: return
    overall = pd.DataFrame([dict(级别="Overall", n=df["n"].sum(), RMSE=df["RMSE"].mean(), MAE=np.nan, R2=df["R2"].mean())])
    by_breed = (df.groupby("品种", dropna=False).agg(n=("n","sum"), RMSE=("RMSE","mean"), R2=("R2","mean"))
                  .reset_index().rename(columns={"品种":"级别"}))
    by_station = (df.groupby("测定站", dropna=False).agg(n=("n","sum"), RMSE=("RMSE","mean"), R2=("R2","mean"))
                    .reset_index().rename(columns={"测定站":"级别"}))
    summary = pd.concat([overall.assign(类别="Overall"),
                         by_breed.assign(类别="ByBreed"),
                         by_station.assign(类别="ByStation")], ignore_index=True)
    summary.to_excel(writer, sheet_name=sheet_name, index=False)

def add_set_means_and_agg(writer, df_per_pig: pd.DataFrame, base_sheet_name: str, horizon: int,
                          target_set: str = "Val"):
    if df_per_pig.empty: return
    df = df_per_pig[df_per_pig["集合"] == target_set].copy()
    if df.empty: return

    rmse_cols = [f"RMSE_h{h}" for h in range(1, horizon+1)]
    r2_cols   = [f"R2_h{h}"   for h in range(1, horizon+1)]
    rmse_mat = df[rmse_cols].apply(pd.to_numeric, errors="coerce")
    r2_mat   = df[r2_cols].apply(pd.to_numeric, errors="coerce")

    rows = []
    for h in range(1, horizon+1):
        rows.append(dict(水平=f"h{h}", n_pigs=int(df.shape[0]),
                         RMSE_mean=float(rmse_mat[f"RMSE_h{h}"].mean()),
                         R2_mean=float(r2_mat[f"R2_h{h}"].mean())))
    table_day_means = pd.DataFrame(rows, columns=["水平","n_pigs","RMSE_mean","R2_mean"])

    rmse_avg_per_pig = rmse_mat.mean(axis=1)
    r2_avg_per_pig   = r2_mat.mean(axis=1)
    table_agg = pd.DataFrame([
        dict(指标="RMSE_avg_h1h7", n_pigs=int(df.shape[0]),
             mean=float(rmse_avg_per_pig.mean()), median=float(rmse_avg_per_pig.median()),
             std=float(rmse_avg_per_pig.std(ddof=1))),
        dict(指标="R2_avg_h1h7",   n_pigs=int(df.shape[0]),
             mean=float(r2_avg_per_pig.mean()), median=float(r2_avg_per_pig.median()),
             std=float(r2_avg_per_pig.std(ddof=1))),
    ])

    sheet_name = f"{base_sheet_name}__{target_set.lower()}_means"
    if len(sheet_name) > 31: sheet_name = sheet_name[:31]
    table_day_means.to_excel(writer, sheet_name=sheet_name, index=False, startrow=0)
    table_agg.to_excel(writer, sheet_name=sheet_name, index=False, startrow=len(table_day_means)+2)

# ---------- SHAP 工具函数（LightGBM 版本） ----------
def _balanced_sample_by_groups(X: pd.DataFrame, groups: np.ndarray, max_per_pig: int, seed: int = 42):
    """按猪等量抽样，最多每头猪取 max_per_pig 个窗口，避免某些猪窗口过多导致 SHAP 偏置。"""
    if groups is None or len(groups) != len(X):
        if len(X) > 5000:
            return X.sample(n=5000, random_state=seed)
        return X
    idxs = []
    rng = np.random.default_rng(seed)
    pigs = pd.unique(groups)
    for p in pigs:
        m = (groups == p)
        ids = np.where(m)[0]
        if len(ids) <= max_per_pig:
            idxs.extend(ids.tolist())
        else:
            pick = rng.choice(ids, size=max_per_pig, replace=False)
            idxs.extend(pick.tolist())
    idxs = sorted(set(idxs))
    return X.iloc[idxs]

def _strip_lag(name: str) -> str:
    """把特征名的 _lagK 去掉，得到“特征族”名；init_weight 原样返回。"""
    if name == "init_weight":
        return name
    return re.sub(r"_lag\d+$", "", name)

def _get_lgbm_booster(est):
    # 兼容不同版本的 sklearn-API 包装器
    b = getattr(est, "booster_", None)
    if b is None:
        b = getattr(est, "_Booster", None)
    return b if b is not None else est

def compute_shap_for_multioutput(model, X_in: pd.DataFrame, groups: np.ndarray | None,
                                 horizon: int, max_per_pig: int = 20, seed: int = 42):
    """
    对 MultiOutputRegressor 内部的 7 个 LGBM 子模型分别做 SHAP；
    返回：
      df_hwise:     每个地平线的特征重要性（|SHAP|均值），列 ['feature','h','abs_shap_mean']
      df_family:    按“特征族”（去掉_lag）聚合后的重要性（各 h 以及 h_avg），按 h_avg 降序
    """
    X = _balanced_sample_by_groups(X_in, groups, max_per_pig=max_per_pig, seed=seed)
    feat_names = list(X.columns)

    rows = []
    for h in range(horizon):
        est_h = model.estimators_[h]  # 第 h 个 LGBMRegressor
        booster = _get_lgbm_booster(est_h)
        explainer = shap.TreeExplainer(booster)
        shap_vals = explainer.shap_values(X)   # 回归：array(n_samples, n_features)
        if isinstance(shap_vals, list):  # 兼容某些返回 list 的情况
            shap_vals = shap_vals[0]
        abs_mean = np.abs(shap_vals).mean(axis=0)  # (n_features,)
        for fname, v in zip(feat_names, abs_mean):
            rows.append({"feature": fname, "h": f"h{h+1}", "abs_shap_mean": float(v)})

    df_hwise = pd.DataFrame(rows)
    df_hwise["family"] = df_hwise["feature"].map(_strip_lag)
    df_family_h = (df_hwise.groupby(["family", "h"])["abs_shap_mean"]
                   .sum().reset_index())  # 同一 family 汇总所有 lag

    family_pivot = df_family_h.pivot(index="family", columns="h", values="abs_shap_mean").fillna(0.0)
    cols = [f"h{i}" for i in range(1, horizon+1)]
    for c in cols:
        if c not in family_pivot.columns:
            family_pivot[c] = 0.0
    family_pivot = family_pivot[cols]
    family_pivot["h_avg"] = family_pivot[cols].mean(axis=1)
    family_pivot = family_pivot.sort_values("h_avg", ascending=False).reset_index()

    return df_hwise.sort_values(["h","abs_shap_mean"], ascending=[True, False]), family_pivot

# ---------- “所有均值（总的）”汇总（新增） ----------
def build_global_means(cv_all: pd.DataFrame, horizon: int) -> pd.DataFrame:
    rows = []
    for set_name in ["Train", "Val"]:
        df = cv_all[cv_all["集合"] == set_name].copy()
        if df.empty:
            continue
        row = dict(
            集合=set_name,
            N_rows=int(df.shape[0]),
            N_pigs=int(df["猪ID"].nunique()),
            RMSE_mean=float(pd.to_numeric(df["RMSE"], errors="coerce").mean()),
            R2_mean=float(pd.to_numeric(df["R2"], errors="coerce").mean()),
        )
        for h in range(1, horizon+1):
            row[f"RMSE_h{h}_mean"] = float(pd.to_numeric(df[f"RMSE_h{h}"], errors="coerce").mean())
            row[f"R2_h{h}_mean"]   = float(pd.to_numeric(df[f"R2_h{h}"], errors="coerce").mean())
        rows.append(row)
    return pd.DataFrame(rows)

# ---------- 命名 ----------
def make_safe_sheetname(base: str, tag: str | None = None, used: set | None = None, maxlen: int = 31) -> str:
    name = base if tag is None else f"{base}__{tag}"
    name = re.sub(r'[:\\/?*\[\]]', '-', name)
    if len(name) > maxlen:
        if tag:
            keep = maxlen - (len(tag) + 2)
            keep = max(1, keep)
            name = f"{base[:keep]}__{tag}"
            if len(name) > maxlen:
                name = name[:maxlen]
        else:
            name = name[:maxlen]
    if used is not None:
        orig = name; i = 1
        while name in used:
            suffix = f"_{i}"
            name = orig[:maxlen - len(suffix)] + suffix
            i += 1
        used.add(name)
    return name

def derive_out_path(base_excel: str | Path, window: int, horizon: int) -> Path:
    p = Path(base_excel)
    p.parent.mkdir(parents=True, exist_ok=True)
    stem = p.stem
    return p.with_name(f"{stem}_W{window}_H{horizon}.xlsx")

# ---------- 主过程 ----------
def run_tune5cv_then_eval10cv():
    # 读数造样
    raw = load_daily_table(DATA_PATH)
    df, bcol, scol = normalize_df(raw)
    feat_cols = pick_features(df, bcol, scol)
    samples = build_all_samples(df, bcol, scol, feat_cols, WINDOW, HORIZON)
    if not samples:
        raise RuntimeError(f"无可用样本（窗口={WINDOW}）")

    # 全量 X/Y/Groups（用于：先 5 折调参；后 10 折划分）
    X_all, Y_all, groups_all = build_Xy_groups(samples)
    uniq_pigs = np.unique(groups_all)
    if len(uniq_pigs) < 3:
        raise RuntimeError("猪头数过少，无法进行 5 折/10 折。")

    # ===== 先：全局 5 折调参（按猪分组，逐猪等权宏平均 RMSE）=====
    best_params, tuning_logs = tune_params_global_5fold(
        X_all, Y_all, groups_all,
        base_params=BASE_LGB_PARAMS, candidates=PARAM_CANDIDATES,
        horizon=HORIZON, inner_folds=INNER_FOLDS_TUNE
    )

    # ===== 后：用最优参数做 10 折评估（按猪分组）=====
    n_splits_outer = max(2, min(OUTER_FOLDS_EVAL, len(uniq_pigs)))
    gkf_outer = GroupKFold(n_splits=n_splits_outer)

    out_file = derive_out_path(OUT_EXCEL, WINDOW, HORIZON)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    used_names = set()
    all_val_rows = []
    split_records = []
    fold_macro_records = []

    with pd.ExcelWriter(out_file, engine="xlsxwriter") as writer:
        for fold_idx, (tr_idx, va_idx) in enumerate(gkf_outer.split(X_all, Y_all, groups_all), start=1):
            fold_tag = f"Fold{fold_idx}"
            base = make_safe_sheetname(fold_tag, used=used_names)

            train_pigs = np.unique(groups_all[tr_idx])
            val_pigs   = np.unique(groups_all[va_idx])

            tr_s = [s for s in samples if s["pid"] in train_pigs]
            va_s = [s for s in samples if s["pid"] in val_pigs]

            Xtr, Ytr, gtr = build_Xy_groups(tr_s)
            Xva, Yva, gva = build_Xy_groups(va_s)
            if len(Xtr) == 0 or len(Xva) == 0:
                print(f"⚠️ 第 {fold_idx} 折样本不足，跳过。")
                continue

            # 终训（本折，最优参数已固定，不再内层调参）
            est_final = LGBMRegressor(**best_params)
            model = MultiOutputRegressor(est_final, n_jobs=-1)
            model.fit(Xtr, Ytr)

            # 预测 & 评估
            yhat_tr = predict_samples(model, tr_s, HORIZON)
            yhat_va = predict_samples(model, va_s, HORIZON)

            df_train = eval_by_pig_named(tr_s, yhat_tr, set_name="Train", horizon=HORIZON)
            df_val   = eval_by_pig_named(va_s, yhat_va, set_name="Val",   horizon=HORIZON)
            df_all   = pd.concat([df_train, df_val], ignore_index=True)\
                         .assign(Fold=fold_tag, 窗口=WINDOW, H=HORIZON, CV=f"GroupKFold({n_splits_outer})")

            # 写入本折主表（逐猪明细）
            df_all.sort_values(["集合","猪ID"]).to_excel(writer, sheet_name=base, index=False)

            # 概览 & Val 均值表（h1..h7 + h_avg）
            add_summary_sheet(writer, df_all, sheet_name=make_safe_sheetname(base, "summary_train", used=used_names), set_filter="Train")
            add_summary_sheet(writer, df_all, sheet_name=make_safe_sheetname(base, "summary_val",   used=used_names), set_filter="Val")
            add_set_means_and_agg(writer, df_all, base, HORIZON, target_set="Val")

            # ====== SHAP（对 Val 做解释）======
            if DO_SHAP:
                df_shap_h, df_shap_family = compute_shap_for_multioutput(
                    model, Xva, groups=gva, horizon=HORIZON,
                    max_per_pig=SHAP_MAX_PER_PIG, seed=SEED
                )
                shap_sheet1 = make_safe_sheetname(base, "shap_val_hwise", used=used_names)
                shap_sheet2 = make_safe_sheetname(base, "shap_val_family", used=used_names)
                df_shap_h.to_excel(writer, sheet_name=shap_sheet1, index=False)
                df_shap_family.to_excel(writer, sheet_name=shap_sheet2, index=False)

            # 记录稳健性（本折 Val 的宏平均 RMSE = 逐猪 RMSE 的简单均值）
            fold_macro_rmse = df_val["RMSE"].mean() if not df_val.empty else np.nan
            fold_macro_records.append({"Fold": fold_tag, "N_val_pigs": int(df_val.shape[0]), "Macro_RMSE_Val": float(fold_macro_rmse)})

            # 累积
            all_val_rows.append(df_all[df_all["集合"]=="Val"].copy())
            split_records.append(dict(Fold=fold_tag,
                                      TrainPigs=",".join(sorted(map(str, train_pigs))),
                                      ValPigs=",".join(sorted(map(str, val_pigs)))))

        # 合并全部折的 Val 逐猪明细
        if all_val_rows:
            cv_all = pd.concat(all_val_rows, ignore_index=True)
            base_all = make_safe_sheetname("CV10_ALL", used=used_names)
            cv_all.to_excel(writer, sheet_name=base_all, index=False)

            # ALL 的宏平均（Val）：overall + h1..h7（含 Std / 95%CI / CV）
            add_macro_reports(writer, base_all, cv_all, HORIZON, target_set="Val")

            # （新增）所有的均值（总的）：这里按当前口径，仅对 Val 进行“总的”均值汇总。
            global_means = build_global_means(pd.concat(all_val_rows + [], ignore_index=True), HORIZON)
            sheet_global = make_safe_sheetname("GLOBAL_MEANS", used=used_names)
            global_means.to_excel(writer, sheet_name=sheet_global, index=False)

            # 稳健性报告：逐折 Val 宏平均 RMSE 及折间均值/标准差/95%CI/CV
            if fold_macro_records:
                df_stab = pd.DataFrame(fold_macro_records)
                mean_, sd_, lo_, hi_ = _bootstrap_ci_mean(df_stab["Macro_RMSE_Val"].values,
                                                          B=BOOTSTRAP_B, alpha=CI_ALPHA)
                cv_ = _cv_ratio(mean_, sd_)
                summary_row = pd.DataFrame([{
                    "Fold": "CV-mean",
                    "N_val_pigs": int(df_stab["N_val_pigs"].sum()),  # 合计（仅参考）
                    "Macro_RMSE_Val": mean_,
                    "Std_over_folds": sd_,
                    "CI95_L": lo_,
                    "CI95_H": hi_,
                    "CV_over_folds": cv_,
                    "CV_over_folds_%": cv_ * 100.0 if not np.isnan(cv_) else np.nan
                }])
                df_out = pd.concat([df_stab, summary_row], ignore_index=True)
                sheet_stab = make_safe_sheetname("STABILITY_Val", used=used_names)
                df_out.to_excel(writer, sheet_name=sheet_stab, index=False)

        # 附：splits & 调参日志
        if split_records:
            pd.DataFrame(split_records).to_excel(writer, sheet_name=make_safe_sheetname("splits_info", used=used_names), index=False)
        if tuning_logs:
            logs_df = pd.DataFrame(tuning_logs)
            logs_df.to_excel(writer, sheet_name=make_safe_sheetname("tuning_info", used=used_names), index=False)
            pd.DataFrame([{"Selected": str(best_params)}]).to_excel(writer, sheet_name=make_safe_sheetname("selected_params", used=used_names), index=False)

    print(f"✅ 已完成：LightGBM 5折调参（按猪）→ 10折评估（按猪）。结果写入：{derive_out_path(OUT_EXCEL, WINDOW, HORIZON)}")
    print(f"⭐ 最优参数：{best_params}")

# ---------- 入口 ----------
if __name__ == "__main__":
    run_tune5cv_then_eval10cv()