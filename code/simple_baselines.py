# -*- coding: utf-8 -*-
"""
Simple baselines for pig body-weight forecasting:
1) Linear regression: Age + Initial body weight
2) Quadratic regression: Age + Initial body weight

评估协议：
- 天粒度
- 固定 W=14, H=7
- 按猪ID分组的 10 折 GroupKFold
- 输出：
    * CV10_ALL_RAW    : 逐样本/逐地平线原始预测（可直接做 Wilcoxon）
    * CV10_ALL        : 逐猪汇总结果
    * CV10_STABILITY  : 逐折 Val 宏平均 RMSE 稳健性
    * CV10_F1 ... F10 : 每折逐猪汇总

依赖：
pip install pandas numpy scikit-learn openpyxl
"""

from pathlib import Path
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# =========================
# 手动参数区
# =========================
DATA_PATH = r"E:\文献\课题\AI\猪场实验数据\输入_采食加环境加日龄_按天"
OUT_DIR   = r"E:\文献\课题\AI\猪场实验数据\正式试验结果\最终结果\win14\论文结果\simple_baselines"

DATE_COL   = "日期"
PIG_COL    = "耳缺号"
WEIGHT_COL = "体重"
AGE_COL    = "日龄"
BREED_COL  = None
STATION_COL = None

WINDOW  = 14
HORIZON = 7
N_SPLITS = 10
SEED = 42

# 两个 simple baselines
RUN_MODELS = {
    "LinearReg_AgeInit": "linear",
    "QuadraticReg_AgeInit": "quadratic",
}


# =========================
# I/O
# =========================
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


# =========================
# 预处理
# =========================
def normalize_df(df: pd.DataFrame):
    for c, nm in [(DATE_COL, "日期"), (PIG_COL, "猪ID"), (WEIGHT_COL, "体重"), (AGE_COL, "日龄")]:
        if c not in df.columns:
            raise ValueError(f"缺少 {nm} 列：{c}")

    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df[PIG_COL] = df[PIG_COL].astype(str)
    df[WEIGHT_COL] = pd.to_numeric(df[WEIGHT_COL], errors="coerce")
    df[AGE_COL] = pd.to_numeric(df[AGE_COL], errors="coerce")

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
    def station_from_pid(pid: str):
        m = re.search(r'(\d+)[\-_]\d+', str(pid))
        return f"S{m.group(1)}" if m else None

    def station_from_source(src: str):
        s = str(src)
        m = re.search(r'(\d+)[\-_]\d+', s)
        if m:
            return f"S{m.group(1)}"
        return None

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


# =========================
# 造样本：simple baselines 用 target_age + initial_weight
# =========================
def build_baseline_rows(df: pd.DataFrame, bcol: str, scol: str, window: int, horizon: int) -> pd.DataFrame:
    """
    为 simple baseline 构造逐样本、逐地平线的监督数据。

    特征：
      - initial_weight：该猪入栏时第一条体重记录
      - target_age    ：目标日的日龄（不是当前日龄）
    标签：
      - y_true        ：目标日体重

    这样做的含义是：给定“初始体重 + 目标日龄”，做简单生长外推。
    """
    rows = []

    for pid, g in df.groupby(PIG_COL, sort=False):
        g = g[[DATE_COL, PIG_COL, WEIGHT_COL, AGE_COL, bcol, scol]].copy()
        g = g.sort_values(DATE_COL).reset_index(drop=True)

        init_series = g[WEIGHT_COL].dropna()
        if init_series.empty:
            continue
        init_w = float(init_series.iloc[0])

        if len(g) < window + horizon:
            continue

        breed = str(g[bcol].iloc[0])
        station = str(g[scol].iloc[0])

        win_idx = 0
        for t in range(window - 1, len(g) - horizon):
            win_idx += 1
            anchor_date = g.loc[t, DATE_COL]

            for h in range(1, horizon + 1):
                target_idx = t + h
                target_date = g.loc[target_idx, DATE_COL]
                target_age = g.loc[target_idx, AGE_COL]
                y_true = g.loc[target_idx, WEIGHT_COL]

                if pd.isna(target_age) or pd.isna(y_true):
                    continue

                rows.append({
                    "猪ID": pid,
                    "品种": breed,
                    "测定站": station,
                    "win_idx": win_idx,
                    "锚定日": anchor_date,
                    "地平线": f"h{h}",
                    "目标日": target_date,
                    "initial_weight": init_w,
                    "target_age": float(target_age),
                    "y_true": float(y_true),
                })

    return pd.DataFrame(rows)


# =========================
# 模型
# =========================
def build_model(kind: str):
    if kind == "linear":
        return LinearRegression()
    elif kind == "quadratic":
        return Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("reg", LinearRegression())
        ])
    else:
        raise ValueError(f"未知模型类型: {kind}")


def fit_and_predict(train_df: pd.DataFrame, test_df: pd.DataFrame, kind: str):
    model = build_model(kind)

    x_cols = ["initial_weight", "target_age"]
    Xtr = train_df[x_cols].to_numpy(dtype=float)
    ytr = train_df["y_true"].to_numpy(dtype=float)

    Xte = test_df[x_cols].to_numpy(dtype=float)

    model.fit(Xtr, ytr)

    yhat_tr = model.predict(Xtr)
    yhat_te = model.predict(Xte)

    return yhat_tr, yhat_te, model


# =========================
# 评估
# =========================
def rmse_safe(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def eval_per_pig(raw_pred_df: pd.DataFrame, set_name: str, horizon: int) -> pd.DataFrame:
    """
    输入是逐样本预测表，输出逐猪汇总表。
    """
    df = raw_pred_df[raw_pred_df["集合"] == set_name].copy()
    if df.empty:
        return pd.DataFrame()

    rows = []
    for (pid, breed, station, fold), g0 in df.groupby(["猪ID", "品种", "测定站", "Fold"], dropna=False):
        rec = {
            "集合": set_name,
            "猪ID": pid,
            "品种": breed,
            "测定站": station,
            "Fold": fold,
            "n_win": int(g0["win_idx"].nunique()),
            "n": int(len(g0)),
        }

        rmse_list = []
        r2_list = []

        for h in range(1, horizon + 1):
            gh = g0[g0["地平线"] == f"h{h}"].copy()
            if gh.empty:
                rec[f"RMSE_h{h}"] = np.nan
                rec[f"R2_h{h}"] = np.nan
                continue

            yt = gh["y_true"].to_numpy(dtype=float)
            yp = gh["y_pred"].to_numpy(dtype=float)

            rmse_h = rmse_safe(yt, yp)
            r2_h = r2_score(yt, yp) if len(np.unique(yt)) > 1 else np.nan

            rec[f"RMSE_h{h}"] = rmse_h
            rec[f"R2_h{h}"] = r2_h

            rmse_list.append(rmse_h)
            r2_list.append(r2_h)

        rec["RMSE"] = float(np.nanmean(rmse_list)) if len(rmse_list) else np.nan
        rec["R2"] = float(np.nanmean([x for x in r2_list if not pd.isna(x)])) if len(r2_list) else np.nan

        rows.append(rec)

    return pd.DataFrame(rows)


# =========================
# 主流程
# =========================
def run_one_baseline(model_name: str, kind: str, base_rows: pd.DataFrame, n_splits: int = 10):
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{model_name}_CV{n_splits}_W{WINDOW}_H{HORIZON}.xlsx"

    pig_ids = np.array(sorted(base_rows["猪ID"].unique()))
    n_splits = min(n_splits, len(pig_ids))

    rng = np.random.default_rng(SEED)
    pig_ids = pig_ids[rng.permutation(len(pig_ids))]

    X_dummy = np.zeros((len(pig_ids), 1), dtype=np.float32)
    gkf = GroupKFold(n_splits=n_splits)

    all_raw = []
    all_per_pig = []
    fold_macro_records = []
    split_records = []

    with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
        for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(X_dummy, y=None, groups=pig_ids), start=1):
            fold_tag = f"CV{n_splits}_F{fold_idx}"

            train_pigs = set(pig_ids[tr_idx])
            val_pigs = set(pig_ids[va_idx])

            tr_df = base_rows[base_rows["猪ID"].isin(train_pigs)].copy()
            va_df = base_rows[base_rows["猪ID"].isin(val_pigs)].copy()

            yhat_tr, yhat_va, _ = fit_and_predict(tr_df, va_df, kind=kind)

            tr_raw = tr_df.copy()
            tr_raw["y_pred"] = yhat_tr
            tr_raw["集合"] = "Train"
            tr_raw["Fold"] = fold_tag

            va_raw = va_df.copy()
            va_raw["y_pred"] = yhat_va
            va_raw["集合"] = "Val"
            va_raw["Fold"] = fold_tag

            fold_raw = pd.concat([tr_raw, va_raw], ignore_index=True)
            fold_raw = fold_raw[[
                "集合", "猪ID", "win_idx", "锚定日", "地平线", "目标日",
                "y_true", "y_pred", "Fold", "品种", "测定站",
                "initial_weight", "target_age"
            ]]

            all_raw.append(fold_raw)

            df_train = eval_per_pig(fold_raw, set_name="Train", horizon=HORIZON)
            df_val = eval_per_pig(fold_raw, set_name="Val", horizon=HORIZON)
            df_fold = pd.concat([df_train, df_val], ignore_index=True)
            df_fold["窗口"] = WINDOW
            df_fold["H"] = HORIZON
            df_fold["CV"] = f"GroupKFold({n_splits})"

            all_per_pig.append(df_fold)

            # 每折 Val 宏平均 RMSE
            fold_macro_rmse = df_val["RMSE"].mean() if not df_val.empty else np.nan
            fold_macro_records.append({
                "Fold": fold_tag,
                "N_val_pigs": int(df_val.shape[0]),
                "Macro_RMSE_Val": float(fold_macro_rmse)
            })

            split_records.append({
                "Sheet": fold_tag,
                "TrainPigs": ",".join(sorted(list(train_pigs))),
                "ValPigs": ",".join(sorted(list(val_pigs))),
                "Model": model_name,
                "Kind": kind,
                "Features": "Age + Initial body weight"
            })

            # 每折逐猪结果
            df_fold.sort_values(["集合", "猪ID"]).to_excel(writer, sheet_name=fold_tag[:31], index=False)

        # 合并输出
        df_all_raw = pd.concat(all_raw, ignore_index=True)
        df_all_raw.sort_values(["集合", "Fold", "猪ID", "win_idx", "地平线"]).to_excel(writer, sheet_name="CV10_ALL_RAW", index=False)

        df_all = pd.concat(all_per_pig, ignore_index=True)
        df_all = df_all.sort_values(["集合", "Fold", "猪ID"])
        df_all.to_excel(writer, sheet_name="CV10_ALL", index=False)

        df_split = pd.DataFrame(split_records)
        df_split.to_excel(writer, sheet_name="splits_info", index=False)

        # 稳健性表
        df_stab = pd.DataFrame(fold_macro_records)
        mean_ = df_stab["Macro_RMSE_Val"].mean()
        std_ = df_stab["Macro_RMSE_Val"].std(ddof=1)
        ci_lo = mean_ - 1.96 * std_ / max(np.sqrt(len(df_stab)), 1e-12)
        ci_hi = mean_ + 1.96 * std_ / max(np.sqrt(len(df_stab)), 1e-12)
        cv_ = std_ / mean_ if abs(mean_) > 1e-12 else np.nan

        summary_row = pd.DataFrame([{
            "Fold": "CV-mean",
            "N_val_pigs": int(df_stab["N_val_pigs"].sum()),
            "Macro_RMSE_Val": float(mean_),
            "Std_over_folds": float(std_),
            "CI95_L": float(ci_lo),
            "CI95_H": float(ci_hi),
            "CV_over_folds": float(cv_) if not pd.isna(cv_) else np.nan,
            "CV_over_folds_%": float(cv_ * 100.0) if not pd.isna(cv_) else np.nan
        }])

        df_stab_out = pd.concat([df_stab, summary_row], ignore_index=True)
        df_stab_out.to_excel(writer, sheet_name="CV10_STABILITY", index=False)

    print(f"已完成：{out_file}")


def main():
    raw = load_daily_table(DATA_PATH)
    df, bcol, scol = normalize_df(raw)

    base_rows = build_baseline_rows(df, bcol, scol, window=WINDOW, horizon=HORIZON)
    if base_rows.empty:
        raise RuntimeError("没有构造出任何 baseline 样本，请检查数据。")

    for model_name, kind in RUN_MODELS.items():
        run_one_baseline(model_name, kind, base_rows, n_splits=N_SPLITS)


if __name__ == "__main__":
    main()