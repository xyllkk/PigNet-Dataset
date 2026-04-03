# -*- coding: utf-8 -*-
"""
TimesNet（天粒度，固定滑窗 W=14 → H=7，多输出 MIMO）
默认运行：按“猪ID分组”的 **10 折 GroupKFold 交叉验证**（防止窗口/个体泄露）。
汇总口径：**逐猪宏平均**（macro over pigs），提供 **均值 / 标准差 / 95%CI（bootstrap）/ CV变异系数**。

依赖：
  pip install torch torchvision torchaudio xlsxwriter pandas scikit-learn numpy
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

# ---------------- PyTorch ----------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ====== 随机种子 ======
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

# =========================
# 手动参数区（只改这里）
# =========================
DATA_PATH = r"E:\文献\课题\AI\猪场实验数据\输入_采食加环境加日龄_按天"
OUT_EXCEL = r"E:\文献\课题\AI\猪场实验数据\正式试验结果\尝试方法\TimesNet_10_fold_CV_win14"  # 文件名我未改，方便与你现有结果对齐

DATE_COL    = "日期"
PIG_COL     = "耳缺号"
WEIGHT_COL  = "体重"
BREED_COL   = None
STATION_COL = None

# 固定：14→7
WINDOW  = 14
HORIZON = 7

# 不剔除湿度/CO2（保持空）
EXCLUDE_FEATURE_KEYWORDS = []

# ======== 训练超参（保留你的键名：用于 TimesNet 的 d_model/e_layers/dropout 映射） ========
BEST_PARAMS = {
    "hidden_size": 128,      # → TimesNet.d_model
    "num_layers": 1,         # → TimesNet.e_layers
    "dropout": 0.25,         # → TimesNet.dropout
    "lr": 1e-3,
    "epochs": 400,
    "batch_size": 128,
    # 可选：TimesNet 专属（若不填用默认值）
    # "times_top_k": 3,
    # "times_num_kernels": 3,
}

WEIGHT_DECAY = 3e-4
USE_HUBER = True
USE_LR_SCHEDULER = True
EARLY_STOPPING = True
EARLY_PATIENCE = 25

# —— 以下模块开关在 TimesNet 版本中不再使用，但保留占位不影响其它流程 ——
USE_TCN_FRONTEND = False
TCN_DILATIONS = [1, 2, 4]
TCN_KERNEL = 3
TCN_DROPOUT = 0.15
USE_FILM_INIT = True
SE_REDUCTION = 4
USE_HQ_ATTENTION = True
ATTN_HEADS = 2
ATTN_DROPOUT = 0.10
FUSE_LAST = True

USE_SMOOTH_LOSS = True
SMOOTH_LAMBDA = 0.02

# 固定划分参数（仅 Global901 用）
GLOBAL901_TEST_COUNT = 4
GLOBAL901_SEED = 42

# ======== 统计配置（逐猪宏平均 + 95%CI + CV） ========
MACRO_TARGET_SET = "Val"
BOOTSTRAP_B = 2000
CI_ALPHA = 0.05
_EPS = 1e-12

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
    Xs, Ys = [], []
    if len(g) < window + horizon:
        return np.empty((0, window, len(feat_cols)+1)), pd.DataFrame()

    init_series = g[WEIGHT_COL].dropna()
    if init_series.empty:
        return np.empty((0, window, len(feat_cols)+1)), pd.DataFrame()
    init_w = float(init_series.iloc[0])

    g_feat = g[feat_cols].copy()
    g_feat = g_feat.apply(pd.to_numeric, errors="coerce")

    for t in range(window - 1, len(g) - horizon):
        w = g_feat.iloc[t - window + 1:t + 1]                  # (window, F)
        x = w.to_numpy(dtype=float)
        init_col = np.full((window, 1), init_w, dtype=float)
        Xs.append(np.concatenate([x, init_col], axis=1))       # (window, F+1)
        Ys.append([float(g[WEIGHT_COL].iloc[t + h]) for h in range(1, horizon + 1)])

    X_seq = np.asarray(Xs, dtype=float)                         # (n_win, window, F+1)
    Y_df  = pd.DataFrame(Ys, columns=[f"y_h{h}" for h in range(1, horizon + 1)])
    return X_seq, Y_df

# ---------- 拼接 ----------
def build_Xy_groups(samples_list: List[dict]):
    X_parts, Y_parts, groups = [], [], []
    for s in samples_list:
        X_seq = s["X"]
        if X_seq is None or len(X_seq) == 0:
            continue
        X_parts.append(X_seq)
        Y_parts.append(s["Y"].to_numpy(dtype=float))
        groups.extend([s["pid"]] * X_seq.shape[0])
    if not X_parts:
        return np.empty((0, WINDOW, 0)), np.empty((0, HORIZON)), np.array([])
    X = np.vstack(X_parts)   # (N, T, F)
    Y = np.vstack(Y_parts)   # (N, H)
    return X, Y, np.array(groups, dtype=object)

# ---------- 标准化器 ----------
class SeqFeatureScaler:
    def __init__(self):
        self.mean_ = None
        self.std_  = None
    def fit(self, X: np.ndarray):
        feat = X.reshape(-1, X.shape[-1])
        self.mean_ = np.nanmean(feat, axis=0)
        self.std_  = np.nanstd(feat, axis=0)
        self.std_[self.std_ < 1e-12] = 1.0
    def transform(self, X: np.ndarray):
        X2 = np.copy(X)
        inds = np.isnan(X2)
        if np.any(inds):
            idx_feat = np.where(inds)[2]
            X2[inds] = np.take(self.mean_, idx_feat)
        return (X2 - self.mean_) / self.std_

# ====================== TimesNet ======================
def sinusoidal_pos_encoding(L: int, d_model: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    pos = torch.arange(L, device=device, dtype=dtype).unsqueeze(1)  # [L,1]
    i = torch.arange(d_model, device=device, dtype=dtype).unsqueeze(0)  # [1,D]
    angle_rates = 1.0 / (10000 ** ( (i//2)*2 / d_model ))
    angles = pos * angle_rates
    pe = torch.zeros(L, d_model, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(angles[:, 0::2])
    pe[:, 1::2] = torch.cos(angles[:, 1::2])
    return pe.unsqueeze(0)  # [1,L,D]

class TimesBlock(nn.Module):
    """
    频域选 k 个主频 → 将时间维按周期 p 折叠为 2D → 2D Inception 卷积 → 按能量权重融合 → 残差 + LN
    输入: x [B, T, C]；输出: [B, T, C]
    """
    def __init__(self, d_model: int, top_k: int = 3, num_kernels: int = 3, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.top_k = max(1, top_k)
        ks = [3,5,7,11][:max(1, num_kernels)]
        self.convs = nn.ModuleList([
            nn.Conv2d(d_model, d_model, kernel_size=(1, k), padding=(0, k//2))
            for k in ks
        ])
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        x_res = x

        # 频谱（沿 T 做 rFFT）
        xp = x.permute(0,2,1)                       # [B,C,T]
        xf = torch.fft.rfft(xp, dim=-1)             # [B,C,T//2+1]
        amp = xf.abs().mean(dim=(0,1))              # [T//2+1]
        if amp.shape[0] <= 1:
            return self.ln(x_res)                   # 极短序列保护
        amp[0] = 0                                  # 去直流
        k = min(self.top_k, amp.shape[0]-1)
        vals, idxs = torch.topk(amp, k=k, dim=0)    # 频率索引
        weights = torch.softmax(vals, dim=0)        # 权重

        outs = []
        for j, f_idx in enumerate(idxs.tolist()):
            period = max(1, int(T // max(1, f_idx)))
            # 折叠为 2D
            pad = (period - (T % period)) % period
            xp2 = xp
            if pad > 0:
                xp2 = F.pad(xp2, (0, pad))
            H = xp2.shape[-1] // period
            x2d = xp2.view(B, C, H, period)         # [B,C,H,p]
            # Inception 2D conv
            y = 0
            for conv in self.convs:
                y = y + self.act(conv(x2d))
            y = y / len(self.convs)
            y = self.dropout(y)
            y = y.reshape(B, C, -1)[:, :, :T]       # 去填充
            outs.append(y)

        # 能量加权融合
        y_sum = torch.zeros_like(xp)
        for w, y in zip(weights, outs):
            y_sum = y_sum + w * y
        y_sum = y_sum.permute(0,2,1)                # [B,T,C]

        return self.ln(y_sum + x_res)

class TimesNetForecaster(nn.Module):
    """
    输入: x [B, seq_len, in_dim]
    输出: y [B, pred_len]    （单目标，与你的评估接口一致）
    """
    def __init__(self, in_dim: int, seq_len: int, pred_len: int,
                 d_model: int = 256, e_layers: int = 3,
                 top_k: int = 3, num_kernels: int = 3, dropout: float = 0.1,
                 c_out: int = 1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.value_emb = nn.Linear(in_dim, d_model)
        self.emb_drop = nn.Dropout(dropout)

        # 将时间长度 T 线性扩展为 T+H（非自回归“一步到位”）
        self.predict_linear = nn.Linear(seq_len, seq_len + pred_len)

        self.blocks = nn.ModuleList([
            TimesBlock(d_model=d_model, top_k=top_k, num_kernels=num_kernels, dropout=dropout)
            for _ in range(max(1, e_layers))
        ])
        self.proj = nn.Linear(d_model, c_out)
        self.final_ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F = x.shape
        assert T == self.seq_len, f"输入序列长度需为 {self.seq_len}，但得到 {T}"

        h = self.value_emb(x)                # [B,T,D]
        h = self.emb_drop(h)

        # 时间扩展到 T+H
        h_btC = h.permute(0,2,1)             # [B,D,T]
        h_btC = self.predict_linear(h_btC)   # [B,D,T+H]
        h = h_btC.permute(0,2,1)             # [B,T+H,D]

        # 加位置编码（sin-cos）
        pe = sinusoidal_pos_encoding(h.shape[1], self.d_model, h.device, h.dtype)  # [1,L,D]
        h = h + pe

        # TimesBlocks
        for blk in self.blocks:
            h = blk(h)
        h = self.final_ln(h)

        y_full = self.proj(h).squeeze(-1)    # [B,T+H]
        y = y_full[:, -self.pred_len:]       # 取未来 H 段 → [B,H]
        return y

# ---------- Dataset ----------
class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# ---------- 训练/验证 ----------
def rmse_safe(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def _flatten_y(y):
    arr = np.asarray(y)
    return arr.reshape(-1)

def build_model(in_dim: int, out_h: int, params: dict):
    # 将你的 BEST_PARAMS 键映射到 TimesNet
    d_model   = int(params.get("hidden_size", 256))
    e_layers  = int(params.get("num_layers", 2))
    dropout   = float(params.get("dropout", 0.1))
    top_k     = int(params.get("times_top_k", 3))
    num_ker   = int(params.get("times_num_kernels", 3))

    return TimesNetForecaster(
        in_dim=in_dim, seq_len=WINDOW, pred_len=out_h,
        d_model=d_model, e_layers=e_layers,
        top_k=top_k, num_kernels=num_ker, dropout=dropout, c_out=1
    )

def train_one_model(Xtr, Ytr, Xva=None, Yva=None, params: dict | None = None, device: str | None = None):
    p = dict(BEST_PARAMS if params is None else params)
    lr          = p.get("lr", 1e-3)
    epochs      = p.get("epochs", 400)
    batch_size  = p.get("batch_size", 128)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    scaler = SeqFeatureScaler(); scaler.fit(Xtr)
    Xtr_n = scaler.transform(Xtr)
    Xva_n = scaler.transform(Xva) if (Xva is not None and len(Xva)) else None

    model = build_model(in_dim=Xtr.shape[-1], out_h=Ytr.shape[-1], params=p).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    crit = nn.SmoothL1Loss(beta=1.0) if USE_HUBER else nn.MSELoss()
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=8)
                 if USE_LR_SCHEDULER and Xva_n is not None else None)

    tr_loader = DataLoader(SeqDataset(Xtr_n, Ytr), batch_size=batch_size, shuffle=True, drop_last=False)
    va_loader = DataLoader(SeqDataset(Xva_n, Yva), batch_size=batch_size, shuffle=False, drop_last=False) if Xva_n is not None else None

    best_val = np.inf
    best_state = None
    bad = 0

    for ep in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad()
            yp = model(xb)                             # (B, H)
            loss_main = crit(yp, yb)
            if USE_SMOOTH_LOSS and yp.shape[1] > 1:
                smooth = torch.mean(torch.abs(yp[:, 1:] - yp[:, :-1]))
                loss = loss_main + SMOOTH_LAMBDA * smooth
            else:
                loss = loss_main
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

        if va_loader is not None:
            model.eval(); y_true, y_pred = [], []
            with torch.no_grad():
                for xb, yb in va_loader:
                    xb = xb.to(device); yb = yb.to(device)
                    yp = model(xb)
                    y_true.append(yb.cpu().numpy()); y_pred.append(yp.cpu().numpy())
            y_true = np.vstack(y_true); y_pred = np.vstack(y_pred)
            val_rmse = rmse_safe(_flatten_y(y_true), _flatten_y(y_pred))
            if scheduler is not None:
                scheduler.step(val_rmse)
            if val_rmse < best_val - 1e-6:
                best_val = val_rmse
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if EARLY_STOPPING and bad >= EARLY_PATIENCE:
                    break
        else:
            best_val = np.nan

    if best_state is not None:
        model.load_state_dict(best_state)

    # 附加 scaler 以便推理
    model.scaler_mean_ = torch.tensor(scaler.mean_, dtype=torch.float32, device=device)
    model.scaler_std_  = torch.tensor(scaler.std_,  dtype=torch.float32, device=device)
    return model, float(best_val)

def model_predict(model: nn.Module, X: np.ndarray, device: str | None = None) -> np.ndarray:
    if X is None or len(X) == 0:
        return np.empty((0, HORIZON))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        mean = model.scaler_mean_; std = model.scaler_std_
        Xn = torch.tensor(X, dtype=torch.float32, device=device)
        Xn = (Xn - mean) / std
        Yp = model(Xn).cpu().numpy()
    return Yp

# ---------- 评估与输出 ----------
def predict_samples(model, samples: List[dict], horizon: int) -> List[np.ndarray]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    outs = []
    for s in samples:
        X_seq = s["X"]  # (n_win, T, F)
        Yhat = model_predict(model, X_seq, device=device)
        outs.append(Yhat)
    return outs

def eval_by_pig_named(samples_list: List[dict], yhat_list: List[np.ndarray], set_name: str, horizon: int) -> pd.DataFrame:
    rows = []
    for s, Yhat in zip(samples_list, yhat_list):
        Y = s["Y"].to_numpy()  # (n_win, H)
        if Y.size == 0:
            continue
        y_pred = np.asarray(Yhat)
        n_win = Y.shape[0]

        rmse_h_list, r2_h_list, ystd_h_list = [], [], []
        rec = dict(集合=set_name, 猪ID=s["pid"], 品种=s["breed"], 测定站=s["station"],
                   n_win=int(n_win), n=int(n_win * horizon))

        for h in range(1, horizon + 1):
            yt = Y[:, h-1]
            yp = y_pred[:, h-1]
            rmse_h = rmse_safe(yt, yp)
            r2_h   = r2_score(yt, yp) if len(np.unique(yt)) > 1 else np.nan
            ystd_h = float(np.std(yt, ddof=1)) if len(yt) > 1 else np.nan

            rec[f"RMSE_h{h}"] = rmse_h
            rec[f"R2_h{h}"]   = r2_h

            rmse_h_list.append(rmse_h)
            r2_h_list.append(r2_h)
            ystd_h_list.append(ystd_h)

        rmse_havg = float(np.nanmean(rmse_h_list)) if len(rmse_h_list) else np.nan
        r2_havg   = float(np.nanmean([v for v in r2_h_list if pd.notnull(v)])) if len(r2_h_list) else np.nan
        ystd_havg = float(np.nanmean([v for v in ystd_h_list if pd.notnull(v)])) if len(ystd_h_list) else np.nan

        rec["RMSE"] = rmse_havg
        rec["R2"]   = r2_havg
        rec["y_std"] = ystd_havg
        rec["rmse_over_std"] = (rmse_havg / ystd_havg) if (ystd_havg is not None and not np.isnan(ystd_havg) and ystd_havg > 0) else np.nan

        rows.append(rec)

    return pd.DataFrame(rows)

# ----------（微平均的地平线汇总；保留） ----------
def eval_set_horizon_metrics(samples_list: List[dict], yhat_list: List[np.ndarray], set_name: str, horizon: int) -> pd.DataFrame:
    Ys, Yhats = [], []
    for s, Yhat in zip(samples_list, yhat_list):
        if len(s["Y"]) == 0:
            continue
        Ys.append(s["Y"].to_numpy())
        Yhats.append(np.asarray(Yhat))
    if not Ys:
        return pd.DataFrame()
    Y = np.vstack(Ys)
    Yhat = np.vstack(Yhats)

    rows = []
    for h in range(1, horizon + 1):
        yt = Y[:, h-1]; yp = Yhat[:, h-1]
        rows.append(dict(集合=set_name, 水平=f"h{h}", n_win=int(Y.shape[0]), n=int(len(yt)),
                         RMSE=rmse_safe(yt, yp),
                         R2=r2_score(yt, yp) if len(np.unique(yt)) > 1 else np.nan))
    yt_all = Y.reshape(-1); yp_all = Yhat.reshape(-1)
    rows.append(dict(集合=set_name, 水平="All", n_win=int(Y.shape[0]), n=int(len(yt_all)),
                     RMSE=rmse_safe(yt_all, yp_all),
                     R2=r2_score(yt_all, yp_all) if len(np.unique(yt_all)) > 1 else np.nan))
    return pd.DataFrame(rows)

# ----------（原版）Test 的逐猪均值表（兼容 Global901） ----------
def add_test_pig_means_and_agg(writer, df_per_pig: pd.DataFrame, base_sheet_name: str, horizon: int, selected_params: dict | None = None, model_modules: str | None = None):
    if df_per_pig.empty:
        return
    df = df_per_pig[df_per_pig["集合"] == "Test"].copy()
    if df.empty:
        return

    rmse_cols = [f"RMSE_h{h}" for h in range(1, horizon+1)]
    r2_cols   = [f"R2_h{h}"   for h in range(1, horizon+1)]
    rmse_mat = df[rmse_cols].apply(pd.to_numeric, errors="coerce")
    r2_mat   = df[r2_cols].apply(pd.to_numeric, errors="coerce")

    rows = []
    for h in range(1, horizon+1):
        rows.append(dict(
            水平=f"h{h}", n_pigs=int(df.shape[0]),
            RMSE_mean=float(rmse_mat[f"RMSE_h{h}"].mean()),
            R2_mean=float(r2_mat[f"R2_h{h}"].mean())
        ))
    table_day_means = pd.DataFrame(rows, columns=["水平","n_pigs","RMSE_mean","R2_mean"])

    rmse_avg_per_pig = rmse_mat.mean(axis=1)
    r2_avg_per_pig   = r2_mat.mean(axis=1)
    table_agg = pd.DataFrame([
        dict(指标="RMSE_avg_h1h7", n_pigs=int(df.shape[0]),
             mean=float(rmse_avg_per_pig.mean()),
             median=float(rmse_avg_per_pig.median()),
             std=float(rmse_avg_per_pig.std(ddof=1))),
        dict(指标="R2_avg_h1h7",   n_pigs=int(df.shape[0]),
             mean=float(r2_avg_per_pig.mean()),
             median=float(r2_avg_per_pig.median()),
             std=float(r2_avg_per_pig.std(ddof=1))),
    ])

    sheet_name = f"{base_sheet_name}__test_means"
    if len(sheet_name) > 31:
        sheet_name = sheet_name[:31]
    start0 = 0
    table_day_means.to_excel(writer, sheet_name=sheet_name, index=False, startrow=start0)
    start1 = start0 + len(table_day_means) + 2
    table_agg.to_excel(writer, sheet_name=sheet_name, index=False, startrow=start1)

    start2 = start1 + len(table_agg) + 2
    meta = {}
    if selected_params is not None: meta["Selected"] = str(selected_params)
    if model_modules is not None:   meta["ModelModules"] = model_modules
    if meta:
        pd.DataFrame([meta]).to_excel(writer, sheet_name=sheet_name, index=False, startrow=start2)

# ----------（通用版）任意集合均值表（CV 用：target_set="Val"） ----------
def add_set_means_and_agg(writer, df_per_pig: pd.DataFrame, base_sheet_name: str, horizon: int,
                          target_set: str = "Test",
                          selected_params: dict | None = None, model_modules: str | None = None):
    if df_per_pig.empty:
        return
    df = df_per_pig[df_per_pig["集合"] == target_set].copy()
    if df.empty:
        return

    rmse_cols = [f"RMSE_h{h}" for h in range(1, horizon+1)]
    r2_cols   = [f"R2_h{h}"   for h in range(1, horizon+1)]
    rmse_mat = df[rmse_cols].apply(pd.to_numeric, errors="coerce")
    r2_mat   = df[r2_cols].apply(pd.to_numeric, errors="coerce")

    rows = []
    for h in range(1, horizon+1):
        rows.append(dict(
            水平=f"h{h}", n_pigs=int(df.shape[0]),
            RMSE_mean=float(rmse_mat[f"RMSE_h{h}"].mean()),
            R2_mean=float(r2_mat[f"R2_h{h}"].mean())
        ))
    table_day_means = pd.DataFrame(rows, columns=["水平","n_pigs","RMSE_mean","R2_mean"])

    rmse_avg_per_pig = rmse_mat.mean(axis=1)
    r2_avg_per_pig   = r2_mat.mean(axis=1)
    table_agg = pd.DataFrame([
        dict(指标="RMSE_avg_h1h7", n_pigs=int(df.shape[0]),
             mean=float(rmse_avg_per_pig.mean()),
             median=float(rmse_avg_per_pig.median()),
             std=float(rmse_avg_per_pig.std(ddof=1))),
        dict(指标="R2_avg_h1h7",   n_pigs=int(df.shape[0]),
             mean=float(r2_avg_per_pig.mean()),
             median=float(r2_avg_per_pig.median()),
             std=float(r2_avg_per_pig.std(ddof=1))),
    ])

    sheet_name = f"{base_sheet_name}__{target_set.lower()}_means"
    if len(sheet_name) > 31:
        sheet_name = sheet_name[:31]

    start0 = 0
    table_day_means.to_excel(writer, sheet_name=sheet_name, index=False, startrow=start0)
    start1 = start0 + len(table_day_means) + 2
    table_agg.to_excel(writer, sheet_name=sheet_name, index=False, startrow=start1)

    start2 = start1 + len(table_agg) + 2
    meta = {}
    if selected_params is not None: meta["Selected"] = str(selected_params)
    if model_modules is not None:   meta["ModelModules"] = model_modules
    if meta:
        pd.DataFrame([meta]).to_excel(writer, sheet_name=sheet_name, index=False, startrow=start2)

# ---------- 宏平均统计辅助（含95%CI与CV） ----------
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
    return float(np.mean(vals)), lo, hi, float(np.std(vals, ddof=1))

def _cv_ratio(mean_val, std_val):
    if mean_val is None or np.isnan(mean_val) or abs(mean_val) < _EPS:
        return np.nan
    return float(std_val / (mean_val + _EPS))

def build_macro_tables(df_per_pig: pd.DataFrame, target_set: str, horizon: int,
                       B: int = BOOTSTRAP_B, alpha: float = CI_ALPHA):
    df = df_per_pig[df_per_pig["集合"] == target_set].copy()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    overall_mean, overall_lo, overall_hi, overall_std = _bootstrap_ci_mean(df["RMSE"].values, B=B, alpha=alpha)
    overall_cv = _cv_ratio(overall_mean, overall_std)
    df_overall = pd.DataFrame([{
        "集合": target_set,
        "N_pigs": int(df.shape[0]),
        "Metric": "RMSE_macro_over_pigs",
        "Mean": overall_mean,
        "Std": overall_std,
        "CI95_L": overall_lo,
        "CI95_H": overall_hi,
        "CV": overall_cv,
        "CV_%": overall_cv * 100.0 if not np.isnan(overall_cv) else np.nan
    }])

    rows = []
    rmse_mat = df[[f"RMSE_h{h}" for h in range(1, horizon+1)]].apply(pd.to_numeric, errors="coerce")

    per_pig_havg = rmse_mat.mean(axis=1).values
    m, lo, hi, sd = _bootstrap_ci_mean(per_pig_havg, B=B, alpha=alpha)
    cv = _cv_ratio(m, sd)
    rows.append({
        "水平": "h_avg",
        "N_pigs": int(df.shape[0]),
        "RMSE_Mean": m,
        "RMSE_Std": sd,
        "CI95_L": lo,
        "CI95_H": hi,
        "RMSE_CV": cv,
        "RMSE_CV_%": (cv * 100.0 if not np.isnan(cv) else np.nan),
    })

    for h in range(1, horizon+1):
        vals = rmse_mat[f"RMSE_h{h}"].values
        m, lo, hi, sd = _bootstrap_ci_mean(vals, B=B, alpha=alpha)
        cv = _cv_ratio(m, sd)
        rows.append({
            "水平": f"h{h}",
            "N_pigs": int(df.shape[0]),
            "RMSE_Mean": m,
            "RMSE_Std": sd,
            "CI95_L": lo,
            "CI95_H": hi,
            "RMSE_CV": cv,
            "RMSE_CV_%": (cv * 100.0 if not np.isnan(cv) else np.nan),
        })
    df_horizon = pd.DataFrame(rows, columns=["水平","N_pigs","RMSE_Mean","RMSE_Std","CI95_L","CI95_H","RMSE_CV","RMSE_CV_%"])

    return df_overall, df_horizon

def add_macro_reports(writer, base_sheet_name: str, df_per_pig: pd.DataFrame, horizon: int,
                      target_set: str = MACRO_TARGET_SET):
    df_overall, df_horizon = build_macro_tables(df_per_pig, target_set=target_set, horizon=horizon)
    if df_overall is not None and not df_overall.empty:
        name1 = make_safe_sheetname(base_sheet_name, f"macro_summary_{target_set.lower()}")
        df_overall.to_excel(writer, sheet_name=name1, index=False)
    if df_horizon is not None and not df_horizon.empty:
        name2 = make_safe_sheetname(base_sheet_name, f"macro_horizon_{target_set.lower()}")
        df_horizon.to_excel(writer, sheet_name=name2, index=False)

# ---------- 划分（仅 Global901） ----------
def split_global_fixed_count_stratified(samples: List[dict], test_count: int, seed: int):
    rng = np.random.default_rng(seed)
    br2pigs: Dict[str, List[str]] = {}
    for s in samples:
        br2pigs.setdefault(s["breed"], []).append(s["pid"])
    total = sum(len(v) for v in br2pigs.values())
    if test_count >= total:
        raise ValueError(f"测试头数({test_count}) >= 猪总数({total})。")
    alloc = {br: max(1, round(len(v) / total * test_count)) for br, v in br2pigs.items()}
    diff = test_count - sum(alloc.values())
    keys = list(alloc.keys()); i = 0
    while diff != 0 and len(keys) > 0:
        k = keys[i % len(keys)]
        if diff > 0:
            alloc[k] += 1; diff -= 1
        else:
            if alloc[k] > 1:
                alloc[k] -= 1; diff += 1
        i += 1
    test = set()
    for br, k in alloc.items():
        pigs = br2pigs[br]
        rng.shuffle(pigs)
        k = min(k, len(pigs))
        test.update(pigs[:k])
    all_pigs = set([s["pid"] for s in samples])
    train = all_pigs - test
    return train, test

# ---------- Sheet 名 & 汇总 ----------
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

def add_summary_sheet(writer, df_per_pig: pd.DataFrame, sheet_name: str, set_filter: str | None = None):
    if df_per_pig.empty: return
    df = df_per_pig if set_filter is None else df_per_pig[df_per_pig["集合"] == set_filter]
    if df.empty: return
    overall = pd.DataFrame([dict(级别="Overall", n=df["n"].sum(), RMSE=df["RMSE"].mean(), MAE=np.nan, R2=df["R2"].mean())])
    by_breed = (df.groupby("品种", dropna=False)
                .agg(n=("n","sum"), RMSE=("RMSE","mean"), R2=("R2","mean"))
                .reset_index().rename(columns={"品种":"级别"}))
    by_station = (df.groupby("测定站", dropna=False)
                  .agg(n=("n","sum"), RMSE=("RMSE","mean"), R2=("R2","mean"))
                  .reset_index().rename(columns={"测定站":"级别"}))
    summary = pd.concat([overall.assign(类别="Overall"),
                         by_breed.assign(类别="ByBreed"),
                         by_station.assign(类别="ByStation")], ignore_index=True)
    summary.to_excel(writer, sheet_name=sheet_name, index=False)

# ---------- 构建全部样本 ----------
def build_all_samples(df: pd.DataFrame, bcol: str, scol: str, feat_cols: List[str], window: int, horizon: int):
    out = []
    for pid, g in df.groupby(PIG_COL, sort=False):
        g = g[[DATE_COL] + feat_cols + [WEIGHT_COL, bcol, scol]].sort_values(DATE_COL).reset_index(drop=True)
        X_seq, Y_df = make_samples_one_pig_sliding_multiH(g, feat_cols, window, horizon)
        if len(Y_df):
            out.append(dict(pid=str(pid), breed=str(g[bcol].iloc[0]), station=str(g[scol].iloc[0]), X=X_seq, Y=Y_df))
    return out

# ---------- 主过程：仅 Global901 ----------
def derive_out_path(base_excel: str | Path, window: int, horizon: int) -> Path:
    p = Path(base_excel)
    p.parent.mkdir(parents=True, exist_ok=True)
    stem = p.stem
    return p.with_name(f"{stem}_W{window}_H{horizon}.xlsx")

def run_global901_only():
    raw = load_daily_table(DATA_PATH)
    df, bcol, scol = normalize_df(raw)
    feat_cols = pick_features(df, bcol, scol)
    samples = build_all_samples(df, bcol, scol, feat_cols, WINDOW, HORIZON)
    if not samples:
        raise RuntimeError(f"无可用样本（窗口={WINDOW}）")

    out_file = derive_out_path(OUT_EXCEL, WINDOW, HORIZON)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    split_records = []

    with pd.ExcelWriter(out_file, engine="xlsxwriter") as writer:
        used_names = set()

        # ===== Global901 =====
        g_train, g_test = split_global_fixed_count_stratified(samples, test_count=GLOBAL901_TEST_COUNT, seed=GLOBAL901_SEED)
        if g_train and g_test:
            tr_s = [s for s in samples if s["pid"] in g_train]
            te_s = [s for s in samples if s["pid"] in g_test]
            Xtr, Ytr, groups = build_Xy_groups(tr_s)
            if len(Xtr):
                best_params = dict(BEST_PARAMS)

                model, _ = train_one_model(Xtr, Ytr, params=best_params)
                yhat_tr = predict_samples(model, tr_s, HORIZON)
                yhat_te = predict_samples(model, te_s, HORIZON)

                df_train = eval_by_pig_named(tr_s, yhat_tr, set_name="Train", horizon=HORIZON)
                df_test  = eval_by_pig_named(te_s, yhat_te, set_name="Test",  horizon=HORIZON)
                df_all   = pd.concat([df_train, df_test], ignore_index=True)

                base = make_safe_sheetname("Global901", used=used_names)
                df_all.assign(Fold="Global901__fixed(stratified)", 窗口=WINDOW, H=HORIZON, CV="Global901")\
                      .sort_values(["集合","Fold","猪ID"]).to_excel(writer, sheet_name=base, index=False)

                # 保留总体/分层概览
                add_summary_sheet(writer, df_all, sheet_name=make_safe_sheetname(base, "summary_all", used=used_names), set_filter=None)
                add_summary_sheet(writer, df_all, sheet_name=make_safe_sheetname(base, "summary_train", used=used_names), set_filter="Train")
                add_summary_sheet(writer, df_all, sheet_name=make_safe_sheetname(base, "summary_test", used=used_names), set_filter="Test")

                modules_desc = (f"TimesNet(d_model={best_params.get('hidden_size',128)}, "
                                f"e_layers={best_params.get('num_layers',1)}, "
                                f"top_k={best_params.get('times_top_k',3)}, "
                                f"kernels={best_params.get('times_num_kernels',3)}, "
                                f"dropout={best_params.get('dropout',0.25)}); "
                                f"PredictLinear(seq={WINDOW},pred={HORIZON}); "
                                f"Loss={'SmoothL1' if USE_HUBER else 'MSE'}+TV-L1*{SMOOTH_LAMBDA}")

                add_test_pig_means_and_agg(writer, df_all, base, HORIZON,
                                           selected_params=best_params, model_modules=modules_desc)

                split_records.append(dict(Sheet="Global901", Fold="Global901__fixed(stratified)",
                                          TestPigs=",".join(sorted(list(g_test))),
                                          TrainPigs=",".join(sorted(list(g_train))),
                                          Selected=str(best_params),
                                          ModelModules=modules_desc))

        # ===== 附加信息 =====
        fi_df = pd.DataFrame({"Feature": ["(TimesNet 非树模型无内置特征重要性；可做Permutation Importance)"],
                              "Importance": [np.nan]})
        fi_df.to_excel(writer, sheet_name=make_safe_sheetname("feature_importance", used=used_names), index=False)
        if split_records:
            pd.DataFrame(split_records).to_excel(writer, sheet_name=make_safe_sheetname("splits_info", used=used_names), index=False)

    print(f"✅ 固定窗口 W={WINDOW}, H={HORIZON} 的 TimesNet 结果已写入：{out_file}")

# ---------- 新增：10折 GroupKFold（按猪ID） ----------
def run_cv_kfold_by_pig(n_splits: int = 10):
    # --------- 读数据 & 造样本 ---------
    raw = load_daily_table(DATA_PATH)
    df, bcol, scol = normalize_df(raw)
    feat_cols = pick_features(df, bcol, scol)
    samples = build_all_samples(df, bcol, scol, feat_cols, WINDOW, HORIZON)
    if not samples:
        raise RuntimeError(f"无可用样本（窗口={WINDOW}）")
    n_pigs = len(samples)
    if n_pigs < 2:
        raise RuntimeError(f"猪头数({n_pigs}) 太少，无法做交叉验证。")
    n_splits = min(n_splits, n_pigs)

    # --------- 构造按猪分组的 K 折 ---------
    pig_ids = np.array([s["pid"] for s in samples])
    rng = np.random.default_rng(GLOBAL901_SEED)
    perm = rng.permutation(len(pig_ids))
    pig_ids = pig_ids[perm]
    X_dummy = np.zeros((len(pig_ids), 1), dtype=np.float32)
    gkf = GroupKFold(n_splits=n_splits)

    # --------- 输出路径 ---------
    p = Path(OUT_EXCEL)
    p.parent.mkdir(parents=True, exist_ok=True)
    out_file = p.with_name(f"{p.stem}_CV{n_splits}_W{WINDOW}_H{HORIZON}.xlsx")

    # --------- 逐折训练与评估 ---------
    all_fold_records = []
    split_records = []
    fold_macro_records = []
    modules_desc = (f"TimesNet(d_model={BEST_PARAMS.get('hidden_size',128)}, "
                    f"e_layers={BEST_PARAMS.get('num_layers',1)}, "
                    f"top_k={BEST_PARAMS.get('times_top_k',3)}, "
                    f"kernels={BEST_PARAMS.get('times_num_kernels',3)}, "
                    f"dropout={BEST_PARAMS.get('dropout',0.25)}); "
                    f"PredictLinear(seq={WINDOW},pred={HORIZON}); "
                    f"Loss={'SmoothL1' if USE_HUBER else 'MSE'}+TV-L1*{SMOOTH_LAMBDA}")

    with pd.ExcelWriter(out_file, engine="xlsxwriter") as writer:
        used_names = set()

        for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(X_dummy, y=None, groups=pig_ids), start=1):
            fold_tag = f"CV{n_splits}_F{fold_idx}"
            base = make_safe_sheetname(fold_tag, used=used_names)

            train_pigs = set(pig_ids[tr_idx])
            val_pigs   = set(pig_ids[va_idx])

            tr_s = [s for s in samples if s["pid"] in train_pigs]
            va_s = [s for s in samples if s["pid"] in val_pigs]

            Xtr, Ytr, _ = build_Xy_groups(tr_s)
            Xva, Yva, _ = build_Xy_groups(va_s)
            if len(Xtr) == 0 or len(Xva) == 0:
                print(f"⚠️ 第 {fold_idx} 折样本不足，跳过。")
                continue

            best_params = dict(BEST_PARAMS)
            model, _ = train_one_model(Xtr, Ytr, Xva, Yva, params=best_params)

            # 预测 & 评估（集合：Train / Val）
            yhat_tr = predict_samples(model, tr_s, HORIZON)
            yhat_va = predict_samples(model, va_s, HORIZON)

            df_train = eval_by_pig_named(tr_s, yhat_tr, set_name="Train", horizon=HORIZON)
            df_val   = eval_by_pig_named(va_s, yhat_va, set_name="Val",   horizon=HORIZON)
            df_all   = pd.concat([df_train, df_val], ignore_index=True)
            df_all = df_all.assign(Fold=fold_tag, 窗口=WINDOW, H=HORIZON, CV=f"GroupKFold({n_splits})")\
                           .sort_values(["集合","Fold","猪ID"])

            # 写入本折主表（逐猪明细）
            df_all.to_excel(writer, sheet_name=base, index=False)

            # 逐猪宏平均：overall & h1..h7（含 Std / 95%CI / CV）
            add_macro_reports(writer, base, df_all, HORIZON, target_set="Val")

            # 概览
            add_summary_sheet(writer, df_all, sheet_name=make_safe_sheetname(base, "summary_all",   used=used_names), set_filter=None)
            add_summary_sheet(writer, df_all, sheet_name=make_safe_sheetname(base, "summary_train", used=used_names), set_filter="Train")
            add_summary_sheet(writer, df_all, sheet_name=make_safe_sheetname(base, "summary_val",   used=used_names), set_filter="Val")

            # 逐猪“Val”均值与总体统计（保留）
            add_set_means_and_agg(writer, df_all, base, HORIZON, target_set="Val",
                                  selected_params=best_params, model_modules=modules_desc)

            # 稳健性（本折 Val 的宏平均 RMSE）
            fold_macro_rmse = df_val["RMSE"].mean() if not df_val.empty else np.nan
            fold_macro_records.append({"Fold": fold_tag, "N_val_pigs": int(df_val.shape[0]), "Macro_RMSE_Val": float(fold_macro_rmse)})

            # 累积
            all_fold_records.append(df_all)
            split_records.append(dict(Sheet=fold_tag,
                                      TrainPigs=",".join(sorted(list(train_pigs))),
                                      ValPigs=",".join(sorted(list(val_pigs))),
                                      Selected=str(best_params),
                                      ModelModules=modules_desc))

        # 合并 K 折
        if all_fold_records:
            cv_all = pd.concat(all_fold_records, ignore_index=True)
            base_all = make_safe_sheetname(f"CV{n_splits}_ALL", used=used_names)
            cv_all.to_excel(writer, sheet_name=base_all, index=False)

            add_summary_sheet(writer, cv_all, sheet_name=make_safe_sheetname(base_all, "summary_all",   used=used_names), set_filter=None)
            add_summary_sheet(writer, cv_all, sheet_name=make_safe_sheetname(base_all, "summary_train", used=used_names), set_filter="Train")
            add_summary_sheet(writer, cv_all, sheet_name=make_safe_sheetname(base_all, "summary_val",   used=used_names), set_filter="Val")

            add_macro_reports(writer, base_all, cv_all, HORIZON, target_set="Val")

            if fold_macro_records:
                df_stab = pd.DataFrame(fold_macro_records)
                mean_, lo_, hi_, sd_ = _bootstrap_ci_mean(df_stab["Macro_RMSE_Val"].values,
                                                          B=BOOTSTRAP_B, alpha=CI_ALPHA)
                cv_ = _cv_ratio(mean_, sd_)
                summary_row = pd.DataFrame([{
                    "Fold": "CV-mean",
                    "N_val_pigs": int(df_stab["N_val_pigs"].sum()),
                    "Macro_RMSE_Val": mean_,
                    "Std_over_folds": sd_,
                    "CI95_L": lo_,
                    "CI95_H": hi_,
                    "CV_over_folds": cv_,
                    "CV_over_folds_%": cv_ * 100.0 if not np.isnan(cv_) else np.nan
                }])
                df_out = pd.concat([df_stab, summary_row], ignore_index=True)
                sheet_stab = make_safe_sheetname(f"CV{n_splits}_STABILITY", used=used_names)
                df_out.to_excel(writer, sheet_name=sheet_stab, index=False)

        # 附：特征重要性占位 & splits_info
        fi_df = pd.DataFrame({"Feature": ["(TimesNet 非树模型无内置特征重要性；可做Permutation Importance)"],
                              "Importance": [np.nan]})
        fi_df.to_excel(writer, sheet_name=make_safe_sheetname("feature_importance", used=used_names), index=False)
        if split_records:
            pd.DataFrame(split_records).to_excel(writer, sheet_name=make_safe_sheetname("splits_info", used=used_names), index=False)

    print(f"✅ 已完成 {n_splits} 折 GroupKFold（按猪ID）交叉验证（TimesNet），结果写入：{out_file}")

# ========= 主入口 =========
if __name__ == "__main__":
    # run_global901_only()
    run_cv_kfold_by_pig(n_splits=10)
