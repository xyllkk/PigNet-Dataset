# -*- coding: utf-8 -*-
"""
LSTM测试（天粒度，序列→多输出 MIMO） + 固定滑动窗口 W=7 → H=7，预测未来 7 天体重。
本版：按猪分组的 **10折交叉验证**（固定最优超参，不再调参），并输出“逐猪宏平均”的统计与稳定性分析。

【已改】：把模型从 LSTM 换成 IEBlock 方法（连续/间隔采样 + 线性交互 + AR 高速路），
         并在 IEBlock 输出后加 Linear(enc_in→1) 读出体重，保证输出形状与原先一致 (B, H)。
其余流程、评估与Excel输出保持不变。

依赖：
  pip install torch torchvision torchaudio xlsxwriter pandas scikit-learn numpy
"""

from pathlib import Path
import re, math
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
OUT_EXCEL = r"E:\文献\课题\AI\猪场实验数据\正式试验结果\尝试方法\IEBlock_10_fold_CV_win14"  # 将生成 _W7_H7_Outer10CV.xlsx

DATE_COL    = "日期"
PIG_COL     = "耳缺号"
WEIGHT_COL  = "体重"
BREED_COL   = None
STATION_COL = None

# 固定：7→7（注意：本脚本仍按你的 WINDOW/HORIZON 造样；IEBlock 内部 chunk_size 会自动取 min(H, T, 24)）
WINDOW  = 14
HORIZON = 7

# 不剔除湿度/CO2（保持空）
EXCLUDE_FEATURE_KEYWORDS = []

# ======== 固定最优“超参”（映射到 IEBlock 的 d_model/训练配置） ========
BEST_PARAMS = {
    "hidden_size": 128,      # 映射为 d_model
    "num_layers": 1,         # IEBlock 不用层数，此项忽略，仅保留接口一致
    "dropout": 0.25,         # 仅用于读出 MLP 的 Dropout（若用到）
    "lr": 1e-3,
    "epochs": 400,           # 设大靠早停（若提供外层验证才会早停）
    "batch_size": 128,
}
WEIGHT_DECAY = 3e-4
USE_HUBER = True
USE_LR_SCHEDULER = True
EARLY_STOPPING = True
EARLY_PATIENCE = 25

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
      X_seq: np.ndarray (n_win, window, F + 1)  # +1 为静态 init_weight
      Y_df : pd.DataFrame (n_win, H)           # 列名 y_h1..y_h7
    """
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

# =====================================================================================
# ============================== IEBlock 方法（替换 LSTM）==============================
# =====================================================================================

class IEBlock(nn.Module):
    def __init__(self, input_dim: int, hid_dim: int, output_dim: int, num_node: int):
        """
        input_dim:   “块内时间长度”或上层特征维
        hid_dim:     隐层（建议与 d_model 对齐）
        output_dim:  输出的时间长度/特征维
        num_node:    节点数（前两层=块数；第三层=通道数 enc_in）
        """
        super().__init__()
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.num_node = num_node

        self.spatial_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.hid_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hid_dim, self.hid_dim // 4)
        )
        self.channel_proj = nn.Linear(self.num_node, self.num_node, bias=True)
        nn.init.eye_(self.channel_proj.weight)  # 节点维恒等初始化
        self.output_proj = nn.Linear(self.hid_dim // 4, self.output_dim)

    def forward(self, x):  # x: [B, input_dim, num_node]
        # 时间维 MLP
        x = self.spatial_proj(x.permute(0, 2, 1))          # [B, num_node, hid/4]
        # 节点维线性交互（带恒等初始化）
        x = x.permute(0, 2, 1) + self.channel_proj(x.permute(0, 2, 1))  # [B, hid/4, num_node]
        # 输出维对齐
        x = self.output_proj(x.permute(0, 2, 1))           # [B, num_node, output_dim]
        x = x.permute(0, 2, 1)                             # [B, output_dim, num_node]
        return x

class IEForecaster(nn.Module):
    """
    兼容你现有数据管线的 IEBlock 版本：
    输入:  x_enc [B, T, F]，输出: [B, H]（仅体重）
    做法:  IEBlock 主干得到 [B, H, F]，再经 Linear(F→1) 读出，并加 AR 高速路（同样读出为标量后相加）。
    """
    def __init__(self, seq_len: int, enc_in: int, pred_len: int, d_model: int = 128, chunk_size: int = 24):
        super().__init__()
        self.seq_len = seq_len
        self.enc_in = enc_in
        self.pred_len = pred_len

        # chunk_size 取 min(pred_len, seq_len, chunk_size) 与原实现一致
        self.chunk_size = min(pred_len, seq_len, chunk_size)
        if self.seq_len % self.chunk_size != 0:
            pad = self.chunk_size - self.seq_len % self.chunk_size
            self.seq_len = self.seq_len + pad
        self.num_chunks = self.seq_len // self.chunk_size

        # IEBlock 主干
        self.layer_1 = IEBlock(input_dim=self.chunk_size, hid_dim=d_model // 4,
                               output_dim=d_model // 4, num_node=self.num_chunks)
        self.chunk_proj_1 = nn.Linear(self.num_chunks, 1)

        self.layer_2 = IEBlock(input_dim=self.chunk_size, hid_dim=d_model // 4,
                               output_dim=d_model // 4, num_node=self.num_chunks)
        self.chunk_proj_2 = nn.Linear(self.num_chunks, 1)

        self.layer_3 = IEBlock(input_dim=d_model // 2, hid_dim=d_model // 2,
                               output_dim=self.pred_len, num_node=self.enc_in)

        # AR 高速路: seq_len → pred_len（逐通道独立），随后用 readout 聚合到体重标量
        self.ar = nn.Linear(self.seq_len, self.pred_len)
        # 体重读出（把 [B, H, F] → [B, H]）
        self.readout = nn.Linear(self.enc_in, 1)

    def encoder(self, x):
        """
        x: [B, T0, F]，其中 T0=WINDOW。若 T0 < self.seq_len 会在时间维做 0 填充（dtype 对齐）。
        返回: [B, pred_len, F]
        """
        B, T0, N = x.size()
        if T0 < self.seq_len:
            pad = torch.zeros((B, self.seq_len - T0, N), device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)

        # AR 高速路: 先 [B, N, T] → [B, N, H] → [B, H, N]
        highway = self.ar(x.permute(0, 2, 1)).permute(0, 2, 1)

        # 连续采样分支
        x1 = x.reshape(B, self.num_chunks, self.chunk_size, N).permute(0, 3, 2, 1)   # [B, N, chunk, num_chunks]
        x1 = x1.reshape(-1, self.chunk_size, self.num_chunks)                        # [B*N, chunk, num_chunks]
        x1 = self.layer_1(x1)                                                        # [B*N, d/4, num_chunks]
        x1 = self.chunk_proj_1(x1).squeeze(-1)                                       # [B*N, d/4]

        # 间隔采样分支
        x2 = x.reshape(B, self.chunk_size, self.num_chunks, N).permute(0, 3, 1, 2)   # [B, N, chunk, num_chunks]
        x2 = x2.reshape(-1, self.chunk_size, self.num_chunks)                        # [B*N, chunk, num_chunks]
        x2 = self.layer_2(x2)                                                        # [B*N, d/4, num_chunks]
        x2 = self.chunk_proj_2(x2).squeeze(-1)                                       # [B*N, d/4]

        # 拼接 → [B, d/2, N]
        x3 = torch.cat([x1, x2], dim=-1).reshape(B, N, -1).permute(0, 2, 1)

        # 解码到 pred_len：得到 [B, H, F]
        out = self.layer_3(x3).permute(0, 1, 2)  # [B, pred_len, enc_in]
        # 与 AR 高速路相加（先把高速路也读出为体重标量再相加）
        return out, highway

    def forward(self, x):  # x: [B, T, F]
        out, highway = self.encoder(x)                 # out, highway: [B, H, F]
        # 体重读出：对最后一维（通道/特征）做线性投影到标量
        y_main = self.readout(out).squeeze(-1)         # [B, H]
        y_ar   = self.readout(highway).squeeze(-1)     # [B, H]
        return y_main + y_ar                           # [B, H]

# ---------- 数据集 ----------
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

def train_one_model(Xtr, Ytr, Xva=None, Yva=None, params: dict | None = None, device: str | None = None):
    """
    与原接口保持一致，但内部模型换成 IEForecaster。
    hidden_size 作为 d_model 使用；num_layers 不生效，仅为兼容。
    """
    p = dict(BEST_PARAMS if params is None else params)
    d_model    = p.get("hidden_size", 128)
    dropout    = p.get("dropout", 0.25)   # 目前仅用于可能的读出 MLP；此实现未额外使用
    lr         = p.get("lr", 1e-3)
    epochs     = p.get("epochs", 400)
    batch_size = p.get("batch_size", 128)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # 标准化
    scaler = SeqFeatureScaler(); scaler.fit(Xtr)
    Xtr_n = scaler.transform(Xtr)
    Xva_n = scaler.transform(Xva) if (Xva is not None and len(Xva)) else None

    # ===== 模型替换处 =====
    in_dim = Xtr.shape[-1]
    out_h  = Ytr.shape[-1]
    model = IEForecaster(seq_len=WINDOW, enc_in=in_dim, pred_len=out_h, d_model=d_model, chunk_size=24).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    crit = nn.SmoothL1Loss(beta=1.0) if USE_HUBER else nn.MSELoss()
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=8)
                 if USE_LR_SCHEDULER and (Xva_n is not None) else None)

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
            yp = model(xb)                         # [B, H]
            loss = crit(yp, yb)                    # 与原 LSTM 相同的 SmoothL1/MSE
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

    # 附加 scaler 以便推理（与你原推理代码兼容）
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
        Yp = model(Xn).cpu().numpy()   # [B, H]
    return Yp

# ---------- 评估（逐猪） ----------
def predict_samples(model, samples: List[dict], horizon: int) -> List[np.ndarray]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    outs = []
    for s in samples:
        X_seq = s["X"]  # (n_win, T, F)
        Yhat = model_predict(model, X_seq, device=device)  # (n_win, H)
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

        rmse_havg = float(np.nanmean(rmse_h_list)) if rmse_h_list else np.nan
        r2_havg   = float(np.nanmean([v for v in r2_h_list if pd.notnull(v)])) if r2_h_list else np.nan
        ystd_havg = float(np.nanmean([v for v in ystd_h_list if pd.notnull(v)])) if ystd_h_list else np.nan

        rec["RMSE"] = rmse_havg
        rec["R2"]   = r2_havg
        rec["y_std"] = ystd_havg
        rec["rmse_over_std"] = (rmse_havg / ystd_havg) if (ystd_havg is not None and not np.isnan(ystd_havg) and ystd_havg > 0) else np.nan

        rows.append(rec)

    return pd.DataFrame(rows)

# ---------- 宏平均与 Bootstrap 工具 ----------
def bootstrap_mean_ci(values: np.ndarray, B: int = 2000, alpha: float = 0.05, seed: int = SEED):
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    mean = float(np.mean(arr))
    if n == 1:
        lo = hi = mean
        cv = 0.0 if mean == 0 else std / (mean + 1e-12)
        return mean, std, lo, hi, cv
    stats = []
    for _ in range(B):
        sample = rng.choice(arr, size=n, replace=True)
        stats.append(float(np.mean(sample)))
    stats = np.asarray(stats)
    lo, hi = float(np.quantile(stats, alpha/2)), float(np.quantile(stats, 1 - alpha/2))
    cv = float(std / (mean + 1e-12))
    return mean, std, lo, hi, cv

def macro_over_pigs_overall(df_val_pig: pd.DataFrame):
    vals = df_val_pig["RMSE"].to_numpy()
    mean, std, lo, hi, cv = bootstrap_mean_ci(vals)
    return pd.DataFrame([dict(metric="RMSE_overall",
                              mean=mean, std=std, CI95_lo=lo, CI95_hi=hi,
                              CI_width=(hi-lo) if (not math.isnan(hi) and not math.isnan(lo)) else np.nan,
                              CV=cv, n_pigs=int(len(vals)) )])

def macro_over_pigs_horizon(df_val_pig: pd.DataFrame, H: int):
    rows = []
    for h in range(1, H+1):
        col = f"RMSE_h{h}"
        if col in df_val_pig.columns:
            vals = df_val_pig[col].to_numpy()
            mean, std, lo, hi, cv = bootstrap_mean_ci(vals)
            rows.append(dict(level=f"h{h}", mean=mean, std=std, CI95_lo=lo, CI95_hi=hi,
                             CI_width=(hi-lo) if (not math.isnan(hi) and not math.isnan(lo)) else np.nan,
                             CV=cv, n_pigs=int(len(vals))))
    rmse_cols = [f"RMSE_h{h}" for h in range(1, H+1) if f"RMSE_h{h}" in df_val_pig.columns]
    if rmse_cols:
        per_pig_avg = df_val_pig[rmse_cols].mean(axis=1).to_numpy()
        mean, std, lo, hi, cv = bootstrap_mean_ci(per_pig_avg)
        rows.append(dict(level="h_avg", mean=mean, std=std, CI95_lo=lo, CI95_hi=hi,
                         CI_width=(hi-lo) if (not math.isnan(hi) and not math.isnan(lo)) else np.nan,
                         CV=cv, n_pigs=int(len(per_pig_avg))))
    return pd.DataFrame(rows, columns=["level","mean","std","CI95_lo","CI95_hi","CI_width","CV","n_pigs"])

def stability_over_folds(fold_macro_means: List[float]):
    vals = np.asarray(fold_macro_means, dtype=float)
    mean, std, lo, hi, cv = bootstrap_mean_ci(vals)
    df_sum = pd.DataFrame([dict(mean=mean, std=std, CI95_lo=lo, CI95_hi=hi,
                                CI_width=(hi-lo) if (not math.isnan(hi) and not math.isnan(lo)) else np.nan,
                                CV=cv, n_folds=int(len(vals)))])
    df_list = pd.DataFrame(dict(fold=list(range(1, len(vals)+1)), val_macro_rmse=vals))
    return df_list, df_sum

# ---------- 构建全部样本 ----------
def build_all_samples(df: pd.DataFrame, bcol: str, scol: str, feat_cols: List[str], window: int, horizon: int):
    out = []
    for pid, g in df.groupby(PIG_COL, sort=False):
        g = g[[DATE_COL] + feat_cols + [WEIGHT_COL, bcol, scol]].sort_values(DATE_COL).reset_index(drop=True)
        X_seq, Y_df = make_samples_one_pig_sliding_multiH(g, feat_cols, window, horizon)
        if len(Y_df):
            out.append(dict(pid=str(pid), breed=str(g[bcol].iloc[0]), station=str(g[scol].iloc[0]), X=X_seq, Y=Y_df))
    return out

def derive_out_path(base_excel: str | Path, window: int, horizon: int) -> Path:
    p = Path(base_excel)
    p.parent.mkdir(parents=True, exist_ok=True)
    stem = p.stem
    return p.with_name(f"{stem}_W{window}_H{horizon}_Outer10CV.xlsx")

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

# ---------- 主过程：外层10折 ----------
def run_outer10cv_eval_fixed_params():
    raw = load_daily_table(DATA_PATH)
    df, bcol, scol = normalize_df(raw)
    feat_cols = pick_features(df, bcol, scol)
    samples = build_all_samples(df, bcol, scol, feat_cols, WINDOW, HORIZON)
    if not samples:
        raise RuntimeError(f"无可用样本（窗口={WINDOW}）")
    X_all, Y_all, groups_all = build_Xy_groups(samples)

    uniq_pigs = np.unique(groups_all)
    n_splits = min(10, len(uniq_pigs))
    gkf = GroupKFold(n_splits=n_splits)

    out_file = derive_out_path(OUT_EXCEL, WINDOW, HORIZON)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    used_names = set()
    all_val_rows = []
    split_records = []
    fold_macro_list = []

    with pd.ExcelWriter(out_file, engine="xlsxwriter") as writer:
        fold = 0
        for tr_idx, te_idx in gkf.split(X_all, Y_all, groups_all):
            fold += 1
            val_pigs = np.unique(groups_all[te_idx])
            tr_pigs  = np.unique(groups_all[tr_idx])
            tr_s = [s for s in samples if s["pid"] in tr_pigs]
            va_s = [s for s in samples if s["pid"] in val_pigs]
            Xtr, Ytr, _ = build_Xy_groups(tr_s)

            # 外层重训（不用外层验证 → 无早停）
            model, _ = train_one_model(Xtr, Ytr, params=BEST_PARAMS)

            # 预测与逐猪评估
            yhat_tr = [model_predict(model, s["X"]) for s in tr_s]
            yhat_va = [model_predict(model, s["X"]) for s in va_s]

            df_train = eval_by_pig_named(tr_s, yhat_tr, set_name=f"Train_fold{fold}", horizon=HORIZON)
            df_val   = eval_by_pig_named(va_s, yhat_va, set_name=f"Val_fold{fold}",   horizon=HORIZON)
            df_all   = pd.concat([df_train, df_val], ignore_index=True)

            # 每折工作表
            sheet_name = f"Fold{fold}"
            df_all.to_excel(writer, sheet_name=sheet_name[:31], index=False)

            # 每折（传统）汇总（可选）
            add_summary_sheet(writer, df_all, sheet_name=f"{sheet_name}_summary_train"[:31], set_filter=f"Train_fold{fold}")
            add_summary_sheet(writer, df_all, sheet_name=f"{sheet_name}_summary_val"[:31],   set_filter=f"Val_fold{fold}")

            # 记录分割
            split_records.append(dict(Fold=f"Fold{fold}",
                                      ValPigs=",".join(sorted(list(val_pigs))),
                                      TrainPigs=",".join(sorted(list(tr_pigs))),
                                      Selected=str(BEST_PARAMS)))

            # 宏平均（Val）本折值：逐猪RMSE后取平均
            fold_macro_mean = float(df_val["RMSE"].mean()) if not df_val.empty else np.nan
            fold_macro_list.append(fold_macro_mean)

            all_val_rows.append(df_val)

        # 汇总全部 Val（逐猪）
        if all_val_rows:
            df_val_all = pd.concat(all_val_rows, ignore_index=True)
            df_val_all.to_excel(writer, sheet_name="Val_AllFolds", index=False)

            # 1) Overall（Val）：逐猪宏平均
            macro_overall = macro_over_pigs_overall(df_val_all)
            macro_overall.to_excel(writer, sheet_name="MacroOverall_Val", index=False)

            # 2) Per-horizon（Val）：逐猪宏平均 + h_avg
            macro_hor = macro_over_pigs_horizon(df_val_all, HORIZON)
            macro_hor.to_excel(writer, sheet_name="MacroHorizon_Val", index=False)

            # 3) 稳健性（STABILITY）：逐折 Val 宏平均 RMSE 列表 + 折间统计
            df_stab_list, df_stab_sum = stability_over_folds(fold_macro_list)
            df_stab_list.to_excel(writer, sheet_name="STABILITY_Val", index=False, startrow=0)
            df_stab_sum.to_excel(writer, sheet_name="STABILITY_Val", index=False, startrow=len(df_stab_list)+2)

        # 分割信息
        if split_records:
            pd.DataFrame(split_records).to_excel(writer, sheet_name=make_safe_sheetname("splits_info", used=used_names), index=False)

    print(f"✅ 外层按猪 {n_splits} 折（固定超参, IEBlock）评估完成：{out_file}")

# ---------- 主入口 ----------
if __name__ == "__main__":
    run_outer10cv_eval_fixed_params()
