# ===== Colab: 예보온셋 분모(TT+FT, 운영정합)에서 모델 비교 — CA·UK(·Chile) =====
# train/test 둘 다 예보온셋(TT+FT). 회귀 omin -> THR 판정 -> F1. 입력 opdenom_{r}.npz.
# 준비: opdenom_{ca,uk}.npz (git pull). !pip install tabpfn lightgbm pytorch-tabnet -q
import numpy as np, torch, torch.nn as nn, os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from lightgbm import LGBMRegressor
from tabpfn import TabPFNRegressor
from pytorch_tabnet.tab_model import TabNetRegressor
DEV = "cuda" if torch.cuda.is_available() else "cpu"; NB = 2000; REGIONS = ["ca", "uk", "chile"]


def p_lr(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr); return LinearRegression().fit(sc.transform(Xtr), ytr).predict(sc.transform(Xte))
def p_lgbm(Xtr, ytr, Xte):
    return LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20, subsample=0.8, verbose=-1).fit(Xtr, ytr).predict(Xte)
def p_tabpfn(Xtr, ytr, Xte):
    m = TabPFNRegressor(device=DEV); m.fit(np.nan_to_num(Xtr), ytr); return np.asarray(m.predict(np.nan_to_num(Xte)))
def p_tabnet(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr); bs = max(16, len(Xtr) // 4)
    m = TabNetRegressor(verbose=0, device_name=DEV, seed=0)
    m.fit(sc.transform(Xtr).astype("float32"), ytr.reshape(-1, 1).astype("float32"), max_epochs=150, batch_size=bs, virtual_batch_size=bs, drop_last=True)
    return m.predict(sc.transform(Xte).astype("float32")).ravel()
class LSTMreg(nn.Module):
    def __init__(s, h=16): super().__init__(); s.lstm = nn.LSTM(1, h, batch_first=True); s.fc = nn.Linear(h, 1)
    def forward(s, x): o, _ = s.lstm(x); return s.fc(o[:, -1]).squeeze(-1)
def p_lstm(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr[:, :5])
    xtr = torch.tensor(sc.transform(Xtr[:, :5])[:, :, None], dtype=torch.float32, device=DEV)
    xte = torch.tensor(sc.transform(Xte[:, :5])[:, :, None], dtype=torch.float32, device=DEV)
    ym, ys = float(ytr.mean()), float(ytr.std() + 1e-8); ytn = torch.tensor((ytr - ym) / ys, dtype=torch.float32, device=DEV)
    torch.manual_seed(0); net = LSTMreg().to(DEV); opt = torch.optim.Adam(net.parameters(), lr=1e-2); lf = nn.MSELoss()
    rng = np.random.default_rng(0); n = len(ytr)
    for ep in range(200):
        net.train(); perm = rng.permutation(n)
        for b in range(0, n, 128):
            bi = torch.as_tensor(perm[b:b + 128], device=DEV); opt.zero_grad(); lf(net(xtr[bi]), ytn[bi]).backward(); opt.step()
    net.eval()
    with torch.no_grad(): return net(xte).cpu().numpy() * ys + ym
MODELS = {"LR": p_lr, "LGBM": p_lgbm, "LSTM": p_lstm, "TabPFN": p_tabpfn, "TabNet": p_tabnet}


def folds(N, od, Nf=5, emb=64):
    f = N // (Nf + 1); out = []
    for k in range(1, Nf + 1):
        ts = k * f; te = (k + 1) * f if k < Nf else N
        out.append((np.array([j for j in range(0, ts) if od[j] <= od[ts] - emb]), np.arange(ts, te)))
    return out
def raw_preds(fcv, y, od, thr):
    yt = []; pb = []; sc = []
    for tr, te in folds(len(y), od):
        if len(tr) < 40 or len(np.unique(y[tr])) < 2: continue
        pb.extend((fcv[te] >= thr).astype(int)); sc.extend(fcv[te]); yt.extend(y[te])
    return np.array(yt), np.array(pb), np.array(sc)
def model_preds(pf, X, omin, y, od, thr):
    yt = []; pb = []; sc = []
    for tr, te in folds(len(y), od):
        if len(tr) < 40 or len(np.unique(y[tr])) < 2: continue
        pred = np.asarray(pf(X[tr], omin[tr], X[te]))
        pb.extend((pred >= thr).astype(int)); sc.extend(pred); yt.extend(y[te])
    return np.array(yt), np.array(pb), np.array(sc)
def boot(yt, pa, pb_):
    rng = np.random.default_rng(0); dd = []
    for _ in range(NB):
        ix = rng.integers(0, len(yt), len(yt))
        if len(np.unique(yt[ix])) > 1: dd.append(f1_score(yt[ix], pa[ix], zero_division=0) - f1_score(yt[ix], pb_[ix], zero_division=0))
    dd = np.array(dd); return np.percentile(dd, 2.5), np.percentile(dd, 97.5), np.mean(dd > 0)


REG = {}
for R in REGIONS:
    if not os.path.exists(f"opdenom_{R}.npz"): print(f"[{R}] opdenom 없음 -> skip"); continue
    d = np.load(f"opdenom_{R}.npz"); D2 = d["D2"]; fcv = d["fcv"]; y = d["y"]; omin = d["omin"]; oday = d["oday"]; THR = float(d["THR"])
    yt0, p0, s0 = raw_preds(fcv, y, oday, THR); f0 = f1_score(yt0, p0, zero_division=0)
    print(f"[{R}] n={len(y)} 지속률={y.mean():.3f} THR={THR:.1f} | 기준 raw F1={f0:.3f} AUC={roc_auc_score(yt0, s0):.3f}")
    REG[R] = {"_raw": (yt0, p0)}
    for name, pf in MODELS.items():
        yt, pb, sc = model_preds(pf, D2, omin, y, oday, THR)
        lo, hi, pp = boot(yt0, pb, p0)
        print(f"  {name:7s}: F1={f1_score(yt, pb, zero_division=0):.3f} AUC={roc_auc_score(yt, sc):.3f} AUPRC={average_precision_score(yt, sc):.3f}  ΔF1={f1_score(yt, pb, zero_division=0)-f0:+.3f} CI[{lo:+.3f},{hi:+.3f}] P={pp:.3f}")
        REG[R][name] = (yt, pb)

# ===== 통합검정: 모델별 평균 ΔF1(지역 부트스트랩) =====
print("\n[통합 ΔF1(vs raw), 지역 평균]")
for name in MODELS:
    rng = np.random.default_rng(0); md = []
    for _ in range(NB):
        ds = []
        for R in REG:
            yt0, p0 = REG[R]["_raw"]; _, pm = REG[R][name]; ix = rng.integers(0, len(yt0), len(yt0))
            if len(np.unique(yt0[ix])) < 2: ds = None; break
            ds.append(f1_score(yt0[ix], pm[ix], zero_division=0) - f1_score(yt0[ix], p0[ix], zero_division=0))
        if ds is not None: md.append(np.mean(ds))
    md = np.array(md)
    obs = np.mean([f1_score(REG[R]["_raw"][0], REG[R][name][1], zero_division=0) - f1_score(REG[R]["_raw"][0], REG[R]["_raw"][1], zero_division=0) for R in REG])
    print(f"  {name:7s} 평균ΔF1={obs:+.3f}  95%CI[{np.percentile(md,2.5):+.3f},{np.percentile(md,97.5):+.3f}]  P(Δ>0)={np.mean(md>0):.3f}")
