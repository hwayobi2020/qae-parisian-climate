# ===== Colab: CA 예보기준 분모(TT+TF+FT, 헛부름 포함)에서 모델 비교 =====
# 입력 opdenom_ca.npz (D2 예보피처9, fcv, y, omin, oday, THR). 모델: 회귀 omin -> THR 판정 -> F1.
# 준비: opdenom_ca.npz (git pull). !pip install tabpfn lightgbm pytorch-tabnet -q
import numpy as np, torch, torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from lightgbm import LGBMRegressor
from tabpfn import TabPFNRegressor
from pytorch_tabnet.tab_model import TabNetRegressor
DEV = "cuda" if torch.cuda.is_available() else "cpu"; NB = 2000


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


d = np.load("opdenom_ca.npz"); D2 = d["D2"]; fcv = d["fcv"]; y = d["y"]; omin = d["omin"]; oday = d["oday"]; THR = float(d["THR"])
print(f"[CA 예보기준 분모] n={len(y)} 지속률={y.mean():.3f} THR={THR:.1f}")
yt0, p0, s0 = raw_preds(fcv, y, oday, THR); f0 = f1_score(yt0, p0, zero_division=0)
print(f"  기준 raw : F1={f0:.3f} AUC={roc_auc_score(yt0, s0):.3f} AUPRC={average_precision_score(yt0, s0):.3f}")
for name, pf in MODELS.items():
    yt, pb, sc = model_preds(pf, D2, omin, y, oday, THR)
    lo, hi, pp = boot(yt0, pb, p0)
    print(f"  {name:7s}: F1={f1_score(yt, pb, zero_division=0):.3f} AUC={roc_auc_score(yt, sc):.3f} AUPRC={average_precision_score(yt, sc):.3f}  ΔF1(vs raw)={f1_score(yt, pb, zero_division=0)-f0:+.3f} CI[{lo:+.3f},{hi:+.3f}] P={pp:.3f}")
