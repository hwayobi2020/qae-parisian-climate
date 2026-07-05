# ===== Colab: 모델 ablation — D-2 예보피처로 omin 회귀 → THR 판정 → F1 =====
# 프레임 고정(예보 9피처 → 관측 min IVT 회귀 → 예측≥THR 판정). 모델만 교체: LR/LGBM/LSTM/TabPFN/TabNet.
# 준비: transfer_{r}.npz + d2feat_{r}.npz + d2env_{r}.npz (git pull).
#   !pip install tabpfn lightgbm pytorch-tabnet -q
import numpy as np, torch, torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from lightgbm import LGBMRegressor
from tabpfn import TabPFNRegressor
from pytorch_tabnet.tab_model import TabNetRegressor
DEV = "cuda" if torch.cuda.is_available() else "cpu"; FEAT = ""; NB = 2000


# ---------- 모델별 예측함수: (Xtr, ytr, Xte) -> 연속 omin 예측 ----------
def p_lr(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr); m = LinearRegression().fit(sc.transform(Xtr), ytr)
    return m.predict(sc.transform(Xte))


def p_lgbm(Xtr, ytr, Xte):
    m = LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20, subsample=0.8, verbose=-1)
    m.fit(Xtr, ytr); return m.predict(Xte)


def p_tabpfn(Xtr, ytr, Xte):
    m = TabPFNRegressor(device=DEV); m.fit(np.nan_to_num(Xtr), ytr)
    return np.asarray(m.predict(np.nan_to_num(Xte)))


def p_tabnet(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    bs = max(16, len(Xtr) // 4)                       # ~4배치, 배치크기 1 방지
    m = TabNetRegressor(verbose=0, device_name=DEV, seed=0)
    m.fit(sc.transform(Xtr).astype("float32"), ytr.reshape(-1, 1).astype("float32"),
          max_epochs=150, batch_size=bs, virtual_batch_size=bs, drop_last=True)   # drop_last로 크기1 잔여배치 제거
    return m.predict(sc.transform(Xte).astype("float32")).ravel()


class LSTMreg(nn.Module):
    def __init__(s, h=16):
        super().__init__(); s.lstm = nn.LSTM(1, h, batch_first=True); s.fc = nn.Linear(h, 1)
    def forward(s, x):
        o, _ = s.lstm(x); return s.fc(o[:, -1]).squeeze(-1)


def p_lstm(Xtr, ytr, Xte):                       # 5점 궤적(#1~5)을 길이5 시퀀스로
    sc = StandardScaler().fit(Xtr[:, :5])
    xtr = torch.tensor(sc.transform(Xtr[:, :5])[:, :, None], dtype=torch.float32, device=DEV)
    xte = torch.tensor(sc.transform(Xte[:, :5])[:, :, None], dtype=torch.float32, device=DEV)
    ym, ys = float(ytr.mean()), float(ytr.std() + 1e-8)
    ytn = torch.tensor((ytr - ym) / ys, dtype=torch.float32, device=DEV)
    torch.manual_seed(0); net = LSTMreg().to(DEV); opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    lf = nn.MSELoss(); rng = np.random.default_rng(0); n = len(ytr)
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


def raw_preds(fcv, yf, odf, thr):
    yt = []; pb = []; sc = []
    for tr, te in folds(len(yf), odf):
        if len(tr) < 40 or len(np.unique(yf[tr])) < 2: continue
        pb.extend((fcv[te] >= thr).astype(int)); sc.extend(fcv[te]); yt.extend(yf[te])
    return np.array(yt), np.array(pb), np.array(sc)


def model_preds(pf, X, omin, yf, odf, thr):
    yt = []; pb = []; sc = []
    for tr, te in folds(len(yf), odf):
        if len(tr) < 40 or len(np.unique(yf[tr])) < 2: continue
        pred = np.asarray(pf(X[tr], omin[tr], X[te]))
        pb.extend((pred >= thr).astype(int)); sc.extend(pred); yt.extend(yf[te])
    return np.array(yt), np.array(pb), np.array(sc)


REG = {R: {} for R in ["ca", "uk", "chile"]}
for R in ["ca", "uk", "chile"]:
    d = np.load(FEAT + f"transfer_{R}.npz"); y, oday, S = d["y"], d["oday"], d["s"].astype(int)
    de = np.load(FEAT + f"d2env_{R}.npz"); THR = float(de["THR"]); OMIN = de["omin"]
    z = np.load(FEAT + f"d2feat_{R}.npz"); s2 = z["s"].astype(int); D2 = z["D2"]; D2min = z["D2min"]
    d2map = {int(s2[k]): k for k in range(len(s2))}
    keep = [i for i in range(len(S)) if int(S[i]) in d2map and not np.isnan(OMIN[i])]
    ridx = np.array([d2map[int(S[i])] for i in keep])
    fcv = D2min[ridx]; D2s = D2[ridx]; y_s = y[keep]; od_s = oday[keep]; omin_s = OMIN[keep]
    yt0, p0, s0 = raw_preds(fcv, y_s, od_s, THR); f0 = f1_score(yt0, p0, zero_division=0)
    a0 = roc_auc_score(yt0, s0); ap0 = average_precision_score(yt0, s0)
    print(f"[{R}] n={len(keep)} 양성률={y_s.mean():.2f} THR={THR:.1f}")
    print(f"  기준 raw : F1={f0:.3f} AUC={a0:.3f} AUPRC={ap0:.3f}")
    REG[R]["_raw"] = (yt0, p0)
    for name, pf in MODELS.items():
        yt, pb, sc = model_preds(pf, D2s, omin_s, y_s, od_s, THR)
        f1v = f1_score(yt, pb, zero_division=0); a1 = roc_auc_score(yt, sc); ap1 = average_precision_score(yt, sc)
        print(f"  {name:7s}: F1={f1v:.3f} AUC={a1:.3f} AUPRC={ap1:.3f}  ΔF1(vs raw)={f1v - f0:+.3f}")
        REG[R][name] = (yt, pb)

# ---------- 모델별 통합 ΔF1 (3지역 평균, 부트스트랩) ----------
print("\n[통합 ΔF1(vs 기준 raw), 3지역 평균]")
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
    print(f"  {name:7s} 평균ΔF1={obs:+.3f}  95%CI[{np.percentile(md, 2.5):+.3f},{np.percentile(md, 97.5):+.3f}]  P(Δ>0)={np.mean(md > 0):.3f}")
