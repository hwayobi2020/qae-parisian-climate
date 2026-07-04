# ===== Colab: D-2(2일전) 예보 프레임 TabPFN =====
# 기준 = raw D-2 예보 min IVT 임계 / 우리모델 = TabPFN(풍부한 D-2 예보피처 9개). env 없음.
# 질문: 우리모델이 raw min 예보보다 잘 맞추나(MCC).  준비: transfer_{r}.npz + d2feat_{r}.npz (git pull). !pip install tabpfn -q
import numpy as np
from tabpfn import TabPFNClassifier
from sklearn.metrics import matthews_corrcoef
DEV = "cuda"; FEAT = ""; NB = 3000


def folds(N, od, Nf=5, emb=64):
    f = N // (Nf + 1); out = []
    for k in range(1, Nf + 1):
        ts = k * f; te = (k + 1) * f if k < Nf else N
        out.append((np.array([j for j in range(0, ts) if od[j] <= od[ts] - emb]), np.arange(ts, te)))
    return out


def bt(yy, pp):
    b = (-2., .5)
    for t in np.unique(pp):
        m = matthews_corrcoef(yy, (pp >= t).astype(int))
        if m > b[0]: b = (m, t)
    return b[1]


def raw_preds(fcv, yf, odf):
    yt = []; pb = []
    for tr, te in folds(len(yf), odf):
        if len(tr) < 40 or len(np.unique(yf[tr])) < 2: continue
        th = bt(yf[tr], fcv[tr]); pb.extend((fcv[te] >= th).astype(int)); yt.extend(yf[te])
    return np.array(yt), np.array(pb)


def tp_preds(X, yf, odf):
    yt = []; pb = []
    for tr, te in folds(len(yf), odf):
        if len(tr) < 40 or len(np.unique(yf[tr])) < 2: continue
        m = TabPFNClassifier(device=DEV); m.fit(np.nan_to_num(X[tr]), yf[tr])
        ptr = m.predict_proba(np.nan_to_num(X[tr]))[:, 1]; pte = m.predict_proba(np.nan_to_num(X[te]))[:, 1]
        th = bt(yf[tr], ptr); pb.extend((pte >= th).astype(int)); yt.extend(yf[te])
    return np.array(yt), np.array(pb)


def boot(yt, pa, pb_):   # Δ = MCC(a) - MCC(b)
    rng = np.random.default_rng(0); dd = []
    for _ in range(NB):
        ix = rng.integers(0, len(yt), len(yt))
        if len(np.unique(yt[ix])) > 1: dd.append(matthews_corrcoef(yt[ix], pa[ix]) - matthews_corrcoef(yt[ix], pb_[ix]))
    dd = np.array(dd); return np.percentile(dd, 2.5), np.percentile(dd, 97.5), np.mean(dd > 0)


for R in ["ca", "uk", "chile"]:
    d = np.load(FEAT + f"transfer_{R}.npz"); y, oday, S = d["y"], d["oday"], d["s"].astype(int)
    z = np.load(FEAT + f"d2feat_{R}.npz"); s2 = z["s"].astype(int); D2 = z["D2"]; D2min = z["D2min"]
    d2map = {int(s2[k]): k for k in range(len(s2))}
    keep = [i for i in range(len(S)) if int(S[i]) in d2map]
    ridx = np.array([d2map[int(S[i])] for i in keep])
    fcv = D2min[ridx]; D2s = D2[ridx]; y_s = y[keep]; od_s = oday[keep]
    print(f"[{R}] n={len(keep)} 양성률={y_s.mean():.2f} fcv-y상관={np.corrcoef(fcv, y_s)[0,1]:+.2f}")
    yt0, p0 = raw_preds(fcv, y_s, od_s); m0 = matthews_corrcoef(yt0, p0)
    yt1, p1 = tp_preds(D2s, y_s, od_s); m1 = matthews_corrcoef(yt1, p1)
    lo, hi, pp = boot(yt0, p1, p0)
    print(f"  기준 raw D-2 min MCC={m0:.3f} | 우리모델 TabPFN MCC={m1:.3f} | Δ={m1-m0:+.3f} 95%CI[{lo:+.3f},{hi:+.3f}] P(Δ>0)={pp:.3f}")
