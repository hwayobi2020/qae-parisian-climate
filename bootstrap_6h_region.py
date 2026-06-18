"""6시간 정의, 강도 위 기여(Δwav/Δcirc/Δfull)의 부트스트랩 신뢰구간 (TabPFN).
모델은 train으로 1회 fit, test 예측확률을 복원추출(B=2000)해 Δ의 CI/P(>0) 산출.
사용: python bootstrap_6h_region.py [ca|uk|chile|nz]   (TABPFN_TOKEN 필요)"""
import torch  # 먼저 import
import os, sys, numpy as np, pywt, warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
REGION = sys.argv[1] if len(sys.argv) > 1 else "ca"
IVT_FILE = {"ca": "ivt_sf_1980_2023.npy"}.get(REGION, f"ivt_{REGION}_1980_2023.npy")
CIRC_FILE = {"ca": "circ_indices.npz"}.get(REGION, f"circ_indices_{REGION}.npz")
def find(name):
    for p in [os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw", name),
              os.path.join("data/raw", name), "/content/" + name,
              os.path.join("/content/qae-parisian-climate/data/raw", name)]:
        if os.path.exists(p): return p
    raise FileNotFoundError(name)
ivt = np.load(find(IVT_FILE)).astype("float64")
ci = np.load(find(CIRC_FILE)); jet = ci["jet"].astype("float64"); blk = ci["blocking"].astype("float64")
dmax = ivt.reshape(-1, 4).max(1); ND = len(dmax); THR = np.percentile(dmax, 85)
T = len(ivt); ar6 = ivt > THR
def wl(a, lvl): return [c[-1] for c in pywt.swt(a, 'db2', level=lvl, trim_approx=True, norm=True)]
runs = []; i = 0
while i < T:
    if ar6[i]:
        j = i
        while j < T and ar6[j]: j += 1
        runs.append((i, j)); i = j
    else: i += 1
Fa = []; Fb = []; Fc = []; Fd = []; steps = []
for s, e in runs:
    o = s // 4
    if s - 63 >= 0 and o - 63 >= 0 and o < ND and not (np.isnan(jet[o]) or np.isnan(blk[o])):
        ivw = wl(ivt[s - 63:s + 1], 5); jw = wl(jet[o - 63:o + 1], 5); bw = wl(blk[o - 63:o + 1], 5)
        Fa.append([dmax[o]]); Fb.append(ivw + [dmax[o]])
        Fc.append([dmax[o], jet[o]] + jw + bw); Fd.append(ivw + [dmax[o], jet[o]] + jw + bw)
        steps.append(e - s)
Fa, Fb, Fc, Fd = map(np.array, (Fa, Fb, Fc, Fd)); steps = np.array(steps)
n = len(steps); i1 = int(n * 0.6); i2 = int(n * 0.8)
from tabpfn import TabPFNClassifier
def pred(X, y):
    m = TabPFNClassifier(random_state=0, ignore_pretraining_limits=True).fit(X[:i1], y[:i1])
    return m.predict_proba(X[i2:])[:, 1]
rng = np.random.RandomState(0); B = 2000
print(f"\n[{REGION}] 6h run n={n}, AR임계(85th)={THR:.0f}, TabPFN, 부트스트랩 B={B}")
for k in [4, 6, 8]:    # 24h, 36h, 48h
    y = (steps >= k).astype(int); yte = y[i2:]
    if len(np.unique(yte)) < 2: print(f"  {k*6}h 클래스부족"); continue
    Pa, Pb, Pc, Pd = pred(Fa, y), pred(Fb, y), pred(Fc, y), pred(Fd, y)
    dw, dc, df = [], [], []
    for _ in range(B):
        idx = rng.randint(0, len(yte), len(yte))
        if len(np.unique(yte[idx])) < 2: continue
        a = roc_auc_score(yte[idx], Pa[idx])
        dw.append(roc_auc_score(yte[idx], Pb[idx]) - a)
        dc.append(roc_auc_score(yte[idx], Pc[idx]) - a)
        df.append(roc_auc_score(yte[idx], Pd[idx]) - a)
    print(f"\n  [{k*6}h] test양성={int(yte.sum())}/{len(yte)}, (a)강도 AUC={roc_auc_score(yte,Pa):.3f}")
    for nm, d in [("Δwav ", dw), ("Δcirc", dc), ("Δfull", df)]:
        d = np.array(d); pg = (d > 0).mean()
        flag = "*유의*" if pg >= 0.975 else ("동급" if pg > 0.5 else "효과없음")
        print(f"    {nm}: 중앙{np.median(d):+.3f} 95%CI[{np.percentile(d,2.5):+.3f},{np.percentile(d,97.5):+.3f}] P(>0)={pg:.2f} {flag}")
