"""6시간 정의(12h부터)에서 '강도→지속 자명함' 검증, Logistic + TabPFN.
(a) onset IVT 강도 단독  vs  (c) 전체 20피처.  (c)-(a)가 0이면 강도가 전부(자명).
사용: python baseline_6h_region.py [ca|uk|chile]   (TABPFN_TOKEN 환경변수 필요)"""
import torch  # 반드시 먼저 import (로컬 c10.dll 로드 순서)
import os, sys, numpy as np, pywt, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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
Fa = []; Fc = []; steps = []
for s, e in runs:
    o = s // 4
    if s - 63 >= 0 and o - 63 >= 0 and o < ND and not (np.isnan(jet[o]) or np.isnan(blk[o])):
        Fa.append([dmax[o]])
        Fc.append(wl(ivt[s - 63:s + 1], 5) + [dmax[o], jet[o]] + wl(jet[o - 63:o + 1], 5) + wl(blk[o - 63:o + 1], 5))
        steps.append(e - s)
Fa = np.array(Fa); Fc = np.array(Fc); steps = np.array(steps)
n = len(steps); i1 = int(n * 0.6); i2 = int(n * 0.8)
from tabpfn import TabPFNClassifier
def auc_lr(X, y):
    sc = StandardScaler().fit(X[:i1])
    m = LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced").fit(sc.transform(X[:i1]), y[:i1])
    return roc_auc_score(y[i2:], m.predict_proba(sc.transform(X[i2:]))[:, 1])
def auc_tp(X, y):
    m = TabPFNClassifier(random_state=0, ignore_pretraining_limits=True).fit(X[:i1], y[:i1])
    return roc_auc_score(y[i2:], m.predict_proba(X[i2:])[:, 1])
print(f"\n[{REGION}] 6h run n={n}, AR임계(85th)={THR:.0f}")
print(f"{'지속임계':>7} {'양성%':>5} | {'(a)LR':>6} {'(c)LR':>6} {'ΔLR':>6} | {'(a)Tab':>7} {'(c)Tab':>7} {'ΔTab':>7} | {'test양성':>6}")
for k in [2, 3, 4, 5, 6, 7, 8, 9, 10]:   # 12h,18h,24h,30h,...,60h (6시간 간격)
    y = (steps >= k).astype(int)
    if len(np.unique(y[i2:])) < 2: print(f"{k*6:>5}h 클래스부족"); continue
    aL, cL = auc_lr(Fa, y), auc_lr(Fc, y)
    aT, cT = auc_tp(Fa, y), auc_tp(Fc, y)
    print(f"{k*6:>5}h {y.mean()*100:>4.0f}% | {aL:>6.3f} {cL:>6.3f} {cL-aL:>+6.3f} | {aT:>7.3f} {cT:>7.3f} {cT-aT:>+7.3f} | {int(y[i2:].sum()):>6}")
