"""6시간 정의, onset 직전 구간의 기초통계(평균·표준편차)가 강도 위에 올리는지.
구간은 wavelet과 동일(IVT 16일, 제트/블로킹 64일). AR run이 아니라 onset 직전 연속 구간.
IVT통계(강도와 얽힘=순환논리 위험) vs 순환통계(jet/blk, 안전)를 분리.
사용: python stats_6h_region.py [ca|uk|chile]   (TABPFN_TOKEN 필요)"""
import torch
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
Fa = []; FaI = []; FaC = []; Fd = []; Fall = []; steps = []
for s, e in runs:
    o = s // 4
    if s - 63 >= 0 and o - 63 >= 0 and o < ND and not (np.isnan(jet[o]) or np.isnan(blk[o])):
        iw = ivt[s - 63:s + 1]; jwd = jet[o - 63:o + 1]; bwd = blk[o - 63:o + 1]
        sI = [iw.mean(), iw.std()]; sJ = [jwd.mean(), jwd.std()]; sB = [bwd.mean(), bwd.std()]
        d = wl(iw, 5) + [dmax[o], jet[o]] + wl(jwd, 5) + wl(bwd, 5)   # 기존 전체
        Fa.append([dmax[o]])
        FaI.append([dmax[o]] + sI)              # 강도 + IVT통계
        FaC.append([dmax[o]] + sJ + sB)         # 강도 + 순환통계
        Fd.append(d)                            # 기존 전체(wavelet+강도+순환)
        Fall.append(d + sI + sJ + sB)           # 기존 전체 + 통계
        steps.append(e - s)
Fa, FaI, FaC, Fd, Fall = map(np.array, (Fa, FaI, FaC, Fd, Fall)); steps = np.array(steps)
n = len(steps); i1 = int(n * 0.6); i2 = int(n * 0.8)
from tabpfn import TabPFNClassifier
def auc_lr(X, y):
    sc = StandardScaler().fit(X[:i1])
    m = LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced").fit(sc.transform(X[:i1]), y[:i1])
    return roc_auc_score(y[i2:], m.predict_proba(sc.transform(X[i2:]))[:, 1])
def auc_tp(X, y):
    m = TabPFNClassifier(random_state=0, ignore_pretraining_limits=True).fit(X[:i1], y[:i1])
    return roc_auc_score(y[i2:], m.predict_proba(X[i2:])[:, 1])
print(f"\n[{REGION}] 6h n={n}, AR임계(85th)={THR:.0f}  (평균·표준편차, 구간=wavelet과 동일)")
for name, auc in [("Logistic", auc_lr), ("TabPFN", auc_tp)]:
    print(f"\n  === {name} ===")
    print(f"  {'지속':>5} {'양성%':>5} | {'(a)강도':>7} {'+IVT통':>7} {'+순환통':>7} | {'(d)기존':>7} {'(d)+통계':>8} | {'test양성':>6}")
    for k in [4, 6, 8]:    # 24h,36h,48h
        y = (steps >= k).astype(int)
        if len(np.unique(y[i2:])) < 2: print(f"  {k*6:>4}h 클래스부족"); continue
        a, ai, ac = auc(Fa, y), auc(FaI, y), auc(FaC, y)
        d, dall = auc(Fd, y), auc(Fall, y)
        print(f"  {k*6:>4}h {y.mean()*100:>4.0f}% | {a:>7.3f} {ai:>7.3f} {ac:>7.3f} | {d:>7.3f} {dall:>8.3f} | {int(y[i2:].sum()):>6}")
