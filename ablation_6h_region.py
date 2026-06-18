"""6시간 정의에서 강도 위에 wavelet/순환이 기여하는지 분해.
(a) 강도단독  (b) 강도+IVT파형  (c) 강도+순환  (d) 전체.  LR + TabPFN.
Δwav=(b)-(a), Δcirc=(c)-(a), Δfull=(d)-(a).
사용: python ablation_6h_region.py [ca|uk|chile|nz]   (TABPFN_TOKEN 필요)"""
import torch  # 먼저 import (로컬 c10.dll)
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
Fa = []; Fb = []; Fc = []; Fd = []; steps = []
for s, e in runs:
    o = s // 4
    if s - 63 >= 0 and o - 63 >= 0 and o < ND and not (np.isnan(jet[o]) or np.isnan(blk[o])):
        ivw = wl(ivt[s - 63:s + 1], 5); jw = wl(jet[o - 63:o + 1], 5); bw = wl(blk[o - 63:o + 1], 5)
        Fa.append([dmax[o]])                              # 강도단독
        Fb.append(ivw + [dmax[o]])                        # 강도 + IVT파형
        Fc.append([dmax[o], jet[o]] + jw + bw)            # 강도 + 순환(IVT파형 없음)
        Fd.append(ivw + [dmax[o], jet[o]] + jw + bw)      # 전체
        steps.append(e - s)
Fa, Fb, Fc, Fd = map(np.array, (Fa, Fb, Fc, Fd)); steps = np.array(steps)
n = len(steps); i1 = int(n * 0.6); i2 = int(n * 0.8)
from tabpfn import TabPFNClassifier
def auc_lr(X, y):
    sc = StandardScaler().fit(X[:i1])
    m = LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced").fit(sc.transform(X[:i1]), y[:i1])
    return roc_auc_score(y[i2:], m.predict_proba(sc.transform(X[i2:]))[:, 1])
def auc_tp(X, y):
    m = TabPFNClassifier(random_state=0, ignore_pretraining_limits=True).fit(X[:i1], y[:i1])
    return roc_auc_score(y[i2:], m.predict_proba(X[i2:])[:, 1])
print(f"\n[{REGION}] 6h run n={n}, AR임계(85th)={THR:.0f}  (Δwav=파형기여, Δcirc=순환기여, Δfull=전체기여)")
for name, auc in [("Logistic", auc_lr), ("TabPFN", auc_tp)]:
    print(f"\n  === {name} ===")
    print(f"  {'지속':>5} {'양성%':>5} | {'(a)강도':>7} {'Δwav':>7} {'Δcirc':>7} {'Δfull':>7} | {'test양성':>6}")
    for k in [2, 3, 4, 5, 6, 7, 8]:    # 12h..48h
        y = (steps >= k).astype(int)
        if len(np.unique(y[i2:])) < 2: print(f"  {k*6:>4}h 클래스부족"); continue
        a, b, c, d = auc(Fa, y), auc(Fb, y), auc(Fc, y), auc(Fd, y)
        print(f"  {k*6:>4}h {y.mean()*100:>4.0f}% | {a:>7.3f} {b-a:>+7.3f} {c-a:>+7.3f} {d-a:>+7.3f} | {int(y[i2:].sum()):>6}")
