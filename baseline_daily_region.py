"""일별 정의 '강도 baseline 대비 방법론 추가기여' 검증, Logistic + TabPFN.
(a) onset IVT 강도 단독  vs  (c) 전체 20피처.  (c)-(a)가 크면 방법론이 강도 이상을 함.
로컬: torch 깨져 TabPFN 자동 스킵(LR만). Colab: TabPFN 포함.
사용: python baseline_daily_region.py [ca|uk|chile]"""
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
dmax = ivt.reshape(-1, 4).max(1); ND = len(dmax); THR = np.percentile(dmax, 85); ar = dmax > THR
def wl(a, lvl): return [c[-1] for c in pywt.swt(a, 'db2', level=lvl, trim_approx=True, norm=True)]
Fa = []; Fc = []; durs = []; i = 0
while i < ND:
    if ar[i]:
        j = i
        while j < ND and ar[j]: j += 1
        o = i; dur = j - i; end6 = (o + 1) * 4
        if end6 - 64 >= 0 and o - 63 >= 0 and not (np.isnan(jet[o]) or np.isnan(blk[o])):
            ivw = wl(ivt[end6 - 64:end6], 5)
            Fa.append([dmax[o]])
            Fc.append(ivw + [dmax[o], jet[o]] + wl(jet[o - 63:o + 1], 5) + wl(blk[o - 63:o + 1], 5))
            durs.append(dur)
        i = j
    else: i += 1
Fa = np.array(Fa); Fc = np.array(Fc); durs = np.array(durs)
n = len(durs); i1 = int(n * 0.6); i2 = int(n * 0.8)
try:
    from tabpfn import TabPFNClassifier; HAVE_TP = True
except Exception as e:
    HAVE_TP = False; print(f"[TabPFN 미설치: LR만] {str(e)[:50]}")
def auc_lr(X, y):
    sc = StandardScaler().fit(X[:i1])
    m = LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced").fit(sc.transform(X[:i1]), y[:i1])
    return roc_auc_score(y[i2:], m.predict_proba(sc.transform(X[i2:]))[:, 1])
def auc_tp(X, y):
    if not HAVE_TP: return np.nan
    m = TabPFNClassifier(random_state=0).fit(X[:i1], y[:i1])
    return roc_auc_score(y[i2:], m.predict_proba(X[i2:])[:, 1])
print(f"\n[{REGION}] 일별 n={n}, AR임계(85th)={THR:.0f}")
print(f"{'임계':>5} {'양성%':>5} | {'(a)LR':>6} {'(c)LR':>6} {'ΔLR':>6} | {'(a)Tab':>7} {'(c)Tab':>7} {'ΔTab':>6} | {'test양성':>6}")
for thr in [2, 3, 4, 5]:
    y = (durs >= thr).astype(int)
    if len(np.unique(y[i2:])) < 2: print(f"{thr:>4}일 클래스부족"); continue
    aL, cL = auc_lr(Fa, y), auc_lr(Fc, y)
    aT, cT = auc_tp(Fa, y), auc_tp(Fc, y)
    print(f"{thr:>4}일 {y.mean()*100:>4.0f}% | {aL:>6.3f} {cL:>6.3f} {cL-aL:>+6.3f} | {aT:>7.3f} {cT:>7.3f} {cT-aT:>+6.3f} | {int(y[i2:].sum()):>6}")
