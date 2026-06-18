"""6시간 정의에서 '자명함(강도→지속)' 검증.
(a) onset IVT 강도 단독  (b) IVT 16일 wav만  (c) 전체 20피처  비교, Logistic test AUC.
(a)가 이미 높으면 = 동어반복(강하면 오래간다). (c)-(a)가 진짜 onset 환경 추가정보.
사용: python baseline_6h_region.py [ca|uk|chile]"""
import os, sys, numpy as np, pywt, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
RAW = "D:/projects/qae-parisian-climate/data/raw"
REGION = sys.argv[1] if len(sys.argv) > 1 else "ca"
IVT_FILE = {"ca": "ivt_sf_1980_2023.npy"}.get(REGION, f"ivt_{REGION}_1980_2023.npy")
CIRC_FILE = {"ca": "circ_indices.npz"}.get(REGION, f"circ_indices_{REGION}.npz")
ivt = np.load(os.path.join(RAW, IVT_FILE)).astype("float64")
ci = np.load(os.path.join(RAW, CIRC_FILE)); jet = ci["jet"].astype("float64"); blk = ci["blocking"].astype("float64")
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
Fa = []; Fb = []; Fc = []; steps = []; onsetIVT = []
for s, e in runs:
    o = s // 4
    if s - 63 >= 0 and o - 63 >= 0 and o < ND and not (np.isnan(jet[o]) or np.isnan(blk[o])):
        ivw = wl(ivt[s - 63:s + 1], 5)
        Fa.append([dmax[o]])                                   # (a) onset 강도 단독
        Fb.append(ivw + [dmax[o]])                             # (b) IVT 파형+강도
        Fc.append(ivw + [dmax[o], jet[o]] + wl(jet[o - 63:o + 1], 5) + wl(blk[o - 63:o + 1], 5))  # (c) 전체
        steps.append(e - s); onsetIVT.append(ivt[s])
Fa = np.array(Fa); Fb = np.array(Fb); Fc = np.array(Fc); steps = np.array(steps); onsetIVT = np.array(onsetIVT)
n = len(steps); i1 = int(n * 0.6); i2 = int(n * 0.8)
def auc(X, y):
    sc = StandardScaler().fit(X[:i1])
    m = LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced").fit(sc.transform(X[:i1]), y[:i1])
    return roc_auc_score(y[i2:], m.predict_proba(sc.transform(X[i2:]))[:, 1])
# onset IVT(연속run 시작값)와 지속의 상관 = 자명함 정도
r = np.corrcoef(onsetIVT, steps)[0, 1]
print(f"[{REGION}] n={n}  corr(onset IVT, 지속스텝)={r:+.2f}  (높을수록 강도→지속 자명)\n")
print(f"{'지속임계':>8} {'양성%':>6} {'(a)강도단독':>11} {'(b)IVT파형':>10} {'(c)전체':>9} {'(c)-(a)':>8}")
for k in [2, 3, 4, 6, 8]:
    y = (steps >= k).astype(int)
    if len(np.unique(y[i2:])) < 2: print(f"{k*6:>6}h  클래스부족"); continue
    aa, ab, ac = auc(Fa, y), auc(Fb, y), auc(Fc, y)
    print(f"{k*6:>6}h  {y.mean()*100:>5.0f}%   {aa:>9.3f}   {ab:>8.3f}  {ac:>7.3f}  {ac-aa:>+7.3f}")
