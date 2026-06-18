"""6시간 해상도 지속기간 임계값 촘촘 스윕 (12h,18h,24h,...).
AR 이벤트 = 6시간 IVT가 지역 85th를 연속 초과하는 구간. onset=시작 6h 스텝.
지속 = (e-s) 스텝 * 6h. 임계를 6시간씩 늘려가며 Logistic/RF test AUC.
사용: python threshold_6h_region.py [ca|uk|chile]"""
import os, sys, numpy as np, pywt, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
RAW = "D:/projects/qae-parisian-climate/data/raw"
REGION = sys.argv[1] if len(sys.argv) > 1 else "ca"
IVT_FILE = {"ca": "ivt_sf_1980_2023.npy"}.get(REGION, f"ivt_{REGION}_1980_2023.npy")
CIRC_FILE = {"ca": "circ_indices.npz"}.get(REGION, f"circ_indices_{REGION}.npz")
ivt = np.load(os.path.join(RAW, IVT_FILE)).astype("float64")
ci = np.load(os.path.join(RAW, CIRC_FILE)); jet = ci["jet"].astype("float64"); blk = ci["blocking"].astype("float64")
dmax = ivt.reshape(-1, 4).max(1); ND = len(dmax)
THR = np.percentile(dmax, 85)                       # 85th은 일별최대 기준 유지
T = len(ivt); ar6 = ivt > THR
def wl(a, lvl): return [c[-1] for c in pywt.swt(a, 'db2', level=lvl, trim_approx=True, norm=True)]
# 6시간 연속 run
runs = []; i = 0
while i < T:
    if ar6[i]:
        j = i
        while j < T and ar6[j]: j += 1
        runs.append((i, j)); i = j
    else: i += 1
feats = []; steps = []
for s, e in runs:
    o = s // 4
    if s - 63 >= 0 and o - 63 >= 0 and o < ND and not (np.isnan(jet[o]) or np.isnan(blk[o])):
        feats.append(wl(ivt[s - 63:s + 1], 5) + [dmax[o], jet[o]] + wl(jet[o - 63:o + 1], 5) + wl(blk[o - 63:o + 1], 5))
        steps.append(e - s)                          # 지속 스텝수 (×6h)
X = np.array(feats); steps = np.array(steps); n = len(steps); i1 = int(n * 0.6); i2 = int(n * 0.8)
med_h = np.median(steps) * 6
print(f"[{REGION}] 85th임계={THR:.0f}, 6h run 이벤트 n={n}, 지속중앙값={med_h:.0f}h, 최대={steps.max()*6}h")
print(f"  train={i1} test={n-i2}\n")
print(f"{'지속임계':>8} {'양성%':>6} {'Logistic':>9} {'RF(5시드)':>15} {'test양성':>8}")
for k in [2, 3, 4, 5, 6, 7, 8, 10, 12]:                # 스텝수 → 12h,18h,...,72h
    y = (steps >= k).astype(int)
    sc = StandardScaler().fit(X[:i1]); Xtr, Xte = sc.transform(X[:i1]), sc.transform(X[i2:]); ytr, yte = y[:i1], y[i2:]
    if len(np.unique(yte)) < 2 or len(np.unique(ytr)) < 2:
        print(f"{k*6:>6}h  {y.mean()*100:>5.0f}%  클래스부족(test양성={int(yte.sum())})"); continue
    lr = LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced").fit(Xtr, ytr)
    al = roc_auc_score(yte, lr.predict_proba(Xte)[:, 1])
    rf = np.array([roc_auc_score(yte, RandomForestClassifier(n_estimators=400, max_depth=4, min_samples_leaf=20,
                  class_weight="balanced", random_state=sd).fit(Xtr, ytr).predict_proba(Xte)[:, 1]) for sd in range(5)])
    print(f"{k*6:>6}h  {y.mean()*100:>5.0f}%   {al:>7.3f}   {rf.mean():.3f}±{rf.std():.3f} {int(yte.sum()):>8}")
