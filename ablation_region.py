"""지역 일반화 기간 ablation: 변수별 wavelet 창 길이 sweep -> test AUC (로지스틱).
AR 임계값=지역 85th percentile. 공통 이벤트셋(o>=127, IVT 64일 가능), 레이블 dur>=3 고정.
사용: python ablation_region.py [ca|uk|chile]"""
import os, sys, numpy as np, pywt, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
REGION = sys.argv[1] if len(sys.argv) > 1 else "uk"
IVT_FILE = {"ca": "ivt_sf_1980_2023.npy"}.get(REGION, f"ivt_{REGION}_1980_2023.npy")
CIRC_FILE = {"ca": "circ_indices.npz"}.get(REGION, f"circ_indices_{REGION}.npz")
RAW = "D:/projects/qae-parisian-climate/data/raw"
ivt = np.load(os.path.join(RAW, IVT_FILE)).astype("float64")
ci = np.load(os.path.join(RAW, CIRC_FILE)); jet = ci["jet"].astype("float64"); blk = ci["blocking"].astype("float64")
dmax = ivt.reshape(-1, 4).max(1); ND = len(dmax)
THR = np.percentile(dmax, 85); ar = dmax > THR
def lvl(n): return int(np.log2(n)) - 1
def wl(a, n): return [c[-1] for c in pywt.swt(a, 'db2', level=lvl(n), trim_approx=True, norm=True)]
events = []; i = 0
while i < ND:
    if ar[i]:
        j = i
        while j < ND and ar[j]: j += 1
        o = i; dur = j - i
        if o >= 127 and (o + 1) * 4 - 256 >= 0 and not (np.isnan(jet[o]) or np.isnan(blk[o])):
            events.append((o, dur))
        i = j
    else: i += 1
def build(ivt_d, u_d, z_d):
    iv_s = ivt_d * 4; X = []; y = []
    for o, dur in events:
        e = (o + 1) * 4
        f = wl(ivt[e - iv_s:e], iv_s) + [dmax[o], jet[o]] + wl(jet[o - u_d + 1:o + 1], u_d)
        if z_d: f += wl(blk[o - z_d + 1:o + 1], z_d)
        X.append(f); y.append(int(dur >= 3))
    return np.array(X), np.array(y)
def auc_of(X, y):
    n = len(y); i1 = int(n * 0.6); i2 = int(n * 0.8)
    sc = StandardScaler().fit(X[:i1]); m = LogisticRegression(max_iter=3000, C=0.5).fit(sc.transform(X[:i1]), y[:i1])
    return roc_auc_score(y[i2:], m.predict_proba(sc.transform(X[i2:]))[:, 1])
print(f"[{REGION}] AR임계(85th)={THR:.0f}  공통이벤트 n={len(events)}, >=3day={sum(d >= 3 for _, d in events)}\n")
print("[IVT 창 sweep]  (U250=64d, Z500=64d 고정, 레이블 dur>=3)")
for iv in [8, 16, 32, 64]:
    X, y = build(iv, 64, 64); print(f"  IVT={iv:>3}일  test AUC={auc_of(X, y):.3f}")
print("[U250 창 sweep] (IVT=16d, Z500=64d 고정)")
for u in [32, 64, 128]:
    X, y = build(16, u, 64); print(f"  U250={u:>3}일  test AUC={auc_of(X, y):.3f}")
print("[Z500 창 sweep] (IVT=16d, U250=64d 고정)")
for z in [None, 32, 64, 128]:
    X, y = build(16, 64, z); print(f"  Z500={str(z):>4}일 test AUC={auc_of(X, y):.3f}")
def build2(use_u, use_z, iv=16):
    iv_s = iv * 4; X = []; y = []
    for o, dur in events:
        e = (o + 1) * 4; f = wl(ivt[e - iv_s:e], iv_s) + [dmax[o]]
        if use_u: f += [jet[o]] + wl(jet[o - 63:o + 1], 64)
        if use_z: f += wl(blk[o - 63:o + 1], 64)
        X.append(f); y.append(int(dur >= 3))
    return np.array(X), np.array(y)
print("[변수 drop ablation] (IVT=16d)")
X, y = build2(True, True); print(f"  full            test AUC={auc_of(X, y):.3f}")
X, y = build2(True, False); print(f"  no-Z500         test AUC={auc_of(X, y):.3f}")
X, y = build2(False, False); print(f"  IVT-only        test AUC={auc_of(X, y):.3f}")
