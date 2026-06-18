"""지역별 SST 지수가 AR 지속(>=3일) 예측에 기여하는지 테스트.
기존 20피처(IVT16 wav + onset + U250/Z500) vs +SST 지수, 시간순 Logistic test AUC.
ersst_v5.nc(전구 월 SST)에서 영역평균 month-anomaly를 onset 월에 매핑.
사용: python sst_test_region.py [ca|uk|chile]"""
import os, sys, numpy as np, pywt, xarray as xr, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
RAW = "D:/projects/qae-parisian-climate/data/raw"
REGION = sys.argv[1] if len(sys.argv) > 1 else "uk"
IVT_FILE = {"ca": "ivt_sf_1980_2023.npy"}.get(REGION, f"ivt_{REGION}_1980_2023.npy")
CIRC_FILE = {"ca": "circ_indices.npz"}.get(REGION, f"circ_indices_{REGION}.npz")
TIMES_FILE = {"ca": "times_sf_1980_2023.npy"}.get(REGION, f"times_{REGION}_1980_2023.npy")
# 지역별 후보 SST 영역 (lat0,lat1,lon0,lon1) — lon 0-360
SST_BOXES = {
    "ca":    {"local_NEpac": (34, 40, 225, 235), "nino34": (-5, 5, 190, 240)},
    "uk":    {"N_Atlantic": (45, 55, 310, 350), "subpolar": (50, 60, 320, 350)},
    "chile": {"nino34": (-5, 5, 190, 240), "local_SEpac": (-36, -30, 280, 288)},
}
ivt = np.load(os.path.join(RAW, IVT_FILE)).astype("float64")
ci = np.load(os.path.join(RAW, CIRC_FILE)); jet = ci["jet"].astype("float64"); blk = ci["blocking"].astype("float64")
ti = np.load(os.path.join(RAW, TIMES_FILE))[::4].astype("datetime64[D]")
dmax = ivt.reshape(-1, 4).max(1); ND = len(dmax); THR = np.percentile(dmax, 85); ar = dmax > THR

ds = xr.open_dataset(os.path.join(RAW, "ersst_v5.nc")); SST = ds["sst"]
def sst_index(box):
    la0, la1, lo0, lo1 = box
    msk = (SST.lat >= la0) & (SST.lat <= la1) & (SST.lon >= lo0) & (SST.lon <= lo1)
    sm = SST.where(msk).mean(["lat", "lon"])
    clim = sm.groupby("time.month").mean("time")
    anom = (sm.groupby("time.month") - clim)
    at = anom["time"].values; av = anom.values
    k2v = {(int(str(t)[:4]), int(str(t)[5:7])): float(v) for t, v in zip(at, av)}
    return np.array([k2v.get((int(str(d)[:4]), int(str(d)[5:7])), np.nan) for d in ti])
sst_feats = {name: sst_index(box) for name, box in SST_BOXES[REGION].items()}

def wl(a, lvl): return [c[-1] for c in pywt.swt(a, 'db2', level=lvl, trim_approx=True, norm=True)]
events = []; i = 0
while i < ND:
    if ar[i]:
        j = i
        while j < ND and ar[j]: j += 1
        o = i; dur = j - i; end6 = (o + 1) * 4
        if end6 - 64 >= 0 and o - 63 >= 0 and not (np.isnan(jet[o]) or np.isnan(blk[o])):
            events.append((o, dur, end6))
        i = j
    else: i += 1
def base_feat(o, end6):
    return wl(ivt[end6 - 64:end6], 5) + [dmax[o], jet[o]] + wl(jet[o - 63:o + 1], 5) + wl(blk[o - 63:o + 1], 5)
def run(extra_names):
    X = []; y = []
    for o, dur, end6 in events:
        f = base_feat(o, end6) + [sst_feats[nm][o] for nm in extra_names]
        if any(np.isnan(v) for v in f): continue
        X.append(f); y.append(int(dur >= 3))
    X = np.array(X); y = np.array(y); n = len(y); i1 = int(n * 0.6); i2 = int(n * 0.8)
    sc = StandardScaler().fit(X[:i1]); m = LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced").fit(sc.transform(X[:i1]), y[:i1])
    return roc_auc_score(y[i2:], m.predict_proba(sc.transform(X[i2:]))[:, 1]), n
print(f"[{REGION}] AR임계(85th)={THR:.0f}, 이벤트 n={len(events)}, 레이블 dur>=3, SST 후보={list(SST_BOXES[REGION])}\n")
a0, n0 = run([]); print(f"  기존 20피처 (SST 없음)      test AUC={a0:.3f}  (n={n0})")
for nm in SST_BOXES[REGION]:
    a1, n1 = run([nm]); print(f"  + SST[{nm:14s}]        test AUC={a1:.3f}  (Δ{a1-a0:+.3f})")
alln = list(SST_BOXES[REGION])
aA, nA = run(alln); print(f"  + SST[전부 {len(alln)}개]            test AUC={aA:.3f}  (Δ{aA-a0:+.3f})")
