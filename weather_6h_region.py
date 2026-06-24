"""6시간 정의, onset 시점 실제 날씨(Open-Meteo)가 강도·순환 위에 기여하는지.
날씨 = 기온·해면기압·강수·습도·풍속·풍향(sin/cos)·운량 (onset 시점 스냅샷).
(a) 강도  (d) 전체(강도+IVTwav+순환)  (d+wx) 전체+날씨.  LR + TabPFN.
사용: python weather_6h_region.py [ca|uk|chile]   (openmeteo_<region>_1980_2023.npz 필요, TABPFN_TOKEN)"""
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
wx = np.load(find(f"openmeteo_{REGION}_1980_2023.npz"), allow_pickle=True)
# 날씨 변수(인덱스가 6시간 IVT와 완전일치). 풍향은 sin/cos.
WV = ["temperature_2m", "pressure_msl", "precipitation", "relative_humidity_2m", "wind_speed_10m", "cloud_cover"]
wxarr = np.stack([wx[v].astype("float64") for v in WV], 1)         # (T6, 6)
wd = np.deg2rad(wx["wind_direction_10m"].astype("float64"))
wxarr = np.hstack([wxarr, np.stack([np.sin(wd), np.cos(wd)], 1)])  # +sin/cos = (T6, 8)
assert len(wxarr) == len(ivt), f"weather/ivt 길이 불일치 {len(wxarr)} vs {len(ivt)}"
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
Fa = []; Fd = []; Fdw = []; steps = []
for s, e in runs:
    o = s // 4
    if s - 63 >= 0 and o - 63 >= 0 and o < ND and not (np.isnan(jet[o]) or np.isnan(blk[o])):
        d = wl(ivt[s - 63:s + 1], 5) + [dmax[o], jet[o]] + wl(jet[o - 63:o + 1], 5) + wl(blk[o - 63:o + 1], 5)
        wsnap = list(wxarr[s])                       # onset 시점 날씨 스냅샷
        Fa.append([dmax[o]]); Fd.append(d); Fdw.append(d + wsnap); steps.append(e - s)
Fa, Fd, Fdw = map(np.array, (Fa, Fd, Fdw)); steps = np.array(steps)
n = len(steps); i1 = int(n * 0.6); i2 = int(n * 0.8)
from tabpfn import TabPFNClassifier
def auc_lr(X, y):
    sc = StandardScaler().fit(X[:i1])
    m = LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced").fit(sc.transform(X[:i1]), y[:i1])
    return roc_auc_score(y[i2:], m.predict_proba(sc.transform(X[i2:]))[:, 1])
def auc_tp(X, y):
    m = TabPFNClassifier(random_state=0, ignore_pretraining_limits=True).fit(X[:i1], y[:i1])
    return roc_auc_score(y[i2:], m.predict_proba(X[i2:])[:, 1])
print(f"\n[{REGION}] 6h n={n}, AR임계(85th)={THR:.0f}  (onset 날씨 8피처: {WV}+풍향sin/cos)")
for name, auc in [("Logistic", auc_lr), ("TabPFN", auc_tp)]:
    print(f"\n  === {name} ===")
    print(f"  {'지속':>5} {'양성%':>5} | {'(a)강도':>7} {'(d)전체':>7} {'(d)+날씨':>8} {'Δ날씨':>7} | {'test양성':>6}")
    for k in [4, 6, 8]:    # 24h,36h,48h
        y = (steps >= k).astype(int)
        if len(np.unique(y[i2:])) < 2: print(f"  {k*6}h 클래스부족"); continue
        a, d, dw = auc(Fa, y), auc(Fd, y), auc(Fdw, y)
        print(f"  {k*6:>4}h {y.mean()*100:>4.0f}% | {a:>7.3f} {d:>7.3f} {dw:>8.3f} {dw-d:>+7.3f} | {int(y[i2:].sum()):>6}")
