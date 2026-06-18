"""6시간 정의, 강도 위 기여(Δwav/Δcirc/Δfull)의 permutation test (TabPFN).
귀무가설: 추가 피처(wavelet 또는 순환)가 지속과 무관.
-> 그 피처 컬럼만 이벤트 간 재배치(시간순서/강도/레이블/train-test분할은 보존)해 재학습,
   Δ_perm 분포를 만들고 관측 Δ가 상위인지(p) 본다.
사용: python permutation_6h_region.py [ca|uk|chile|nz]   (TABPFN_TOKEN 필요)"""
import torch
import os, sys, numpy as np, pywt, warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
REGION = sys.argv[1] if len(sys.argv) > 1 else "ca"
B = int(sys.argv[2]) if len(sys.argv) > 2 else 200
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
# Fd 컬럼 배치: ivw(0-5) | dmax(6) | jet(7) | jw(8-13) | bw(14-19)
Fa = []; Fd = []; steps = []
for s, e in runs:
    o = s // 4
    if s - 63 >= 0 and o - 63 >= 0 and o < ND and not (np.isnan(jet[o]) or np.isnan(blk[o])):
        ivw = wl(ivt[s - 63:s + 1], 5); jw = wl(jet[o - 63:o + 1], 5); bw = wl(blk[o - 63:o + 1], 5)
        Fa.append([dmax[o]]); Fd.append(ivw + [dmax[o], jet[o]] + jw + bw); steps.append(e - s)
Fa = np.array(Fa); Fd = np.array(Fd); steps = np.array(steps)
n = len(steps); i1 = int(n * 0.6); i2 = int(n * 0.8)
WAV = list(range(0, 6)); CIRC = [7] + list(range(8, 20)); FULL = WAV + CIRC   # 강도(6) 제외 전부
from tabpfn import TabPFNClassifier
def auc_of(X, y):
    m = TabPFNClassifier(random_state=0, ignore_pretraining_limits=True).fit(X[:i1], y[:i1])
    return roc_auc_score(y[i2:], m.predict_proba(X[i2:])[:, 1])
rng = np.random.RandomState(0)
print(f"\n[{REGION}] 6h n={n}, AR임계(85th)={THR:.0f}, TabPFN permutation B={B}")
for k in [4, 6, 8]:    # 24h,36h,48h
    y = (steps >= k).astype(int); yte = y[i2:]
    if len(np.unique(yte)) < 2: print(f"  {k*6}h 클래스부족"); continue
    a = auc_of(Fa, y); d_full = auc_of(Fd, y)
    print(f"\n  [{k*6}h] test양성={int(yte.sum())}/{len(yte)}  (a)강도={a:.3f}  (d)전체={d_full:.3f}  Δfull관측={d_full-a:+.3f}")
    for nm, cols in [("Δwav ", WAV), ("Δcirc", CIRC), ("Δfull", FULL)]:
        keep = sorted(set(cols + [6]))          # 해당 블록 + 강도(6)
        d_obs = auc_of(Fd[:, keep], y) - a       # 관측: 강도 위에 이 블록을 더한 기여
        perm = []
        for _ in range(B):
            p = rng.permutation(n)               # 이 블록 컬럼만 이벤트 간 재배치
            Xp = Fd.copy(); Xp[:, cols] = Fd[p][:, cols]
            perm.append(auc_of(Xp[:, keep], y) - a)
        perm = np.array(perm); pval = (perm >= d_obs).mean()
        flag = "*유의*" if pval <= 0.05 else "효과없음"
        print(f"    {nm}: 관측Δ={d_obs:+.3f}  귀무중앙{np.median(perm):+.3f} 95%[{np.percentile(perm,2.5):+.3f},{np.percentile(perm,97.5):+.3f}]  p={pval:.3f} {flag}")
