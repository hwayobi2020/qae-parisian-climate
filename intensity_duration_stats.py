"""강도(IVT)가 높으면 AR 지속이 길어지나 — 순수 기술통계 (ML 없음).

각 이벤트: onset 강도 두 정의로
  - dmax  : onset 당일 최대 IVT (관례 강도, 단 intra-day look-ahead 포함)
  - rec24 : onset 직전 24h(4×6h) 최대 IVT (엄격 causal)
지속: steps×6 시간.

출력(지역×강도정의):
  - Spearman 상관(강도, 지속)
  - 강도 5분위 구간별: n, 지속 중앙값(h), P(≥24h), P(≥48h)
사용: python intensity_duration_stats.py
"""
import os, numpy as np
from scipy.stats import spearmanr

def find(name):
    for p in [os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw", name),
              os.path.join("data/raw", name)]:
        if os.path.exists(p): return p
    raise FileNotFoundError(name)

def load_events(region):
    ivt = np.load(find({"ca": "ivt_sf_1980_2023.npy"}.get(region, f"ivt_{region}_1980_2023.npy"))).astype("float64")
    ci = np.load(find({"ca": "circ_indices.npz"}.get(region, f"circ_indices_{region}.npz")))
    jet = ci["jet"].astype("float64"); blk = ci["blocking"].astype("float64")
    dmaxd = ivt.reshape(-1, 4).max(1); ND = len(dmaxd); THR = np.percentile(dmaxd, 85)
    T = len(ivt); ar6 = ivt > THR
    runs = []; i = 0
    while i < T:
        if ar6[i]:
            j = i
            while j < T and ar6[j]: j += 1
            runs.append((i, j)); i = j
        else: i += 1
    dmax, rec24, dur = [], [], []
    for s, e in runs:
        o = s // 4
        if s - 63 >= 0 and o - 63 >= 0 and o < ND and not (np.isnan(jet[o]) or np.isnan(blk[o])):
            dmax.append(dmaxd[o]); rec24.append(ivt[s - 3:s + 1].max()); dur.append((e - s) * 6)
    return THR, np.array(dmax), np.array(rec24), np.array(dur)

def report(region):
    THR, dmax, rec24, dur = load_events(region)
    n = len(dur)
    print(f"\n=== [{region}] n={n}, AR임계(85th)={THR:.0f} kg/m/s, "
          f"지속 중앙값={np.median(dur):.0f}h, P(≥24h)={(dur>=24).mean():.2f}, P(≥48h)={(dur>=48).mean():.2f} ===")
    for name, x in [("dmax(당일최대,look-ahead)", dmax), ("rec24(직전24h,causal) ", rec24)]:
        rho, p = spearmanr(x, dur)
        print(f"  [{name}]  Spearman ρ(강도,지속)={rho:+.3f} (p={p:.1e})")
        edges = np.quantile(x, [0, .2, .4, .6, .8, 1.0]); edges[-1] += 1e-9
        b = np.digitize(x, edges[1:-1])
        print(f"     {'5분위':<6}{'IVT범위':>20}{'n':>6}{'지속중앙(h)':>11}{'P(≥24h)':>9}{'P(≥48h)':>9}")
        for q in range(5):
            m = b == q
            if m.sum() == 0: continue
            print(f"     Q{q+1:<5}{f'{x[m].min():.0f}-{x[m].max():.0f}':>20}{m.sum():>6}"
                  f"{np.median(dur[m]):>11.0f}{(dur[m]>=24).mean():>9.2f}{(dur[m]>=48).mean():>9.2f}")

for r in ["uk", "ca", "chile"]:
    report(r)
