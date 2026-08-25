# ===== Colab: 수문학적 검증 (심사자1 3·5번) =====
# 원예보 vs TabPFN 의 판정 이동을 강수·유량으로 평가한다.
#   회수(gain)  = 원예보 미판정, 모델 판정, 실제 지속  -> 새로 경고하게 된 사건
#   손실(loss)  = 원예보 판정,   모델 미판정, 실제 지속 -> 잃은 사건
#   오경보(FP)  = 실제 미지속인데 판정               -> 늘었는지 확인
# 자료: data/hydro/precip_hourly_{ca,chile}.npz (ERA5 기반 시간별 강수, IVT 동일 격자점)
#       data/hydro/q_*.txt (USGS 일유량, 캘리포니아 격자점 40km 내 결측 0인 4개 유역)
# 준비: colab_dump_pred.py 를 먼저 실행해 pred_dump_{R}_24h.npz 생성.
import numpy as np
from scipy import stats

GAUGE = {"11181040": ("San Lorenzo C", 32.0), "11460400": ("Lagunitas C", 37.1),
         "11162630": ("Pilarcitos C", 32.0), "11181000": ("San Lorenzo C (Hayward)", 39.0)}
IVTF = {"ca": "sf", "chile": "chile"}


def daily_precip(R):
    z = np.load(f"data/hydro/precip_hourly_{R}.npz", allow_pickle=True)
    t = z["times"].astype(str); p = z["precipitation"].astype("f8")
    idx = {s: i for i, s in enumerate(t)}
    return idx, p


def load_q(sid):
    D = {}
    for L in open(f"data/hydro/q_{sid}.txt", encoding="utf-8"):
        if L.startswith("#"): continue
        f = L.rstrip("\n").split("\t")
        if len(f) < 4 or f[0] != "USGS": continue
        try: D[np.datetime64(f[2])] = float(f[3])
        except: pass
    return D


def desc(x, unit=""):
    if len(x) == 0: return "n=0"
    return f"n={len(x):3d} 중앙 {np.median(x):7.2f} 평균 {np.mean(x):7.2f}{unit}"


for R in ["ca", "chile"]:
    z = np.load(f"pred_dump_{R}_24h.npz")
    y = z["y"].astype(bool); raw = z["raw"].astype(bool); tab = z["tabpfn"].astype(bool)
    oday = z["oday"]
    # IVT 계열은 1980-01-01 00Z 시작 6시간 간격 -> oday 일차 = 1980-01-01 + oday일 (로컬 대조 확인함)
    dates = np.datetime64("1980-01-01") + oday.astype("timedelta64[D]")

    gain = (~raw & tab & y); loss = (raw & ~tab & y)
    both = (raw & tab & y); fp_raw = (raw & ~y); fp_tab = (tab & ~y)

    print("=" * 78)
    print(f"[{R}] n={len(y)}  실제 지속 {int(y.sum())}건")
    print(f"  회수 {int(gain.sum())}건 / 손실 {int(loss.sum())}건 "
          f"= 순증 {int(gain.sum())-int(loss.sum())}건")
    print(f"  오경보 원예보 {int(fp_raw.sum())} -> 모델 {int(fp_tab.sum())} "
          f"({int(fp_tab.sum())-int(fp_raw.sum()):+d})")

    # ---- 강수 (온셋일 기준 24h)
    pidx, pp = daily_precip(R)
    P24 = np.array([pp[pidx[str(d)+"T00:00"]+1: pidx[str(d)+"T00:00"]+25].sum()
                    if str(d)+"T00:00" in pidx else np.nan for d in dates])
    print("  --- 온셋일 강수 (mm) ---")
    for nm, m in [("둘 다 판정", both), ("회수(gain)", gain), ("손실(loss)", loss)]:
        print(f"    {nm:14s} {desc(P24[m & ~np.isnan(P24)], ' mm')}")

    # ---- 유량 (캘리포니아만)
    if R != "ca":
        print("  (칠레: 공개 관측 유량이 2018-03 종료·반건조 소하천이라 미수행)\n")
        continue
    print("  --- 첨두유량 (cfs, 온셋일~+2일 최대) ---")
    for sid, (nm, dist) in GAUGE.items():
        D = load_q(sid)
        def pk(d):
            v = [D.get(d + np.timedelta64(k, "D")) for k in range(3)]
            v = [x for x in v if x is not None]
            return max(v) if v else np.nan
        Q = np.array([pk(d) for d in dates]); ok = ~np.isnan(Q)
        g = Q[gain & ok]; l = Q[loss & ok]; b = Q[both & ok]
        pv = stats.mannwhitneyu(g, l, alternative="two-sided")[1] if len(g) and len(l) else np.nan
        print(f"    {nm:22s}({dist:4.1f}km)  둘다 {np.median(b):7.1f} | "
              f"회수 {np.median(g):7.1f} (n={len(g)}) | 손실 {np.median(l):7.1f} (n={len(l)}) | "
              f"회수vs손실 p={pv:.3f}")
    print()
