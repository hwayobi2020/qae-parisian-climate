"""헛부름(false alarm) 후보 날짜 추출: 관측 IVT 일최대가 [0.5THR, THR)인 날(=near-miss, 관측 AR 아님).
예보 기준 분모 재설계용 — 이 날짜의 D-2 c00를 받아 예보가 THR 넘는지(=예보온셋=헛부름) 확인.
저장 nearmiss_{region}.npz {s0(일 00Z의 6h인덱스), init_str(D-2 발표 YYYYMMDD00), day_str}.  사용: python export_nearmiss.py
"""
import numpy as np, os
FILES = {"ca": ("ivt_sf_1980_2023.npy", "times_sf_1980_2023.npy"),
         "uk": ("ivt_uk_1980_2023.npy", "times_uk_1980_2023.npy"),
         "chile": ("ivt_chile_1980_2023.npy", "times_chile_1980_2023.npy")}
BAND = 0.5


def find(n):
    for p in ["data/raw/" + n, n]:
        if os.path.exists(p): return p
    raise FileNotFoundError(n)


for R, (fi, ft) in FILES.items():
    ivt = np.load(find(fi)).astype("float64"); times = np.load(find(ft))
    THR = np.percentile(ivt.reshape(-1, 4).max(1), 85)
    dmax = ivt.reshape(-1, 4).max(1)                    # 일 최대 IVT
    day0 = times.reshape(-1, 4)[:, 0]                   # 각 날의 00Z 시각
    yr = day0.astype("datetime64[Y]").astype(int) + 1970
    nd = len(dmax)
    s0 = []; inits = []; days = []
    for d in range(nd):
        if not (2000 <= yr[d] <= 2019): continue
        if not (THR * BAND <= dmax[d] < THR): continue  # near-miss (관측 AR 아님)
        dd = day0[d].astype("datetime64[D]")
        init = dd - np.timedelta64(2, "D")              # D-2 발표
        if init.astype("datetime64[Y]").astype(int) + 1970 < 2000: continue
        s0.append(d * 4); inits.append(str(init).replace("-", "") + "00"); days.append(str(dd))
    np.savez(f"nearmiss_{R}.npz", s0=np.array(s0), init_str=np.array(inits), day_str=np.array(days), THR=THR, band=BAND)
    print(f"{R}: THR={THR:.0f} | near-miss 후보(0.5THR~THR, 관측AR아님) = {len(s0)}일 -> nearmiss_{R}.npz")
