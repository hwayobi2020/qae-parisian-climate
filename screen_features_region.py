"""6h 정의, AR 지속기간 예측 후보변수 기초통계 선별(screening).

목적: 모델(cv_causal) 투입 전, 지속기간과 단변량 연관이 있는 causal 후보를 거른다.
원칙:
  1) 모든 후보는 LEAK-FREE (IVT계열 ≤ s, 순환/모드 ≤ o-1, 측정규약상 강수도 s는 직전1h누적이라 causal).
  2) 선별 통계는 TRAIN 구간(시간순 앞 60% 이벤트)에서만 → CV test 오염 방지(이중 사용 금지).
  3) 단변량 AUC(방향성) + 상호정보량(MI, 비선형) + 점이연상관(point-biserial) 3종 동시.
  4) 판정 기준 = 지역 간(특히 CA·Chile) 일관성. 단일 p값 신뢰 금지.

신규 후보(기존 cv_causal 미사용): 제트 위도(jet latitude), IVT 방향, 다중스케일 IVT 최대/추세,
  직전 AR활동, 기후모드(ENSO/MJO/QBO/SST), 계절성, 지표변수(기압/기온/습도/바람/구름/강수).

사용: python screen_features_region.py [ca|uk|chile|all]
출력: 지역×수평선(24/36/48h)별 후보 랭킹표 + 지역간 36h 일관성 요약.
"""
import os, sys, numpy as np, warnings
warnings.filterwarnings("ignore")
from scipy.stats import pointbiserialr
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw")
PTS = {"ca": (37.77, -122.42), "uk": (50.0, -5.0), "chile": (-33.0, -71.5)}
IVT_NPY = {"ca": "ivt_sf_1980_2023.npy"}
CIRC = {"ca": "circ_indices.npz"}
TIMES = {"ca": "times_sf_1980_2023.npy"}
GRID_NC = {"ca": "npac_z500_u250_1980_2023.nc",
           "uk": "natl_z500_u250_1980_2023_uk.nc",
           "chile": "sepac_z500_u250_1980_2023_chile.nc"}
HORIZONS = [4, 6, 8]
TRAIN_FRAC = 0.6


def p(name, region):
    return os.path.join(RAW, IVT_NPY.get(region, name) if name.startswith("ivt_") else
                        CIRC.get(region, name) if name.startswith("circ") else
                        TIMES.get(region, name) if name.startswith("times") else name)


def load_region_core(region):
    ivt = np.load(os.path.join(RAW, IVT_NPY.get(region, f"ivt_{region}_1980_2023.npy"))).astype("float64")
    times = np.load(os.path.join(RAW, TIMES.get(region, f"times_{region}_1980_2023.npy")))
    ci = np.load(os.path.join(RAW, CIRC.get(region, f"circ_indices_{region}.npz")), allow_pickle=True)
    jet = ci["jet"].astype("float64"); blk = ci["blocking"].astype("float64")
    cdates = ci["dates"].astype("datetime64[D]")
    return ivt, times, jet, blk, cdates


def load_aux_6h(region, T):
    """6h 정렬 보조변수 (length T 가정, 아니면 날짜 dict 정렬)."""
    out = {}
    om = np.load(os.path.join(RAW, f"openmeteo_{region}_1980_2023.npz"), allow_pickle=True)
    for v in ["temperature_2m", "pressure_msl", "precipitation",
              "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m", "cloud_cover"]:
        a = om[v].astype("float64")
        out[v] = a if len(a) == T else np.full(T, np.nan)
    # ENSO/SST: CA 기준 64284 6h. 타지역도 같은 전구 모드값이라 동일 배열 재사용(시간정렬 동일).
    for f, key in [("enso_anom.npy", "enso"), ("sst_anom.npy", "sst")]:
        a = np.load(os.path.join(RAW, f)).astype("float64")
        out[key] = a if len(a) == T else np.full(T, np.nan)
    return out


def load_jet_lat(region, cdates):
    """격자 U250에서 일별 제트코어 위도(area-max 위치의 위도) → circ 날짜(o)에 정렬."""
    import xarray as xr
    ds = xr.open_dataset(os.path.join(RAW, GRID_NC[region]))
    u = ds["u250"]
    latname = "lat" if "lat" in u.dims else "latitude"
    lonname = "lon" if "lon" in u.dims else "longitude"
    lats = ds[latname].values
    U = u.values  # (time, lat, lon)
    Tn, nlat, nlon = U.shape
    flat = U.reshape(Tn, nlat * nlon)
    am = np.nanargmax(flat, axis=1)
    jlat = lats[am // nlon]
    gdates = ds["time"].values.astype("datetime64[D]")
    ds.close()
    d2l = {d: v for d, v in zip(gdates, jlat)}
    return np.array([d2l.get(d, np.nan) for d in cdates], dtype="float64")


def load_ivt_dir(region, times):
    """IVT 방향(동/북 성분 → 방위각)을 onset 격자셀 최근접에서 추출, times에 정렬."""
    import xarray as xr, glob
    la, lo = PTS[region]
    files = sorted(glob.glob(os.path.join(RAW, f"era5_ivt_{region}_*_*.nc")))
    if not files:
        return None, None
    e_all = {}; n_all = {}
    for f in files:
        ds = xr.open_dataset(f)
        latn = "latitude" if "latitude" in ds.coords else "lat"
        lonn = "longitude" if "longitude" in ds.coords else "lon"
        sub = ds.sel({latn: la, lonn: lo}, method="nearest")
        tn = "valid_time" if "valid_time" in ds.coords else "time"
        td = ds[tn].values.astype("datetime64[h]")
        ve = sub["viwve"].values.ravel(); vn = sub["viwvn"].values.ravel()
        for d, a, b in zip(td, ve, vn):
            e_all[d] = a; n_all[d] = b
        ds.close()
    th = times.astype("datetime64[h]")
    ve = np.array([e_all.get(d, np.nan) for d in th], dtype="float64")
    vn = np.array([n_all.get(d, np.nan) for d in th], dtype="float64")
    ang = np.arctan2(vn, ve)
    return np.sin(ang), np.cos(ang)


def load_mjo(cdates):
    """MJO RMM1, RMM2, amplitude → o 날짜 정렬(직전일 o-1 사용은 빌드 단계에서)."""
    path = os.path.join(RAW, "mjo_rmm_new.txt")
    d2 = {}
    with open(path) as fh:
        for ln in fh:
            t = ln.split()
            if len(t) < 7 or not t[0].isdigit() or len(t[0]) != 4:
                continue
            try:
                y, mo, da = int(t[0]), int(t[1]), int(t[2])
                r1, r2, amp = float(t[3]), float(t[4]), float(t[6])
            except Exception:
                continue
            if abs(r1) > 100 or abs(r2) > 100:  # missing 1e36/999
                continue
            d2[np.datetime64(f"{y:04d}-{mo:02d}-{da:02d}")] = (r1, r2, amp)
    r1 = np.array([d2.get(d, (np.nan,)*3)[0] for d in cdates])
    r2 = np.array([d2.get(d, (np.nan,)*3)[1] for d in cdates])
    amp = np.array([d2.get(d, (np.nan,)*3)[2] for d in cdates])
    return r1, r2, amp


def load_qbo(cdates):
    """QBO 30mb 월별 → o 날짜(해당 월) 정렬."""
    path = os.path.join(RAW, "qbo_noaa30.txt")
    ym = {}
    with open(path) as fh:
        for ln in fh:
            t = ln.split()
            if len(t) >= 13 and t[0].isdigit() and len(t[0]) == 4:
                y = int(t[0])
                for mo in range(1, 13):
                    try:
                        ym[(y, mo)] = float(t[mo])
                    except Exception:
                        pass
    months = cdates.astype("datetime64[M]")
    yv = months.astype("datetime64[Y]").astype(int) + 1970
    mv = months.astype(int) % 12 + 1
    return np.array([ym.get((int(a), int(b)), np.nan) for a, b in zip(yv, mv)], dtype="float64")


def build_features(region):
    ivt, times, jet, blk, cdates = load_region_core(region)
    T = len(ivt); ND = len(jet)
    THR = np.percentile(ivt.reshape(-1, 4).max(1), 85)
    aux = load_aux_6h(region, T)
    jet_lat = load_jet_lat(region, cdates)
    dir_sin, dir_cos = load_ivt_dir(region, times)
    mjo1, mjo2, mjoA = load_mjo(cdates)
    qbo = load_qbo(cdates)

    ar6 = ivt > THR
    runs = []; i = 0
    while i < T:
        if ar6[i]:
            j = i
            while j < T and ar6[j]:
                j += 1
            runs.append((i, j)); i = j
        else:
            i += 1

    doy = times.astype("datetime64[D]").astype("datetime64[Y]")
    # day-of-year
    doy_num = ((times.astype("datetime64[D]") - times.astype("datetime64[D]").astype("datetime64[Y]"))
               / np.timedelta64(1, "D")).astype("float64")
    month = times.astype("datetime64[M]").astype(int) % 12 + 1

    feats = {}
    def add(name, val):
        feats.setdefault(name, []).append(val)

    steps = []
    NEW = set()  # 신규(기존 cv_causal 미사용) 표시
    new_names = {"ivt_max_48h","ivt_max_72h","ivt_max_7d","ivt_mean_24h","ivt_mean_7d",
                 "ivt_trend_24h","ivt_trend_48h","ivt_std_7d","ar_frac_30d","ar_frac_60d",
                 "ivt_dir_sin","ivt_dir_cos","jet_lat_o1","jet_lat_trend","blk_trend","jet_trend",
                 "enso","sst","mjo_rmm1","mjo_rmm2","mjo_amp","qbo","doy_sin","doy_cos",
                 "pressure_msl","pressure_trend","temperature_2m","relative_humidity_2m",
                 "wind_speed_10m","wind_dir_sin","wind_dir_cos","cloud_cover","precip","precip_24h"}

    for s, e in runs:
        o = s // 4
        if not (s - 63 >= 0 and o - 2 >= 0 and o < ND):
            continue
        # --- IVT 동역학 (≤ s) ---
        add("ivt_onset", ivt[s])
        add("ivt_max_24h", ivt[s-3:s+1].max())
        add("ivt_max_48h", ivt[s-7:s+1].max())
        add("ivt_max_72h", ivt[s-11:s+1].max())
        add("ivt_max_7d", ivt[s-27:s+1].max())
        add("ivt_mean_24h", ivt[s-3:s+1].mean())
        add("ivt_mean_7d", ivt[s-27:s+1].mean())
        add("ivt_trend_24h", ivt[s] - ivt[s-4])
        add("ivt_trend_48h", ivt[s] - ivt[s-8])
        add("ivt_std_7d", ivt[s-27:s+1].std())
        add("ar_frac_30d", ar6[max(0, s-119):s+1].mean())
        add("ar_frac_60d", ar6[max(0, s-239):s+1].mean())
        add("ivt_dir_sin", dir_sin[s] if dir_sin is not None else np.nan)
        add("ivt_dir_cos", dir_cos[s] if dir_cos is not None else np.nan)
        # --- 순환 (≤ o-1) ---
        add("jet_o1", jet[o-1])
        add("blk_o1", blk[o-1])
        add("jet_trend", jet[o-1] - jet[o-2])
        add("blk_trend", blk[o-1] - blk[o-2])
        add("jet_lat_o1", jet_lat[o-1])
        add("jet_lat_trend", jet_lat[o-1] - jet_lat[o-2])
        # --- 기후모드 (≤ o-1; 느린 변수) ---
        add("enso", aux["enso"][s])
        add("sst", aux["sst"][s])
        add("mjo_rmm1", mjo1[o-1]); add("mjo_rmm2", mjo2[o-1]); add("mjo_amp", mjoA[o-1])
        add("qbo", qbo[o-1])
        # --- 계절성 (≤ s) ---
        add("doy_sin", np.sin(2*np.pi*doy_num[s]/365.25))
        add("doy_cos", np.cos(2*np.pi*doy_num[s]/365.25))
        # --- 지표 (≤ s; 강수는 직전1h누적이라 causal) ---
        add("pressure_msl", aux["pressure_msl"][s])
        add("pressure_trend", aux["pressure_msl"][s] - aux["pressure_msl"][s-4])
        add("temperature_2m", aux["temperature_2m"][s])
        add("relative_humidity_2m", aux["relative_humidity_2m"][s])
        add("wind_speed_10m", aux["wind_speed_10m"][s])
        wd = np.deg2rad(aux["wind_direction_10m"][s])
        add("wind_dir_sin", np.sin(wd)); add("wind_dir_cos", np.cos(wd))
        add("cloud_cover", aux["cloud_cover"][s])
        add("precip", aux["precipitation"][s])
        add("precip_24h", np.nansum(aux["precipitation"][s-3:s+1]))
        steps.append(e - s)

    X = {k: np.array(v, dtype="float64") for k, v in feats.items()}
    steps = np.array(steps)
    return X, steps, THR, new_names


def screen(region):
    X, steps, THR, new_names = build_features(region)
    n = len(steps); i60 = int(n * TRAIN_FRAC)
    names = list(X.keys())
    print(f"\n{'='*78}\n[{region}] n_events={n}, AR임계(85th)={THR:.0f}, "
          f"선별 train={i60} (앞 {int(TRAIN_FRAC*100)}%)\n{'='*78}")
    region_summary = {}
    for k in HORIZONS:
        y_full = (steps >= k).astype(int)
        y = y_full[:i60]
        if len(np.unique(y)) < 2:
            print(f"\n[{k*6}h] train 클래스부족 skip"); continue
        rows = []
        for nm in names:
            x = X[nm][:i60]
            m = ~np.isnan(x)
            if m.sum() < 30 or len(np.unique(y[m])) < 2 or np.nanstd(x[m]) == 0:
                continue
            xa, ya = x[m], y[m]
            auc = roc_auc_score(ya, xa)
            auc_dir = max(auc, 1 - auc)
            sign = "+" if auc >= 0.5 else "-"
            try:
                r, pv = pointbiserialr(ya, xa)
            except Exception:
                r, pv = np.nan, np.nan
            mi = mutual_info_classif(xa.reshape(-1, 1), ya, random_state=0,
                                     discrete_features=False)[0]
            rows.append((nm, auc_dir, sign, r, pv, mi, m.sum()))
        rows.sort(key=lambda t: -t[1])
        pos = int(y.sum())
        print(f"\n[{k*6}h] train 양성={pos}/{i60}  (단변량 AUC 내림차순 상위 18)")
        print(f"  {'feature':<20}{'AUC':>6} {'dir':>4} {'pbis_r':>8} {'p':>8} {'MI':>7}  new")
        for nm, auc, sign, r, pv, mi, nn in rows[:18]:
            star = "*" if pv < 0.05 else " "
            new = "NEW" if nm in new_names else ""
            print(f"  {nm:<20}{auc:6.3f} {sign:>4} {r:+8.3f} {pv:7.3f}{star}{mi:7.3f}  {new}")
        region_summary[k] = {nm: (auc, sign) for nm, auc, sign, *_ in rows}
    return region_summary


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    regions = ["ca", "uk", "chile"] if arg == "all" else [arg]
    summ = {}
    for rg in regions:
        summ[rg] = screen(rg)
    if len(regions) == 3:
        print(f"\n{'='*78}\n[지역간 36h 일관성] 단변량 AUC (방향) — CA·Chile 공통 강세 후보 탐색\n{'='*78}")
        allf = sorted({nm for rg in regions for nm in summ[rg].get(6, {})})
        print(f"  {'feature':<20}{'CA':>10}{'UK':>10}{'Chile':>10}")
        def cell(rg, nm):
            d = summ[rg].get(6, {}).get(nm)
            return f"{d[1]}{d[0]:.3f}" if d else "  -  "
        ranked = sorted(allf, key=lambda nm: -max(summ[rg].get(6, {}).get(nm, (0,))[0]
                                                   for rg in regions))
        for nm in ranked:
            print(f"  {nm:<20}{cell('ca',nm):>10}{cell('uk',nm):>10}{cell('chile',nm):>10}")


if __name__ == "__main__":
    main()
