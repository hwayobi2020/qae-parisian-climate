"""
Download real weather (Open-Meteo Historical API, ERA5-based, no key) at each AR
point, 1980-2023, 6-hourly. Real weather variables to add as onset features:
2m temperature, MSL pressure, precipitation, relative humidity, 10m wind, cloud.
3 regions all gap-free (incl. Chile, where station data was missing).

Output: data/raw/openmeteo_<region>_1980_2023.npz  (times + each variable, 6-hourly)

Run: python scripts/download_openmeteo_weather.py [ca|uk|chile|all]
"""
import os
import sys
import time
import numpy as np
import requests

PTS = {"ca": (37.77, -122.42), "uk": (50.0, -5.0), "chile": (-33.0, -71.5)}
VARS = ["temperature_2m", "pressure_msl", "precipitation",
        "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m", "cloud_cover"]
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw")
os.makedirs(OUT_DIR, exist_ok=True)


def fetch_region(region):
    la, lo = PTS[region]
    times = []
    data = {v: [] for v in VARS}
    for y in range(1980, 2024):
        url = ("https://archive-api.open-meteo.com/v1/archive"
               f"?latitude={la}&longitude={lo}&start_date={y}-01-01&end_date={y}-12-31"
               f"&hourly={','.join(VARS)}&timezone=UTC")
        for attempt in range(4):
            try:
                r = requests.get(url, timeout=60).json()
                h = r["hourly"]; break
            except Exception as ex:
                print(f"  {region} {y} retry {ex}", flush=True); time.sleep(3)
        else:
            print(f"  {region} {y} FAIL", flush=True); continue
        times += h["time"]
        for v in VARS:
            data[v] += h[v]
        print(f"  {region} {y}: {len(h['time'])} hourly", flush=True)
    # 6시간 서브샘플 (00/06/12/18) -> IVT와 정렬
    t = np.array(times)
    hr = np.array([int(s[11:13]) for s in t])
    m = np.isin(hr, [0, 6, 12, 18])
    out = {"times": t[m].astype(str)}
    for v in VARS:
        arr = np.array([np.nan if x is None else x for x in data[v]], dtype="float64")
        out[v] = arr[m]
    path = os.path.join(OUT_DIR, f"openmeteo_{region}_1980_2023.npz")
    np.savez(path, **out)
    print(f"  saved {os.path.basename(path)}: {m.sum()} 6h steps, vars={VARS}", flush=True)
    nanpct = {v: round(np.isnan(out[v]).mean() * 100, 1) for v in VARS}
    print(f"  결측%: {nanpct}", flush=True)


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    regions = list(PTS) if arg == "all" else [arg]
    for r in regions:
        print(f"=== {r} ({PTS[r]}) ===", flush=True)
        fetch_region(r)


if __name__ == "__main__":
    main()
