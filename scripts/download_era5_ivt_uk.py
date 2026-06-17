"""
Download ERA5 IVT (integrated vapour transport) single point for UK SW coast
(50.0N, 5.0W) via CDS timeseries API, 1980-2023, subsampled to 6-hourly.
IVT = sqrt(viwve^2 + viwvn^2), mirrors scripts/download_era5_t2m.py exactly.

Output:
  - data/raw/ivt_uk_1980_2023.npy    (6-hourly IVT magnitude, kg/m/s)
  - data/raw/times_uk_1980_2023.npy  (6-hourly timestamps, for circ alignment)

Run: python scripts/download_era5_ivt_uk.py   (needs ~/.cdsapirc API key)
"""
import os
import zipfile
import cdsapi
import numpy as np

# UK SW coast (Cornwall). Change LAT/LON/TAG to retarget (e.g. Chile -33.0/-71.5).
LAT, LON = 50.0, -5.0
TAG = "uk"
VARS = [
    "vertical_integral_of_eastward_water_vapour_flux",   # viwve
    "vertical_integral_of_northward_water_vapour_flux",  # viwvn
]

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw")
os.makedirs(OUT_DIR, exist_ok=True)


def download():
    """Download viwve+viwvn via timeseries API in 5-year chunks (CSV)."""
    c = cdsapi.Client()
    boundaries = [
        (1980, 1984), (1985, 1989), (1990, 1994), (1995, 1999),
        (2000, 2004), (2005, 2009), (2010, 2014), (2015, 2019),
        (2020, 2023),
    ]
    times, ve, vn = [], [], []
    for start_year, end_year in boundaries:
        cache = os.path.join(OUT_DIR, f"era5_ivt_{TAG}_ts_{start_year}_{end_year}.csv")
        if os.path.exists(cache):
            print(f"  {start_year}-{end_year}: cached", flush=True)
            content = open(cache, "r").read()
        else:
            print(f"  {start_year}-{end_year}: downloading...", flush=True)
            zip_path = os.path.join(OUT_DIR, f"_tmp_ivt_{TAG}_{start_year}_{end_year}.zip")
            c.retrieve(
                "reanalysis-era5-single-levels-timeseries",
                {
                    "variable": VARS,
                    "location": {"latitude": LAT, "longitude": LON},
                    "date": f"{start_year}-01-01/{end_year}-12-31",
                    "data_format": "csv",
                },
                zip_path,
            )
            with zipfile.ZipFile(zip_path) as z:
                with z.open(z.namelist()[0]) as zf:
                    content = zf.read().decode("utf-8")
            open(cache, "w").write(content)
            os.remove(zip_path)
            print(f"    saved cache: {os.path.basename(cache)}", flush=True)

        lines = [l for l in content.split("\n") if l.strip()]
        hdr = [h.strip().lower() for h in lines[0].split(",")]

        def col(keys):
            for i, h in enumerate(hdr):
                if any(k in h for k in keys):
                    return i
            return None

        it = col(["valid_time", "time"])
        ie = col(["eastward", "viwve"])
        iN = col(["northward", "viwvn"])
        if ie is None or iN is None:  # fallback: take the two non-time/non-coord numeric columns
            skip = {it, col(["latitude", "lat"]), col(["longitude", "lon"])}
            nums = [i for i in range(len(hdr)) if i not in skip and i is not None]
            ie, iN = nums[0], nums[1]
        if start_year == 1980:
            print(f"    header={hdr}  -> time@{it} east@{ie} north@{iN}", flush=True)

        for l in lines[1:]:
            p = l.split(",")
            if len(p) <= max(it, ie, iN):
                continue
            times.append(p[it])
            ve.append(float(p[ie]))
            vn.append(float(p[iN]))
    return times, np.array(ve), np.array(vn)


def main():
    print(f"ERA5 IVT download - {TAG.upper()} ({LAT}N, {LON}E)")
    print("=" * 56)
    times, ve, vn = download()
    ivt = np.sqrt(ve ** 2 + vn ** 2)
    print(f"\n  total obs: {len(ivt)}  ({times[0]} .. {times[-1]})", flush=True)

    # subsample to 6-hourly (00/06/12/18) to match SF data
    def hour(t):
        t = t.replace("T", " ")
        return int(t.split(" ")[1].split(":")[0]) if " " in t else 0
    hours = np.array([hour(t) for t in times])
    m = np.isin(hours, [0, 6, 12, 18])
    ivt6 = ivt[m]
    t6 = [t for t, k in zip(times, m) if k]

    np.save(os.path.join(OUT_DIR, f"ivt_{TAG}_1980_2023.npy"), ivt6.astype(np.float32))
    np.save(os.path.join(OUT_DIR, f"times_{TAG}_1980_2023.npy"), np.array(t6))
    print(f"  saved ivt_{TAG}_1980_2023.npy: {len(ivt6)} 6h steps", flush=True)
    print(f"  IVT mean {ivt6.mean():.0f}  max {ivt6.max():.0f} kg/m/s", flush=True)

    # AR sanity: daily-max IVT > 250 frequency (SF coast ~ a few %/season)
    nd = len(ivt6) // 4
    dmax = ivt6[:nd * 4].reshape(-1, 4).max(1)
    print(f"  daily-max IVT>250 (AR-day frequency): {(dmax > 250).mean() * 100:.1f}%", flush=True)


if __name__ == "__main__":
    main()
