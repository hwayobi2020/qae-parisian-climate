"""
Download ERA5 IVT for central Chile coast (Valparaiso, 33.0S, 71.5W) via the
FULL single-levels dataset. Same method as download_era5_ivt_uk.py, coords only.
IVT magnitude = sqrt(viwve^2 + viwvn^2). 2-year chunks, 6-hourly, 1980-2023.

Output:
  - data/raw/era5_ivt_chile_<y0>_<y1>.nc
  - data/raw/ivt_chile_1980_2023.npy
  - data/raw/times_chile_1980_2023.npy

Run: python scripts/download_era5_ivt_chile.py            (resumable)
     python scripts/download_era5_ivt_chile.py --test     (first chunk only)
needs ~/.cdsapirc API key.
"""
import os
import sys
import glob
import cdsapi
import numpy as np
import xarray as xr

# Central Chile coast (Valparaiso / Santiago). Andes windward.
LAT, LON = -33.0, -71.5
TAG = "chile"
AREA = [-32.5, -72.0, -33.5, -71.0]   # N, W, S, E  (small box around the point)
VARS = [
    "vertical_integral_of_eastward_water_vapour_flux",   # viwve
    "vertical_integral_of_northward_water_vapour_flux",  # viwvn
]
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw")
os.makedirs(OUT_DIR, exist_ok=True)
CHUNKS = [(y, y + 1) for y in range(1980, 2024, 2)]   # 22 chunks


def download(test=False):
    c = cdsapi.Client()
    chunks = CHUNKS[:1] if test else CHUNKS
    for s, e in chunks:
        out = os.path.join(OUT_DIR, f"era5_ivt_{TAG}_{s}_{e}.nc")
        if os.path.exists(out):
            print(f"  {s}-{e}: cached", flush=True)
            continue
        print(f"  {s}-{e}: downloading...", flush=True)
        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": ["reanalysis"],
                "variable": VARS,
                "year": [str(y) for y in range(s, e + 1)],
                "month": [f"{m:02d}" for m in range(1, 13)],
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": ["00:00", "06:00", "12:00", "18:00"],
                "area": AREA,
                "data_format": "netcdf",
                "download_format": "unarchived",
            },
            out,
        )
        print(f"    saved {os.path.basename(out)}", flush=True)


def build():
    files = sorted(glob.glob(os.path.join(OUT_DIR, f"era5_ivt_{TAG}_*.nc")))
    print(f"\n  building from {len(files)} chunk files", flush=True)
    ivts, times = [], []
    tname = None
    for f in files:
        ds = xr.open_dataset(f)
        dv = list(ds.data_vars)
        assert len(dv) >= 2, f"expected 2 flux vars, got {dv} in {f}"
        latname = "latitude" if "latitude" in ds.coords else "lat"
        lonname = "longitude" if "longitude" in ds.coords else "lon"
        if tname is None:
            tname = "valid_time" if "valid_time" in ds.coords else "time"
            print(f"    vars={dv}  coords time={tname} lat={latname} lon={lonname}", flush=True)
        ve = ds[dv[0]].sel({latname: LAT, lonname: LON}, method="nearest").values
        vn = ds[dv[1]].sel({latname: LAT, lonname: LON}, method="nearest").values
        ivts.append(np.sqrt(ve.astype("float64") ** 2 + vn.astype("float64") ** 2))
        times.append(ds[tname].values)
        ds.close()
    ivt = np.concatenate(ivts)
    t = np.concatenate(times)
    order = np.argsort(t)
    ivt, t = ivt[order], t[order]
    np.save(os.path.join(OUT_DIR, f"ivt_{TAG}_1980_2023.npy"), ivt.astype(np.float32))
    np.save(os.path.join(OUT_DIR, f"times_{TAG}_1980_2023.npy"), t.astype("datetime64[ns]"))
    print(f"  saved ivt_{TAG}_1980_2023.npy: {len(ivt)} 6h steps", flush=True)
    print(f"  IVT mean {ivt.mean():.0f}  max {ivt.max():.0f} kg/m/s", flush=True)
    nd = len(ivt) // 4
    dmax = ivt[:nd * 4].reshape(-1, 4).max(1)
    print(f"  daily-max IVT>250 (AR-day frequency): {(dmax > 250).mean() * 100:.1f}%", flush=True)


def main():
    test = "--test" in sys.argv
    print(f"ERA5 IVT (full single-levels) - {TAG.upper()} ({LAT}N, {LON}E), test={test}")
    print("=" * 60)
    download(test=test)
    build()


if __name__ == "__main__":
    main()
