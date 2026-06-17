"""
Build UK circulation indices (jet, blocking) for the AR-duration model.
Downloads NCEP/NCAR daily Z500 & U250 over the North Atlantic (upstream of the
UK SW coast) via OPeNDAP, then derives circ_indices_uk.npz the SAME way as the
California circ_indices.npz:
  blocking = day-of-year anomaly of AREA-MEAN Z500
  jet      = AREA-MAX U250  (jet-core speed)
aligned to the UK IVT daily dates (times_uk_1980_2023.npy[::4]).

Run AFTER the UK IVT download (needs times_uk_1980_2023.npy).
NCEP OPeNDAP is public (no API key). 1980-2023.
"""
import os
import numpy as np
import xarray as xr

TAG = "uk"
# North Atlantic, upstream (west) of the UK. Mirrors California's N-Pacific basin.
LAT = slice(70, 20)        # 70N -> 20N (NCEP lat is descending)
LON = slice(280, 360)      # 280-360E = 80W-0W
RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw")
NCF = os.path.join(RAW, f"natl_z500_u250_1980_2023_{TAG}.nc")
base = "https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis/Dailies/pressure"


def download_nc():
    if os.path.exists(NCF):
        print(f"  {os.path.basename(NCF)}: cached", flush=True)
        return
    zl, ul = [], []
    for y in range(1980, 2024):
        for attempt in range(3):
            try:
                z = xr.open_dataset(f"{base}/hgt.{y}.nc")['hgt'].sel(level=500).sel(lat=LAT, lon=LON).load()
                u = xr.open_dataset(f"{base}/uwnd.{y}.nc")['uwnd'].sel(level=250).sel(lat=LAT, lon=LON).load()
                zl.append(z); ul.append(u); print(y, "ok", flush=True); break
            except Exception as e:
                print(y, "retry", e, flush=True)
                if attempt == 2: print(y, "FAIL", flush=True)
    Z = xr.concat(zl, "time"); U = xr.concat(ul, "time")
    ds = xr.Dataset({"z500": Z.drop_vars("level", errors="ignore"),
                     "u250": U.drop_vars("level", errors="ignore")})
    ds.to_netcdf(NCF)
    print("  saved", NCF, dict(ds.sizes), flush=True)


def build_indices():
    ds = xr.open_dataset(NCF)
    zm = ds['z500'].mean(['lat', 'lon'])                          # area-mean Z500
    zc = zm.groupby('time.dayofyear').mean('time')               # climatology
    zanom = (zm.groupby('time.dayofyear') - zc).values           # blocking = anomaly
    jet = ds['u250'].max(['lat', 'lon']).values                  # jet = area-max U250
    zt = ds['time'].values.astype('datetime64[D]')

    tpath = os.path.join(RAW, f"times_{TAG}_1980_2023.npy")
    assert os.path.exists(tpath), "times_uk_1980_2023.npy 없음 - IVT 다운로드 먼저 실행"
    ti = np.load(tpath)[::4].astype('datetime64[D]')             # 6h -> daily
    print(f"  len z/ivt-daily: {len(zt)}/{len(ti)}  match first/last: "
          f"{zt[0] == ti[0]}/{zt[-1] == ti[-1]}", flush=True)

    bd = {d: v for d, v in zip(zt, zanom)}
    jd = {d: v for d, v in zip(zt, jet)}
    block = np.array([bd.get(d, np.nan) for d in ti])
    jetA = np.array([jd.get(d, np.nan) for d in ti])
    print(f"  nan block/jet: {np.isnan(block).sum()}/{np.isnan(jetA).sum()}", flush=True)
    print(f"  block range [{np.nanmin(block):.1f}, {np.nanmax(block):.1f}]  "
          f"jet range [{np.nanmin(jetA):.1f}, {np.nanmax(jetA):.1f}]", flush=True)
    out = os.path.join(RAW, f"circ_indices_{TAG}.npz")
    np.savez(out, blocking=block, jet=jetA, dates=ti.astype(str))
    print(f"  saved {os.path.basename(out)}, len={len(block)}", flush=True)


def main():
    print(f"UK circulation indices (N Atlantic {LAT.start}-{LAT.stop}N, {LON.start}-{LON.stop}E)")
    print("=" * 60)
    download_nc()
    build_indices()


if __name__ == "__main__":
    main()
