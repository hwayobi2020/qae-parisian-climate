"""E 신규피처를 6h-정렬 시계열로 사전계산해 data/raw/efeat_{region}.npz 로 저장.
목적: Colab 전달 — 큰 raw(openmeteo, era5_ivt nc)가 없는 Colab에서도
      cv_causal 의 피처셋 E 를 재현(TabPFN 포함 실행)할 수 있게 한다.

저장 키(각 길이 T=64284, IVT 인덱스 s 에 1:1 정렬):
  pmsl     : 해면기압 (openmeteo pressure_msl)
  cloud    : 운량 (openmeteo cloud_cover)
  dir_sin  : IVT 방향 sin (viwve/viwvn 방위각)
  dir_cos  : IVT 방향 cos
(나머지 E 재료 — 다중스케일 IVT, 직전 AR활동, 강도 — 는 ivt 에서 직접 계산되므로 불필요.)

사용: python build_efeat_npz.py [ca|uk|chile|all]
"""
import os, sys, numpy as np
from screen_features_region import load_aux_6h, load_ivt_dir

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw")
IVT_FILE = {"ca": "ivt_sf_1980_2023.npy"}
TIMES_FILE = {"ca": "times_sf_1980_2023.npy"}


def build(region):
    ivf = IVT_FILE.get(region, f"ivt_{region}_1980_2023.npy")
    tif = TIMES_FILE.get(region, f"times_{region}_1980_2023.npy")
    ivt = np.load(os.path.join(RAW, ivf))
    times = np.load(os.path.join(RAW, tif))
    T = len(ivt)
    aux = load_aux_6h(region, T)
    dir_sin, dir_cos = load_ivt_dir(region, times)
    if dir_sin is None:
        raise FileNotFoundError(f"{region}: era5_ivt nc 없음 → IVT 방향 계산 불가")
    pmsl = aux["pressure_msl"].astype("float32")
    cloud = aux["cloud_cover"].astype("float32")
    dir_sin = dir_sin.astype("float32"); dir_cos = dir_cos.astype("float32")
    for nm, a in [("pmsl", pmsl), ("cloud", cloud), ("dir_sin", dir_sin), ("dir_cos", dir_cos)]:
        assert len(a) == T, f"{region} {nm} 길이 {len(a)} != T {T}"
    out = os.path.join(RAW, f"efeat_{region}.npz")
    np.savez(out, pmsl=pmsl, cloud=cloud, dir_sin=dir_sin, dir_cos=dir_cos)
    nan = {k: int(np.isnan(v).sum()) for k, v in
           [("pmsl", pmsl), ("cloud", cloud), ("dir_sin", dir_sin), ("dir_cos", dir_cos)]}
    print(f"[{region}] saved {os.path.basename(out)}  T={T}  NaN={nan}", flush=True)


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    for r in (["ca", "uk", "chile"] if arg == "all" else [arg]):
        build(r)


if __name__ == "__main__":
    main()
