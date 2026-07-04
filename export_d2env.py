"""D-2 컷 env 피처: 인코더용. 모든 env 관측을 온셋 2일 전(s0-8, 일단위 o-2)에서 끝냄 (D-2 예보 시점과 정합).
transfer_{r}.npz 와 동일 온셋 순서/길이 (D-2 컷 불가 온셋은 NaN 행). 저장 d2env_{r}.npz {s, FE}.
사용: python export_d2env.py
"""
import numpy as np, pywt, warnings; warnings.filterwarnings("ignore")
RAW = "data/raw/"; HH = 24; KS = 4; GAP = 8   # GAP=8 six-hour steps = 2일
GRIDF = ["zanom_landfall", "ridge_up", "trough_up", "waveamp_up", "u250_landfall", "u250_up_mean", "merid_grad_lf", "ghgn", "ghgs", "jetlat_local"]


def wl(a, lvl=5): return [c[-1] for c in pywt.swt(a, 'db2', level=lvl, trim_approx=True, norm=True)]


for REGION in ["ca", "uk", "chile"]:
    IVTF = "ivt_sf_1980_2023.npy" if REGION == "ca" else f"ivt_{REGION}_1980_2023.npy"
    CIRCF = "circ_indices.npz" if REGION == "ca" else f"circ_indices_{REGION}.npz"
    ivt = np.load(RAW + IVTF).astype("float64")
    ci = np.load(RAW + CIRCF); jet = ci["jet"].astype("float64"); blk = ci["blocking"].astype("float64")
    ef = np.load(RAW + f"efeat_{REGION}.npz"); pmsl = ef["pmsl"].astype("float64"); cloud = ef["cloud"].astype("float64")
    dir_sin = ef["dir_sin"].astype("float64"); dir_cos = ef["dir_cos"].astype("float64")
    gz = np.load(RAW + f"gridfeat_{REGION}.npz"); GRID = [gz[k].astype("float64") for k in GRIDF]; NG = len(GRID[0])
    ea = np.load(RAW + "enso_anom.npy").astype("float64"); ep = np.load(RAW + "enso_phase.npy").astype("float64")
    ND = len(jet); T = len(ivt)
    THR = np.percentile(ivt.reshape(-1, 4).max(1), 85); ar6 = ivt > THR
    runs = []; i = 0
    while i < T:
        if ar6[i]:
            j = i
            while j < T and ar6[j]: j += 1
            runs.append((i, j)); i = j
        else: i += 1
    FE = []; sidx = []
    for s, e in runs:
        o = s // 4; s0 = o * 4
        # transfer와 동일 포함 기준 (온셋 집합 일치)
        if not (s0 - 63 >= 0 and o - 64 >= 0 and o < ND and o < NG and not (np.isnan(jet[o - 1]) or np.isnan(blk[o - 1]))): continue
        c = s0 - GAP; od = o - 2                     # D-2 컷 시점
        sidx.append(s)
        # D-2 컷이 불가능하거나(초반) daily env NaN이면 NaN 행
        if not (c - 63 >= 0 and od - 64 >= 0 and od >= 1 and od < NG and not (np.isnan(jet[od - 1]) or np.isnan(blk[od - 1]))):
            FE.append([np.nan] * 44); continue
        ivw = wl(ivt[c - 63:c + 1]); jw = wl(jet[od - 64:od]); bw = wl(blk[od - 64:od])
        d_row = ivw + [ivt[c - 3:c + 1].max(), jet[od - 1]] + jw + bw
        extra = [pmsl[c], pmsl[c] - pmsl[c - 4], dir_sin[c], dir_cos[c],
                 ar6[max(0, c - 119):c + 1].mean(), ar6[max(0, c - 239):c + 1].mean(), cloud[c],
                 ivt[c - 7:c + 1].max(), ivt[c - 11:c + 1].max(), ivt[c - 27:c + 1].max(),
                 ivt[c - 27:c + 1].mean(), ivt[c - 27:c + 1].std()]
        FE.append(d_row + extra + [g[od - 1] for g in GRID] + [ea[c], ep[c]])
    FE = np.array(FE, float)
    np.savez(f"d2env_{REGION}.npz", s=np.array(sidx), FE=FE)
    print(f"{REGION}: FE={FE.shape} (D-2컷) | NaN행 {int(np.isnan(FE).any(1).sum())}/{len(FE)} -> d2env_{REGION}.npz")
