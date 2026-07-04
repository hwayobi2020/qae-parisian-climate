"""전이 검증용 피처 내보내기 (Colab TabPFN/LSTM):
FE(tabular env 00Z컷), XSEQ(x64x7 IVT+웨이블릿 시퀀스), FC(x5 예보, 비예보=NaN),
y, oday, fmask(예보 있는 온셋). 저장: transfer_{region}.npz  사용: python export_transfer_feats.py
"""
import numpy as np, pywt, warnings; warnings.filterwarnings("ignore")
RAW = "data/raw/"; HH = 24; KS = 4; SEQ = 64
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
    z = np.load(f"gefs_ivt_{REGION}.npz"); S2 = z["s"].astype(int); OH2 = z["onset_hour"].astype(int)
    leads = list(z["leads"]); IVT = z["ivt"]
    fcmap = {}
    for k in range(len(S2)):
        Ls = [OH2[k] + o for o in range(6, HH, 6)]; Ls = [L for L in Ls if L in leads]
        if not Ls: continue
        mat = np.array([IVT[k, :, leads.index(L)] for L in Ls])
        if np.all(np.isnan(mat)): continue
        mm = np.nanmin(mat, axis=0)
        if not np.any(np.isnan(mm)): fcmap[int(S2[k])] = mm
    runs = []; i = 0
    while i < T:
        if ar6[i]:
            j = i
            while j < T and ar6[j]: j += 1
            runs.append((i, j)); i = j
        else: i += 1
    FE = []; XSEQ = []; FC = []; y = []; oday = []; fmask = []; sidx = []
    for s, e in runs:
        o = s // 4; s0 = o * 4
        if not (s0 - 63 >= 0 and o - 64 >= 0 and o < ND and o < NG and not (np.isnan(jet[o - 1]) or np.isnan(blk[o - 1]))): continue
        ivw = wl(ivt[s0 - 63:s0 + 1]); jw = wl(jet[o - 64:o]); bw = wl(blk[o - 64:o])
        d_row = ivw + [ivt[s0 - 3:s0 + 1].max(), jet[o - 1]] + jw + bw
        extra = [pmsl[s0], pmsl[s0] - pmsl[s0 - 4], dir_sin[s0], dir_cos[s0],
                 ar6[max(0, s0 - 119):s0 + 1].mean(), ar6[max(0, s0 - 239):s0 + 1].mean(), cloud[s0],
                 ivt[s0 - 7:s0 + 1].max(), ivt[s0 - 11:s0 + 1].max(), ivt[s0 - 27:s0 + 1].max(),
                 ivt[s0 - 27:s0 + 1].mean(), ivt[s0 - 27:s0 + 1].std()]
        FE.append(d_row + extra + [g[o - 1] for g in GRID] + [ea[s0], ep[s0]])
        w = ivt[s0 - SEQ:s0]
        XSEQ.append(np.stack([np.log1p(w)] + list(pywt.swt(w, 'db2', level=5, trim_approx=True, norm=True)), axis=1))
        has = int(s) in fcmap
        FC.append(fcmap[int(s)] if has else [np.nan] * 5); fmask.append(has)
        y.append(int(e - s >= KS)); oday.append(o); sidx.append(s)
    FE = np.array(FE, float); XSEQ = np.array(XSEQ, float); FC = np.array(FC, float)
    y = np.array(y); oday = np.array(oday); fmask = np.array(fmask)
    np.savez(f"transfer_{REGION}.npz", FE=FE, XSEQ=XSEQ, FC=FC, y=y, oday=oday, fmask=fmask, s=np.array(sidx))
    print(f"{REGION}: FE={FE.shape} XSEQ={XSEQ.shape} 예보온셋={int(fmask.sum())}/{len(y)} 양성={int(y.sum())} -> transfer_{REGION}.npz")
