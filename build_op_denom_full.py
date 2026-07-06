"""0.5THR 전체 분모 (TT+TF+FT+FF) + D-2컷 env-44(인코더용) + 3클래스 라벨.
- 후보 = 관측 온셋(d2c00, TT+TF) + near-miss(d2c00fa, FT+FF).
- 피처 = 예보 peak 정렬(target day 리드 48/54/60/66 중 최대 예보 시각 기준 궤적). 운영정합(예측시 예보만).
- 라벨 y = 관측 온셋 지속(온셋: ivt[s+1:s+4]>=THR / near-miss: 0=그날 온셋 아님). omin = 관측 지속창 min.
- ENV = 각 후보일 oday 의 D-2컷 env-44 (export_d2env.py 와 동일 슬라이싱; c=oday*4-GAP, od=oday-2). D-2컷 불가/일별 NaN 이면 NaN 행.
- y3 = 3클래스 인코더 라벨: 비온셋 0 / 온셋<24h(omin<THR) 1 / 온셋>=24h(omin>=THR) 2.  (기존 이진 y == (y3==2))
저장 opdenom_full_{r}.npz {D2, fcv, y, omin, oday, THR, ENV, y3}.  사용: python build_op_denom_full.py
"""
import numpy as np, os, pywt, warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import f1_score
IVTF = {"ca": "ivt_sf_1980_2023.npy", "uk": "ivt_uk_1980_2023.npy", "chile": "ivt_chile_1980_2023.npy"}
LEADS = [48, 54, 60, 66, 72, 78, 84, 90]; Li = {L: i for i, L in enumerate(LEADS)}
GAP = 8   # GAP=8 six-hour steps = 2일 (D-2 컷)
GRIDF = ["zanom_landfall", "ridge_up", "trough_up", "waveamp_up", "u250_landfall", "u250_up_mean", "merid_grad_lf", "ghgn", "ghgs", "jetlat_local"]


def find(n):
    for p in ["data/raw/" + n, n]:
        if os.path.exists(p): return p
    raise FileNotFoundError(n)


def wl(a, lvl=5): return [c[-1] for c in pywt.swt(a, 'db2', level=lvl, trim_approx=True, norm=True)]


def peak_feats(f):
    day = [(L, f[Li[L]]) for L in [48, 54, 60, 66] if not np.isnan(f[Li[L]])]
    if not day: return None
    onL = max(day, key=lambda x: x[1])[0]                 # 예보 peak 시각(그날 최대 예보 IVT)
    if any((onL + o) not in Li or np.isnan(f[Li[onL + o]]) for o in [6, 12, 18, 24]): return None
    traj = [float(f[Li[onL + o]]) for o in [0, 6, 12, 18, 24]]
    cont = [traj[1], traj[2], traj[3]]
    feat = traj + [min(cont), float(np.mean(cont)), float(np.std(cont)), traj[4] - traj[0]]
    return feat, min(cont)


def folds(N, od, Nf=5, emb=64):
    f = N // (Nf + 1); out = []
    for k in range(1, Nf + 1):
        ts = k * f; te = (k + 1) * f if k < Nf else N
        out.append((np.array([j for j in range(0, ts) if od[j] <= od[ts] - emb]), np.arange(ts, te)))
    return out


for R in ["ca", "uk", "chile"]:
    fa = f"gefs_ivt_{R}_d2c00fa.npz"; tt = f"gefs_ivt_{R}_d2c00.npz"
    if not (os.path.exists(fa) and os.path.exists(tt) and "ivt" in np.load(fa).files and "ivt" in np.load(tt).files):
        print(f"[{R}] 예보 파일 미완 -> skip"); continue
    ivt = np.load(find(IVTF[R])).astype("float64"); T = len(ivt)
    THR = np.percentile(ivt.reshape(-1, 4).max(1), 85); ar6 = ivt > THR

    # ===== env 소스 로드 (export_d2env.py 와 동일 파일/키) =====
    CIRCF = "circ_indices.npz" if R == "ca" else f"circ_indices_{R}.npz"
    ci = np.load(find(CIRCF)); jet = ci["jet"].astype("float64"); blk = ci["blocking"].astype("float64")
    ef = np.load(find(f"efeat_{R}.npz")); pmsl = ef["pmsl"].astype("float64"); cloud = ef["cloud"].astype("float64")
    dir_sin = ef["dir_sin"].astype("float64"); dir_cos = ef["dir_cos"].astype("float64")
    gz = np.load(find(f"gridfeat_{R}.npz")); GRID = [gz[k].astype("float64") for k in GRIDF]; NG = len(GRID[0])
    ea = np.load(find("enso_anom.npy")).astype("float64"); ep = np.load(find("enso_phase.npy")).astype("float64")
    ND = len(jet)

    def env_feats(oday):
        """후보일 oday 의 D-2컷 env-44. export_d2env.py 줄 40-48 과 동일 슬라이싱."""
        s0 = oday * 4; c = s0 - GAP; od = oday - 2                     # D-2 컷 시점 (관측 2일 전)
        if not (c - 63 >= 0 and od - 64 >= 0 and od >= 1 and od < NG and od < ND
                and not (np.isnan(jet[od - 1]) or np.isnan(blk[od - 1]))):
            return [np.nan] * 44
        ivw = wl(ivt[c - 63:c + 1]); jw = wl(jet[od - 64:od]); bw = wl(blk[od - 64:od])
        d_row = ivw + [ivt[c - 3:c + 1].max(), jet[od - 1]] + jw + bw
        extra = [pmsl[c], pmsl[c] - pmsl[c - 4], dir_sin[c], dir_cos[c],
                 ar6[max(0, c - 119):c + 1].mean(), ar6[max(0, c - 239):c + 1].mean(), cloud[c],
                 ivt[c - 7:c + 1].max(), ivt[c - 11:c + 1].max(), ivt[c - 27:c + 1].max(),
                 ivt[c - 27:c + 1].mean(), ivt[c - 27:c + 1].std()]
        return d_row + extra + [g[od - 1] for g in GRID] + [ea[c], ep[c]]

    rows = []
    z = np.load(tt); S = z["s"].astype(int); IV = z["ivt"][:, 0, :]           # 관측온셋 TT+TF
    for k in range(len(S)):
        f = IV[k]
        if np.isnan(f).all(): continue
        pf = peak_feats(f)
        if pf is None: continue
        s = S[k]
        if s + 3 >= T: continue
        omin = float(ivt[s + 1:s + 4].min()); y = int(omin >= THR)             # 관측 온셋 지속
        oday = s // 4
        y3 = 2 if omin >= THR else 1                                           # 온셋: >=24h(2) / <24h(1)
        rows.append((pf[0], pf[1], y, omin, oday, "onset", env_feats(oday), y3))
    zf = np.load(fa); Sf = zf["s"].astype(int); IVf = zf["ivt"][:, 0, :]       # near-miss FT+FF
    for k in range(len(Sf)):
        f = IVf[k]
        if np.isnan(f).all(): continue
        pf = peak_feats(f)
        if pf is None: continue
        s0 = (Sf[k] // 4) * 4
        if s0 + 3 >= T: continue
        omin = float(ivt[s0 + 1:s0 + 4].min()); y = 0                          # 온셋 아님(관측<THR)
        oday = s0 // 4
        y3 = 0                                                                 # 비온셋
        rows.append((pf[0], pf[1], y, omin, oday, "nm", env_feats(oday), y3))

    src = [r[5] for r in rows]; ys = np.array([r[2] for r in rows])
    D2 = np.array([r[0] for r in rows]); fcv = np.array([r[1] for r in rows])
    omin = np.array([r[3] for r in rows]); oday = np.array([r[4] for r in rows])
    ENV = np.array([r[6] for r in rows], float); y3 = np.array([r[7] for r in rows])
    o = np.argsort(oday)
    D2, fcv, ys, omin, oday, ENV, y3 = D2[o], fcv[o], ys[o], omin[o], oday[o], ENV[o], y3[o]
    np.savez(f"opdenom_full_{R}.npz", D2=D2, fcv=fcv, y=ys, omin=omin, oday=oday, THR=THR, ENV=ENV, y3=y3)

    yt = []; pb = []
    for tr, te in folds(len(ys), oday):
        if len(tr) < 40 or len(np.unique(ys[tr])) < 2: continue
        pb.extend((fcv[te] >= THR).astype(int)); yt.extend(ys[te])
    rf1 = f1_score(np.array(yt), np.array(pb), zero_division=0) if yt else float("nan")
    nan_env = int(np.isnan(ENV).any(1).sum())
    print(f"[{R}] 분모 {len(rows)} (온셋 {src.count('onset')} + near-miss {src.count('nm')}) | 지속 {int(ys.sum())} base {ys.mean():.3f} "
          f"| raw F1={rf1:.3f} | ENV NaN행 {nan_env}/{len(ENV)} | y3 [비온셋 {int((y3==0).sum())} / 온셋<24h {int((y3==1).sum())} / 온셋>=24h {int((y3==2).sum())}] -> opdenom_full_{R}.npz")
