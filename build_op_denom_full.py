"""0.5THR 전체 분모 (TT+TF+FT+FF) — 지속 horizon 24h·30h.
- 후보 = 관측 온셋(d2c00, TT+TF) + near-miss(d2c00fa, FT+FF).
- 피처 D2(9) = 예보 peak 정렬(target day 리드 48/54/60/66 중 최대 예보 시각 기준 궤적, onset·+6·+12·+18·+24). 운영정합.
- horizon H(24/30): 지속창 = 온셋+6 … 온셋+(H-6). fcv_H = 예보 min(온셋+6..+(H-6)), omin_H = 관측 min(동일창).
    y_H = 온셋이고 omin_H>=THR (near-miss=0).  30h 는 온셋+24h 까지 필요 → 예보 리드 90h 안(무손실).
- 인코더용 ENV 44(D-2컷, horizon 무관) + y3(3클래스, 라벨은 horizon별 omin 기준) 을 전 horizon npz 에 동봉.
- D8(8) = 8-고정 리드(48~90h) 원시 예보 IVT (peak정렬 vs 고정리드 ablation용). fcv(raw 기준선)는 계속 peak정렬.
저장: opdenom_full_{r}{,_18h,_30h}.npz {D2,fcv,y,omin,oday,THR,ENV,y3,D8}.  사용: python build_op_denom_full.py
"""
import numpy as np, os, pywt, warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import f1_score
IVTF = {"ca": "ivt_sf_1980_2023.npy", "uk": "ivt_uk_1980_2023.npy", "chile": "ivt_chile_1980_2023.npy"}
LEADS = [48, 54, 60, 66, 72, 78, 84, 90]; Li = {L: i for i, L in enumerate(LEADS)}
GAP = 8   # GAP=8 six-hour steps = 2일 (D-2 컷)
GRIDF = ["zanom_landfall", "ridge_up", "trough_up", "waveamp_up", "u250_landfall", "u250_up_mean", "merid_grad_lf", "ghgn", "ghgs", "jetlat_local"]
HORIZONS = [18, 24, 30]   # 지속 임계(시간). 18h=온셋+12h·24h=+18h·30h=+24h 필요 (전부 예보 리드 90h 안, 무손실)


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
    traj = [float(f[Li[onL + o]]) for o in [0, 6, 12, 18, 24]]   # onset,+6,+12,+18,+24
    cont = [traj[1], traj[2], traj[3]]
    feat = traj + [min(cont), float(np.mean(cont)), float(np.std(cont)), traj[4] - traj[0]]   # D2 9개
    return feat, traj


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

    # ===== env 소스 로드 (export_d2env.py 와 동일 파일/키) — 24h ENV 동봉용 =====
    CIRCF = "circ_indices.npz" if R == "ca" else f"circ_indices_{R}.npz"
    ci = np.load(find(CIRCF)); jet = ci["jet"].astype("float64"); blk = ci["blocking"].astype("float64")
    ef = np.load(find(f"efeat_{R}.npz")); pmsl = ef["pmsl"].astype("float64"); cloud = ef["cloud"].astype("float64")
    dir_sin = ef["dir_sin"].astype("float64"); dir_cos = ef["dir_cos"].astype("float64")
    gz = np.load(find(f"gridfeat_{R}.npz")); GRID = [gz[k].astype("float64") for k in GRIDF]; NG = len(GRID[0])
    ea = np.load(find("enso_anom.npy")).astype("float64"); ep = np.load(find("enso_phase.npy")).astype("float64")
    ND = len(jet)

    def env_feats(oday):
        """후보일 oday 의 D-2컷 env-44. export_d2env.py 줄 40-48 과 동일 슬라이싱."""
        s0 = oday * 4; c = s0 - GAP; od = oday - 2
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

    # ===== 후보 수집 (온셋 + near-miss). 각 행: feat9(peak정렬), traj5, f8(8고정리드 원시), base_idx(6h), oday, env44, is_onset =====
    cand = []
    z = np.load(tt); S = z["s"].astype(int); IV = z["ivt"][:, 0, :]           # 관측온셋 TT+TF
    for k in range(len(S)):
        f = IV[k]
        if np.isnan(f).all(): continue
        pf = peak_feats(f)
        if pf is None: continue
        s = int(S[k])
        cand.append((pf[0], pf[1], np.asarray(f, float), s, s // 4, env_feats(s // 4), True))
    zf = np.load(fa); Sf = zf["s"].astype(int); IVf = zf["ivt"][:, 0, :]       # near-miss FT+FF
    for k in range(len(Sf)):
        f = IVf[k]
        if np.isnan(f).all(): continue
        pf = peak_feats(f)
        if pf is None: continue
        s0 = (int(Sf[k]) // 4) * 4
        cand.append((pf[0], pf[1], np.asarray(f, float), s0, s0 // 4, env_feats(s0 // 4), False))

    # ===== horizon 별 라벨/타깃 생성·저장 =====
    for H in HORIZONS:
        npts = H // 6                                         # 관측 연속점 수 인덱스 (24h:4, 30h:5)
        rows = []
        for feat, traj, f8, base, oday, env, is_on in cand:
            if base + npts > T: continue                      # 관측 지속창이 시계열 끝 초과 → 제외
            fcv = float(min(traj[1:npts]))                    # 예보 min(온셋+6..+(H-6)) — peak정렬 유지(raw 기준선)
            omin = float(ivt[base + 1:base + npts].min())     # 관측 min(동일창)
            y = int(is_on and omin >= THR)                    # 온셋이고 지속
            y3 = 0 if not is_on else (2 if omin >= THR else 1)
            rows.append((feat, fcv, y, omin, oday, env, y3, is_on, f8))
        D2 = np.array([r[0] for r in rows]); fcv = np.array([r[1] for r in rows])
        ys = np.array([r[2] for r in rows]); omin = np.array([r[3] for r in rows]); oday = np.array([r[4] for r in rows])
        ENV = np.array([r[5] for r in rows], float); y3 = np.array([r[6] for r in rows])
        on = np.array([r[7] for r in rows]); D8 = np.array([r[8] for r in rows], float)   # 8-고정 리드 원시 예보 IVT (정렬비교용)
        o = np.argsort(oday)
        D2, fcv, ys, omin, oday, ENV, y3, on, D8 = D2[o], fcv[o], ys[o], omin[o], oday[o], ENV[o], y3[o], on[o], D8[o]
        suf = "" if H == 24 else f"_{H}h"
        np.savez(f"opdenom_full_{R}{suf}.npz", D2=D2, fcv=fcv, y=ys, omin=omin, oday=oday, THR=THR, ENV=ENV, y3=y3, D8=D8)   # ENV 44·y3(3클래스, horizon별 라벨) 전 horizon 동봉
        yt = []; pb = []
        for tr, te in folds(len(ys), oday):
            if len(tr) < 40 or len(np.unique(ys[tr])) < 2: continue
            pb.extend((fcv[te] >= THR).astype(int)); yt.extend(ys[te])
        rf1 = f1_score(np.array(yt), np.array(pb), zero_division=0) if yt else float("nan")
        print(f"[{R} {H}h] 분모 {len(rows)} (온셋 {int(on.sum())} + nm {int((~on).sum())}) | 지속 {int(ys.sum())} base {ys.mean():.3f} | raw F1={rf1:.3f} -> opdenom_full_{R}{suf}.npz")
