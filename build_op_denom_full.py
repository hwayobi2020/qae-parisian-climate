"""0.5THR 전체 분모 (TT+TF+FT+FF): "≥0.5THR 날에 2일전 예보로 24h+ 지속 AR 발생 예측".
- 후보 = 관측 온셋(d2c00, TT+TF) + near-miss(d2c00fa, FT+FF).
- 피처 = 예보 peak 정렬(target day 리드 48/54/60/66 중 최대 예보 시각 기준 궤적). 운영정합(예측시 예보만).
- 라벨 y = 관측 온셋 지속(온셋: ivt[s+1:s+4]≥THR / near-miss: 0=그날 온셋 아님). omin = 관측 지속창 min.
저장 opdenom_full_{r}.npz.  사용: python build_op_denom_full.py
"""
import numpy as np, os
from sklearn.metrics import f1_score
IVTF = {"ca": "ivt_sf_1980_2023.npy", "uk": "ivt_uk_1980_2023.npy", "chile": "ivt_chile_1980_2023.npy"}
LEADS = [48, 54, 60, 66, 72, 78, 84, 90]; Li = {L: i for i, L in enumerate(LEADS)}


def find(n):
    for p in ["data/raw/" + n, n]:
        if os.path.exists(p): return p
    raise FileNotFoundError(n)


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
    THR = np.percentile(ivt.reshape(-1, 4).max(1), 85)
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
        rows.append((pf[0], pf[1], y, omin, s // 4, "onset"))
    zf = np.load(fa); Sf = zf["s"].astype(int); IVf = zf["ivt"][:, 0, :]       # near-miss FT+FF
    for k in range(len(Sf)):
        f = IVf[k]
        if np.isnan(f).all(): continue
        pf = peak_feats(f)
        if pf is None: continue
        s0 = (Sf[k] // 4) * 4
        if s0 + 3 >= T: continue
        omin = float(ivt[s0 + 1:s0 + 4].min()); y = 0                          # 온셋 아님(관측<THR)
        rows.append((pf[0], pf[1], y, omin, s0 // 4, "nm"))

    src = [r[5] for r in rows]; ys = np.array([r[2] for r in rows])
    D2 = np.array([r[0] for r in rows]); fcv = np.array([r[1] for r in rows])
    omin = np.array([r[3] for r in rows]); oday = np.array([r[4] for r in rows])
    o = np.argsort(oday); D2, fcv, ys, omin, oday = D2[o], fcv[o], ys[o], omin[o], oday[o]
    np.savez(f"opdenom_full_{R}.npz", D2=D2, fcv=fcv, y=ys, omin=omin, oday=oday, THR=THR)
    yt = []; pb = []
    for tr, te in folds(len(ys), oday):
        if len(tr) < 40 or len(np.unique(ys[tr])) < 2: continue
        pb.extend((fcv[te] >= THR).astype(int)); yt.extend(ys[te])
    rf1 = f1_score(np.array(yt), np.array(pb), zero_division=0) if yt else float("nan")
    print(f"[{R}] 분모 {len(rows)} (온셋 {src.count('onset')} + near-miss {src.count('nm')}) | 지속 {int(ys.sum())} base {ys.mean():.3f} | raw F1={rf1:.3f} -> opdenom_full_{R}.npz")
