"""예보 기준 분모 = 예보 온셋(TT+FT). 운영 정합: 모든 피처를 **예보 온셋**(예보 IVT가 target day에 THR 넘는 시각)에 정렬.
- TT = 관측 온셋(d2c00) 중 예보도 THR 넘긴 것. TF(예보 미스)는 예보온셋 없어 자동 제외.
- FT = near-miss(d2c00fa) 중 예보가 THR 넘긴 것 = 거짓경보.
피처9 = 예보온셋 정렬 궤적. 라벨 = 관측 IVT(예보온셋+6/12/18h) 지속·omin. 저장 opdenom_{r}.npz.
사용: python build_op_denom.py   (d2c00fa 있는 지역만; Chile은 다운로드 후)
"""
import numpy as np, os
from sklearn.metrics import f1_score
IVTF = {"ca": "ivt_sf_1980_2023.npy", "uk": "ivt_uk_1980_2023.npy", "chile": "ivt_chile_1980_2023.npy"}
LEADS = [48, 54, 60, 66, 72, 78, 84, 90]; Li = {L: i for i, L in enumerate(LEADS)}


def find(n):
    for p in ["data/raw/" + n, n]:
        if os.path.exists(p): return p
    raise FileNotFoundError(n)


def detect(fnpz, ivt, THR, T, src):
    z = np.load(fnpz); S = z["s"].astype(int); IV = z["ivt"][:, 0, :]
    rows = []
    for k in range(len(S)):
        f = IV[k]
        if np.isnan(f).all(): continue
        s0 = (S[k] // 4) * 4
        onL = None                                   # 예보 온셋 = target day(48/54/60/66) 첫 THR 돌파
        for L in [48, 54, 60, 66]:
            if not np.isnan(f[Li[L]]) and f[Li[L]] >= THR: onL = L; break
        if onL is None: continue                     # 예보가 AR 안 부름 -> 분모 제외 (TF 자동 제외)
        traj = []; ok = True
        for off in [0, 6, 12, 18, 24]:
            L2 = onL + off
            if L2 not in Li or np.isnan(f[Li[L2]]): ok = False; break
            traj.append(float(f[Li[L2]]))
        if not ok: continue
        cont = [traj[1], traj[2], traj[3]]; fcv = min(cont)
        feat = traj + [min(cont), float(np.mean(cont)), float(np.std(cont)), traj[4] - traj[0]]
        j = s0 + (onL - 48) // 6                      # 예보온셋의 관측 절대 인덱스
        if j + 3 >= T: continue
        oc = ivt[j + 1:j + 4]; y = int(np.all(oc >= THR)); omin = float(oc.min())
        rows.append((feat, fcv, y, omin, s0 // 4, src))
    return rows


def folds(N, od, Nf=5, emb=64):
    f = N // (Nf + 1); out = []
    for k in range(1, Nf + 1):
        ts = k * f; te = (k + 1) * f if k < Nf else N
        out.append((np.array([j for j in range(0, ts) if od[j] <= od[ts] - emb]), np.arange(ts, te)))
    return out


for R in ["ca", "uk", "chile"]:
    fa = f"gefs_ivt_{R}_d2c00fa.npz"; tt = f"gefs_ivt_{R}_d2c00.npz"
    if not (os.path.exists(fa) and os.path.exists(tt) and "ivt" in np.load(fa).files and "ivt" in np.load(tt).files):
        print(f"[{R}] 예보 파일 미완(ivt 필드 없음, 다운로드 중) -> skip"); continue
    ivt = np.load(find(IVTF[R])).astype("float64"); T = len(ivt)
    THR = np.percentile(ivt.reshape(-1, 4).max(1), 85)
    rows = detect(f"gefs_ivt_{R}_d2c00.npz", ivt, THR, T, "TT") + detect(fa, ivt, THR, T, "FT")
    src = [r[5] for r in rows]; ys = np.array([r[2] for r in rows])
    D2 = np.array([r[0] for r in rows]); fcv = np.array([r[1] for r in rows])
    omin = np.array([r[3] for r in rows]); oday = np.array([r[4] for r in rows])
    o = np.argsort(oday); D2, fcv, ys2, omin, oday = D2[o], fcv[o], ys[o], omin[o], oday[o]
    np.savez(f"opdenom_{R}.npz", D2=D2, fcv=fcv, y=ys2, omin=omin, oday=oday, THR=THR)
    yt = []; pb = []
    for tr, te in folds(len(ys2), oday):
        if len(tr) < 40 or len(np.unique(ys2[tr])) < 2: continue
        pb.extend((fcv[te] >= THR).astype(int)); yt.extend(ys2[te])
    rf1 = f1_score(np.array(yt), np.array(pb), zero_division=0) if yt else float("nan")
    print(f"[{R}] 예보온셋(TT+FT)={len(rows)} | TT={src.count('TT')} FT={src.count('FT')} | 지속={int(ys.sum())} base={ys.mean():.3f} | raw F1={rf1:.3f} THR={THR:.1f} -> opdenom_{R}.npz")
