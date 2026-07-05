"""CA 예보 기준 분모 = TT + TF + FT (거짓경보 포함).
- TT+TF = 관측 온셋 801 (기존 d2feat, 관측온셋 정렬 피처 + y + omin). 예보가 잡았든(TT) 놓쳤든(TF) 다 포함.
- FT = near-miss 날 중 예보가 THR 넘긴 것(예보온셋 정렬 피처, 관측 라벨). = 헛부름(대개 y=0).
저장 opdenom_ca.npz.  사용: python build_op_denom.py
"""
import numpy as np, os
from sklearn.metrics import f1_score
def find(n):
    for p in ["data/raw/" + n, n]:
        if os.path.exists(p): return p
    raise FileNotFoundError(n)


ivt = np.load(find("ivt_sf_1980_2023.npy")).astype("float64"); T = len(ivt)
THR = np.percentile(ivt.reshape(-1, 4).max(1), 85)

# ---- TT+TF : 관측 온셋 801 (관측온셋 정렬 피처, 기존) ----
d = np.load("transfer_ca.npz"); yA, odA, S = d["y"], d["oday"], d["s"].astype(int)
de = np.load("d2env_ca.npz"); OMIN = de["omin"]
z = np.load("d2feat_ca.npz"); s2 = z["s"].astype(int); D2 = z["D2"]; D2min = z["D2min"]
mp = {int(s2[k]): k for k in range(len(s2))}
keep = [i for i in range(len(S)) if int(S[i]) in mp and not np.isnan(OMIN[i])]
ri = np.array([mp[int(S[i])] for i in keep])
rows = [(D2[ri][t], float(D2min[ri][t]), int(yA[keep][t]), float(OMIN[keep][t]), int(odA[keep][t]), "obs")
        for t in range(len(keep))]
n_obs = len(rows); n_pos_obs = sum(r[2] for r in rows)

# ---- FT : near-miss 예보온셋 (예보온셋 정렬 피처, 관측 라벨) ----
LEADS = [48, 54, 60, 66, 72, 78, 84, 90]; Li = {L: i for i, L in enumerate(LEADS)}
zf = np.load("gefs_ivt_ca_d2c00fa.npz"); Sf = zf["s"].astype(int); IVf = zf["ivt"][:, 0, :]
n_fa = 0
for k in range(len(Sf)):
    f = IVf[k]
    if np.isnan(f).all(): continue
    s0 = (Sf[k] // 4) * 4
    onL = None
    for L in [48, 54, 60, 66]:
        if not np.isnan(f[Li[L]]) and f[Li[L]] >= THR: onL = L; break
    if onL is None: continue                          # 예보가 AR 안 부름 -> FT 아님(FF), 제외
    traj = []; ok = True
    for off in [0, 6, 12, 18, 24]:
        L2 = onL + off
        if L2 not in Li or np.isnan(f[Li[L2]]): ok = False; break
        traj.append(float(f[Li[L2]]))
    if not ok: continue
    cont = [traj[1], traj[2], traj[3]]; fcv = min(cont)
    feat = traj + [min(cont), float(np.mean(cont)), float(np.std(cont)), traj[4] - traj[0]]
    j = s0 + (onL - 48) // 6
    if j + 3 >= T: continue
    oc = ivt[j + 1:j + 4]; y = int(np.all(oc >= THR)); omin = float(oc.min())
    rows.append((np.array(feat), fcv, y, omin, s0 // 4, "fa")); n_fa += 1

src = [r[5] for r in rows]; ys = np.array([r[2] for r in rows])
print(f"[CA 예보기준 분모 = TT+TF+FT]  총 {len(rows)}")
print(f"  TT+TF (관측온셋) = {n_obs} (지속 {n_pos_obs}) | FT (near-miss 예보온셋) = {n_fa}")
print(f"  지속 y=1: {int(ys.sum())} | 지속률(base rate): {ys.mean():.3f}   (기존 관측온셋만 = {n_pos_obs/n_obs:.3f})")

D2a = np.array([r[0] for r in rows]); fcv = np.array([r[1] for r in rows]); y = ys
omin = np.array([r[3] for r in rows]); oday = np.array([r[4] for r in rows])
o = np.argsort(oday); D2a, fcv, y, omin, oday = D2a[o], fcv[o], y[o], omin[o], oday[o]
np.savez("opdenom_ca.npz", D2=D2a, fcv=fcv, y=y, omin=omin, oday=oday, THR=THR)


def folds(N, od, Nf=5, emb=64):
    f = N // (Nf + 1); out = []
    for k in range(1, Nf + 1):
        ts = k * f; te = (k + 1) * f if k < Nf else N
        out.append((np.array([j for j in range(0, ts) if od[j] <= od[ts] - emb]), np.arange(ts, te)))
    return out


yt = []; pb = []
for tr, te in folds(len(y), oday):
    if len(tr) < 40 or len(np.unique(y[tr])) < 2: continue
    pb.extend((fcv[te] >= THR).astype(int)); yt.extend(y[te])
print(f"  [기준 raw] F1={f1_score(np.array(yt), np.array(pb), zero_division=0):.3f}  (TT+TF+FT 분모)  -> opdenom_ca.npz, THR={THR:.1f}")
