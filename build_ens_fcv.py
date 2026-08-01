"""앙상블(5멤버) 기준선용 지속창 최소 예보값(fcv) 산출.

각 멤버(c00, p01~p04)가 **자기 궤적의 peak을 기준으로 독립적으로** 지속창을 잡고
그 구간의 최소 예보 IVT(fcv)를 계산한다. 즉 "멤버들이 각자 예보하고 각자 판정한다"는 구성.

후보일 집합과 정렬 순서는 build_op_denom_full.py 와 동일하게 재현하며,
저장 직후 c00 열이 기존 opdenom_full_{r}{suf}.npz 의 fcv 와 일치하는지 검증한다.

저장: ens_fcv_{region}{suf}.npz  { fcv_mem (N,5), fcv_meantraj (N,), D8_mem (N,5,8), members, oday }
사용: python build_ens_fcv.py
"""
import numpy as np, os

IVTF = {"ca": "ivt_sf_1980_2023.npy", "chile": "ivt_chile_1980_2023.npy"}
LEADS = [48, 54, 60, 66, 72, 78, 84, 90]
Li = {L: i for i, L in enumerate(LEADS)}
HORIZONS = [18, 24, 30]
MEMBERS = ["c00", "p01", "p02", "p03", "p04"]


def find(n):
    for p in ["data/raw/" + n, n]:
        if os.path.exists(p): return p
    raise FileNotFoundError(n)


def peak_traj(f):
    """멤버 궤적 f(8개 고정리드)에서 예보 peak 기준 궤적 5점(peak,+6,+12,+18,+24) 반환.
    build_op_denom_full.py 의 peak_feats 와 동일한 규칙."""
    day = [(L, f[Li[L]]) for L in [48, 54, 60, 66] if not np.isnan(f[Li[L]])]
    if not day:
        return None
    onL = max(day, key=lambda x: x[1])[0]
    if any((onL + o) not in Li or np.isnan(f[Li[onL + o]]) for o in [6, 12, 18, 24]):
        return None
    return [float(f[Li[onL + o]]) for o in [0, 6, 12, 18, 24]]


for R in IVTF:
    # ---- 5멤버 예보 로드 (c00 + p01~p04), s 인덱스 정렬 검증 ----
    z_on_c = np.load(f"gefs_ivt_{R}_d2c00.npz")
    z_on_p = np.load(f"gefs_ivt_{R}_d2p.npz")
    z_nm_c = np.load(f"gefs_ivt_{R}_d2c00fa.npz")
    z_nm_p = np.load(f"gefs_ivt_{R}_d2pfa.npz")
    assert np.array_equal(z_on_c["s"], z_on_p["s"]), f"{R} 온셋 s 인덱스 불일치"
    assert np.array_equal(z_nm_c["s"], z_nm_p["s"]), f"{R} near-miss s 인덱스 불일치"

    IV_on = np.concatenate([z_on_c["ivt"], z_on_p["ivt"]], axis=1)   # (N_on, 5, 8)
    IV_nm = np.concatenate([z_nm_c["ivt"], z_nm_p["ivt"]], axis=1)   # (N_nm, 5, 8)
    S_on = z_on_c["s"].astype(int)
    S_nm = z_nm_c["s"].astype(int)

    ivt = np.load(find(IVTF[R])).astype("float64")
    T = len(ivt)

    # ---- 후보 수집: build_op_denom_full.py 와 동일한 순서·필터(c00 기준) ----
    cand = []            # (멤버별 궤적 리스트, base 인덱스, oday)
    for k in range(len(S_on)):
        f_c00 = IV_on[k, 0]
        if np.isnan(f_c00).all():
            continue
        if peak_traj(f_c00) is None:
            continue
        trajs = [peak_traj(IV_on[k, m]) for m in range(5)]
        traj_mean = peak_traj(np.nanmean(IV_on[k], axis=0))   # 멤버 평균 궤적 -> peak 정렬
        s = int(S_on[k])
        cand.append((trajs, traj_mean, np.asarray(IV_on[k], float), s, s // 4))
    for k in range(len(S_nm)):
        f_c00 = IV_nm[k, 0]
        if np.isnan(f_c00).all():
            continue
        if peak_traj(f_c00) is None:
            continue
        trajs = [peak_traj(IV_nm[k, m]) for m in range(5)]
        traj_mean = peak_traj(np.nanmean(IV_nm[k], axis=0))
        s0 = (int(S_nm[k]) // 4) * 4
        cand.append((trajs, traj_mean, np.asarray(IV_nm[k], float), s0, s0 // 4))

    # ---- horizon 별 멤버 fcv 산출 ----
    for H in HORIZONS:
        npts = H // 6
        rows = []
        for trajs, traj_mean, d8m, base, oday in cand:
            if base + npts > T:
                continue
            fcv_m = []
            for tr in trajs:
                fcv_m.append(np.nan if tr is None else float(min(tr[1:npts])))
            fcv_avg = np.nan if traj_mean is None else float(min(traj_mean[1:npts]))
            rows.append((fcv_m, fcv_avg, d8m, oday))
        fcv_mem = np.array([r[0] for r in rows], float)
        fcv_meantraj = np.array([r[1] for r in rows], float)
        D8_mem = np.array([r[2] for r in rows], float)      # (N, 5, 8) 멤버별 고정리드 원값
        oday = np.array([r[3] for r in rows])
        o = np.argsort(oday)
        fcv_mem, fcv_meantraj, D8_mem, oday = fcv_mem[o], fcv_meantraj[o], D8_mem[o], oday[o]

        suf = "" if H == 24 else f"_{H}h"
        out = f"ens_fcv_{R}{suf}.npz"
        np.savez(out, fcv_mem=fcv_mem, fcv_meantraj=fcv_meantraj, D8_mem=D8_mem, members=np.array(MEMBERS), oday=oday)

        # ---- 정렬 검증: c00 열이 기존 fcv 와 일치해야 함 ----
        ref = np.load(f"opdenom_full_{R}{suf}.npz")
        ok_shape = len(ref["fcv"]) == len(fcv_mem)
        ok_c00 = ok_shape and np.allclose(ref["fcv"], fcv_mem[:, 0], rtol=0, atol=1e-9)
        n_missing = int(np.isnan(fcv_mem[:, 1:]).any(axis=1).sum())
        print(f"[{R} {H}h] n={len(fcv_mem)} | c00 일치={ok_c00} | 섭동멤버 결측 있는 행={n_missing} -> {out}")
        if not ok_c00:
            print(f"   !! 정렬 불일치: ref n={len(ref['fcv'])}, 산출 n={len(fcv_mem)}")