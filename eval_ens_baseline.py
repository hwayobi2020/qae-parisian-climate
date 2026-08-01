"""앙상블(5멤버) 기준선 평가 — 원예보(c00) 대비.

세 가지 파라미터 없는 판정을 같은 후보일·같은 임계값으로 비교한다.
  (1) c00 단일   : fcv_c00 >= THR                     (본 연구의 원예보 기준선)
  (2) 멤버 투표  : 유효 멤버 중 fcv_m >= THR 인 비율이 과반이면 지속
  (3) 앙상블 평균: 5멤버 고정리드 궤적을 평균한 뒤 지속창 최소값 >= THR

평가 조건은 본 실험과 동일: 워크포워드 5폴드(64일 임베고) test 풀링, F1.
유의성은 사례 재표집 부트스트랩(2,000회) — 지역별 및 2지역 평균.
사용: python eval_ens_baseline.py
"""
import numpy as np
from sklearn.metrics import f1_score

REGIONS = ["ca", "chile"]
HORIZONS = [("_18h", "18h"), ("", "24h"), ("_30h", "30h")]
NB = 2000


def folds(N, od, Nf=5, emb=64):
    f = N // (Nf + 1); out = []
    for k in range(1, Nf + 1):
        ts = k * f; te = (k + 1) * f if k < Nf else N
        out.append((np.array([j for j in range(0, ts) if od[j] <= od[ts] - emb]), np.arange(ts, te)))
    return out


def boot_ci(yt, pa, pb):
    rng = np.random.default_rng(0); dd = []
    for _ in range(NB):
        ix = rng.integers(0, len(yt), len(yt))
        if len(np.unique(yt[ix])) > 1:
            dd.append(f1_score(yt[ix], pa[ix], zero_division=0) - f1_score(yt[ix], pb[ix], zero_division=0))
    dd = np.array(dd); return np.percentile(dd, 2.5), np.percentile(dd, 97.5), np.mean(dd > 0)


def boot_pooled(REG, ka, kb):
    rng = np.random.default_rng(0); md = []
    for _ in range(NB):
        ds = []
        for R in REG:
            yt = REG[R]["_yt"]; ix = rng.integers(0, len(yt), len(yt))
            if len(np.unique(yt[ix])) < 2: ds = None; break
            ds.append(f1_score(yt[ix], REG[R][ka][ix], zero_division=0)
                      - f1_score(yt[ix], REG[R][kb][ix], zero_division=0))
        if ds is not None: md.append(np.mean(ds))
    md = np.array(md)
    obs = np.mean([f1_score(REG[R]["_yt"], REG[R][ka], zero_division=0)
                   - f1_score(REG[R]["_yt"], REG[R][kb], zero_division=0) for R in REG])
    return obs, np.percentile(md, 2.5), np.percentile(md, 97.5), np.mean(md > 0)


NAMES = ["c00 단일", "멤버 투표", "앙상블 평균"]

for suf, hlab in HORIZONS:
    print(f"\n{'='*66}\n지속 기준 {hlab} — 앙상블 기준선 비교 (파라미터 없는 판정)\n{'='*66}")
    REG = {}
    for R in REGIONS:
        d = np.load(f"opdenom_full_{R}{suf}.npz")
        e = np.load(f"ens_fcv_{R}{suf}.npz")
        y = d["y"]; oday = d["oday"]; THR = float(d["THR"])
        fcv_mem = e["fcv_mem"]; fcv_avg = e["fcv_meantraj"]
        assert np.allclose(d["fcv"], fcv_mem[:, 0], atol=1e-9), f"{R}{suf} c00 정렬 불일치"

        # 판정 3종 (전 행에 대해 먼저 계산)
        j_c00 = (fcv_mem[:, 0] >= THR).astype(int)
        valid = ~np.isnan(fcv_mem)
        exceed = np.where(valid, fcv_mem >= THR, False).sum(axis=1)
        nvalid = valid.sum(axis=1)
        j_vote = (exceed * 2 > nvalid).astype(int)          # 유효 멤버 중 과반
        j_avg = np.where(np.isnan(fcv_avg), j_c00, (fcv_avg >= THR).astype(int))

        FL = [(tr, te) for tr, te in folds(len(y), oday) if len(tr) >= 40 and len(np.unique(y[tr])) > 1]
        te_all = np.concatenate([te for _, te in FL])
        yt = y[te_all]
        REG[R] = {"_yt": yt, "c00 단일": j_c00[te_all], "멤버 투표": j_vote[te_all], "앙상블 평균": j_avg[te_all]}

        print(f"[{R}] n={len(yt)} 지속={int(yt.sum())} THR={THR:.1f}")
        for nm in NAMES:
            f1 = f1_score(yt, REG[R][nm], zero_division=0)
            line = f"  {nm:10s}: F1={f1:.3f}"
            if nm != "c00 단일":
                lo, hi, pp = boot_ci(yt, REG[R][nm], REG[R]["c00 단일"])
                line += f"  Δ(vs c00)={f1 - f1_score(yt, REG[R]['c00 단일'], zero_division=0):+.3f} CI[{lo:+.3f},{hi:+.3f}] P={pp:.3f}"
            print(line)

    print(f"[통합 2지역 평균 — vs c00 단일]")
    for nm in NAMES[1:]:
        obs, lo, hi, pp = boot_pooled(REG, nm, "c00 단일")
        print(f"  {nm:10s} 평균ΔF1={obs:+.3f}  95%CI[{lo:+.3f},{hi:+.3f}]  P(Δ>0)={pp:.3f}")
