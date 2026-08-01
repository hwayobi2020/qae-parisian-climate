# ===== Colab: 앙상블을 '모델 입력'으로 — 5멤버 정보가 지속 예측을 개선하는가 =====
# 질문: 컨트롤 1멤버(8피처) 대신 5멤버 정보를 넣으면 나아지는가?
#   B        = c00 고정리드 8 (본 연구 메인)
#   B_all    = 5멤버 × 8리드 = 40 (전 멤버 원값)
#   B_spread = c00 8 + 멤버평균 8 + 멤버표준편차 8 = 24 (앙상블 요약: 중심 + 산포)
# 판정·타깃·폴드는 본 실험과 동일. 유의성: 부트스트랩 2,000회 (vs 원예보, vs B).
# 준비: opdenom_full_{ca,chile}{,_18h,_30h}.npz, ens_fcv_{ca,chile}{,_18h,_30h}.npz (git pull)
#       !pip install tabpfn lightgbm -q
import warnings, logging
warnings.filterwarnings("ignore"); logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
import numpy as np, torch
from sklearn.metrics import f1_score
from lightgbm import LGBMRegressor
from tabpfn import TabPFNRegressor
DEV = "cuda" if torch.cuda.is_available() else "cpu"; NB = 2000
REGIONS = ["ca", "chile"]; HORIZONS = [("_18h", "18h"), ("", "24h"), ("_30h", "30h")]
LGBM_HP = dict(num_leaves=15, learning_rate=0.03, n_estimators=200, min_child_samples=20)


def p_lgbm(Xtr, ytr, Xte):
    return LGBMRegressor(subsample=0.8, verbose=-1, **LGBM_HP).fit(np.nan_to_num(Xtr), ytr).predict(np.nan_to_num(Xte))
def p_tabpfn(Xtr, ytr, Xte):
    m = TabPFNRegressor(device=DEV); m.fit(np.nan_to_num(Xtr), ytr); return np.asarray(m.predict(np.nan_to_num(Xte)))


def folds(N, od, Nf=5, emb=64):
    f = N // (Nf + 1); out = []
    for k in range(1, Nf + 1):
        ts = k * f; te = (k + 1) * f if k < Nf else N
        out.append((np.array([j for j in range(0, ts) if od[j] <= od[ts] - emb]), np.arange(ts, te)))
    return out


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


SETS = ["B", "B_all(40)", "B_spread(24)"]
for MNAME, PF in [("TabPFN", p_tabpfn), ("LGBM", p_lgbm)]:
    for suf, hlab in HORIZONS:
        print(f"\n===== {MNAME} / {hlab} : 앙상블 입력 비교 =====")
        REG = {}
        for R in REGIONS:
            d = np.load(f"opdenom_full_{R}{suf}.npz"); e = np.load(f"ens_fcv_{R}{suf}.npz")
            D8 = d["D8"]; fcv = d["fcv"]; y = d["y"]; omin = d["omin"]; oday = d["oday"]; THR = float(d["THR"])
            D8m = e["D8_mem"]                                    # (N, 5, 8)
            assert np.allclose(D8m[:, 0, :], D8, equal_nan=True), f"{R}{suf} c00 정렬 불일치"
            X = {"B": D8,
                 "B_all(40)": D8m.reshape(len(D8m), -1),
                 "B_spread(24)": np.column_stack([D8, np.nanmean(D8m, axis=1), np.nanstd(D8m, axis=1)])}
            FL = [(tr, te) for tr, te in folds(len(y), oday) if len(tr) >= 40 and len(np.unique(y[tr])) > 1]
            yt = np.concatenate([y[te] for _, te in FL])
            p0 = np.concatenate([(fcv[te] >= THR).astype(int) for _, te in FL])
            f0 = f1_score(yt, p0, zero_division=0)
            REG[R] = {"_yt": yt, "_raw": p0}
            line = f"  [{R}] raw={f0:.3f}"
            for sn in SETS:
                Z = X[sn]
                pb = np.concatenate([(np.asarray(PF(Z[tr], omin[tr], Z[te])) >= THR).astype(int) for tr, te in FL])
                REG[R][sn] = pb
                line += f" | {sn}={f1_score(yt, pb, zero_division=0):.3f}"
            print(line)
        print("  [통합 vs 원예보]", end=" ")
        for sn in SETS:
            obs, lo, hi, pp = boot_pooled(REG, sn, "_raw")
            print(f"{sn}={obs:+.3f}[{lo:+.3f},{hi:+.3f}]P{pp:.2f}", end="  ")
        print("\n  [통합 vs B (앙상블 추가 효과)]", end=" ")
        for sn in SETS[1:]:
            obs, lo, hi, pp = boot_pooled(REG, sn, "B")
            print(f"{sn}={obs:+.3f}[{lo:+.3f},{hi:+.3f}]P{pp:.2f}", end="  ")
        print()