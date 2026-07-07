# ===== Colab: 피처 "정렬" ablation — peak정렬 vs 고정리드 =====
# 질문: 예보 peak 정렬(현재)이 고정 리드보다 나은가? 아니면 8개뿐이라 차이 없나?
# 피처셋: A_peak9 = D2(현재 peak정렬 9) / B_fix8 = D8(리드48~90 고정 8) / C_fix8+요약 = D8 + D2의 요약4(min/mean/std/기울기)
# raw 기준선 = fcv(peak정렬) 고정. 모델 = TabPFN·LGBM 회귀 -> omin -> THR -> F1. 3지역 × 18/24/30h + 통합.
# 누수방지: 매 폴드 train만 학습. 준비: opdenom_full_{r}{,_18h,_30h}.npz (git pull; D8 포함). !pip install tabpfn lightgbm -q
import numpy as np, torch, os
from tabpfn import TabPFNRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import f1_score
DEV = "cuda" if torch.cuda.is_available() else "cpu"; NB = 2000; REGIONS = ["ca", "uk", "chile"]
HORIZONS = [("_18h", "18h"), ("", "24h"), ("_30h", "30h")]


def folds(N, od, Nf=5, emb=64):
    f = N // (Nf + 1); out = []
    for k in range(1, Nf + 1):
        ts = k * f; te = (k + 1) * f if k < Nf else N
        out.append((np.array([j for j in range(0, ts) if od[j] <= od[ts] - emb]), np.arange(ts, te)))
    return out
def p_tabpfn(Xtr, ytr, Xte):
    m = TabPFNRegressor(device=DEV); m.fit(np.nan_to_num(Xtr), ytr); return np.asarray(m.predict(np.nan_to_num(Xte)))
def p_lgbm(Xtr, ytr, Xte):
    return LGBMRegressor(n_estimators=200, learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8, verbose=-1).fit(np.nan_to_num(Xtr), ytr).predict(np.nan_to_num(Xte))
MODELS = {"TabPFN": p_tabpfn, "LGBM": p_lgbm}


def raw_preds(fcv, y, od, thr):
    yt = []; pb = []
    for tr, te in folds(len(y), od):
        if len(tr) < 40 or len(np.unique(y[tr])) < 2: continue
        pb.extend((fcv[te] >= thr).astype(int)); yt.extend(y[te])
    return np.array(yt), np.array(pb)
def reg_preds(X, omin, y, od, thr, pf):
    yt = []; pb = []
    for tr, te in folds(len(y), od):
        if len(tr) < 40 or len(np.unique(y[tr])) < 2: continue
        pred = np.asarray(pf(X[tr], omin[tr], X[te])); pb.extend((pred >= thr).astype(int)); yt.extend(y[te])
    return np.array(yt), np.array(pb)
def boot(yt, pa, pb_):
    rng = np.random.default_rng(0); dd = []
    for _ in range(NB):
        ix = rng.integers(0, len(yt), len(yt))
        if len(np.unique(yt[ix])) > 1: dd.append(f1_score(yt[ix], pa[ix], zero_division=0) - f1_score(yt[ix], pb_[ix], zero_division=0))
    return np.mean(np.array(dd) > 0)


SETNAMES = ["A_peak9", "B_fix8", "C_fix8+요약"]
for suf, hlab in HORIZONS:
    print(f"\n========== {hlab} : 피처 정렬 ablation (raw=peak-fcv 고정) ==========")
    for mname, pf in MODELS.items():
        print(f"  --- {mname} ---")
        REG = {}
        for R in REGIONS:
            d = np.load(f"opdenom_full_{R}{suf}.npz")
            D2 = d["D2"]; D8 = d["D8"]; fcv = d["fcv"]; y = d["y"]; omin = d["omin"]; oday = d["oday"]; THR = float(d["THR"])
            SETS = {"A_peak9": D2, "B_fix8": D8, "C_fix8+요약": np.column_stack([D8, D2[:, 5:9]])}
            yt0, p0 = raw_preds(fcv, y, oday, THR); f0 = f1_score(yt0, p0, zero_division=0)
            REG[R] = {"_raw": (yt0, p0)}; line = f"    [{R}] raw={f0:.3f}"
            for sn in SETNAMES:
                yt, pb = reg_preds(SETS[sn], omin, y, oday, THR, pf); ff = f1_score(yt, pb, zero_division=0)
                line += f" | {sn}={ff:.3f}(Δ{ff - f0:+.3f},P{boot(yt0, pb, p0):.2f})"
                REG[R][sn] = (yt, pb)
            print(line)
        for sn in SETNAMES:                                    # 통합(지역평균 ΔF1)
            rng = np.random.default_rng(0); md = []
            for _ in range(NB):
                ds = []
                for R in REG:
                    yt0, p0 = REG[R]["_raw"]; _, pm = REG[R][sn]; ix = rng.integers(0, len(yt0), len(yt0))
                    if len(np.unique(yt0[ix])) < 2: ds = None; break
                    ds.append(f1_score(yt0[ix], pm[ix], zero_division=0) - f1_score(yt0[ix], p0[ix], zero_division=0))
                if ds is not None: md.append(np.mean(ds))
            md = np.array(md); obs = np.mean([f1_score(REG[R]["_raw"][0], REG[R][sn][1], zero_division=0) - f1_score(REG[R]["_raw"][0], REG[R]["_raw"][1], zero_division=0) for R in REG])
            print(f"    [통합 {sn}] 평균Δ{obs:+.3f} CI[{np.percentile(md,2.5):+.3f},{np.percentile(md,97.5):+.3f}] P{np.mean(md>0):.3f}")
