# ===== Colab: 0.5THR 완전분모 — 예보9 에 IVT 16일 웨이블릿 6개 직접 추가 (인코더 없음) =====
# 질문: D-2 예보9개 위에 IVT 저주파 배경(16일 웨이블릿 6개)을 얹으면 raw/예보9 대비 F1 오르나.
# 근거: D-2 예보(리드 48-90h)는 초기조건에 현재 기상상태가 녹아 있어 env 44개 대부분이 예보와 중복->희석
#   (인코더 44->8 은 전 지역 F1 폭락). IVT 16일 웨이블릿만 예보 창(2-4일) 밖의 저주파 배경 = 예보에 없는 잔여정보 후보.
# 인코더 폐기: 6개는 이미 작아 압축 불필요, 예보와 같은 IVT 스케일이라 raw 로 직접 투입.
# 누수방지: TabPFN 매 폴드 train 에서만 학습 -> test 평가 (walk-forward 5폴드).
# 준비: opdenom_full_{ca,uk,chile}.npz (git pull; ENV 포함). !pip install tabpfn -q
import numpy as np, torch, os
from tabpfn import TabPFNRegressor
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
DEV = "cuda" if torch.cuda.is_available() else "cpu"; NB = 2000; REGIONS = ["ca", "uk", "chile"]
IVW = slice(0, 6)   # ENV 앞 6열 = IVT 16일 웨이블릿 (build_op_denom_full.py env_feats 순서: ivw 가 맨 앞)


def folds(N, od, Nf=5, emb=64):
    f = N // (Nf + 1); out = []
    for k in range(1, Nf + 1):
        ts = k * f; te = (k + 1) * f if k < Nf else N
        out.append((np.array([j for j in range(0, ts) if od[j] <= od[ts] - emb]), np.arange(ts, te)))
    return out


def raw_preds(fcv, y, od, thr):
    yt = []; pb = []; sc = []
    for tr, te in folds(len(y), od):
        if len(tr) < 40 or len(np.unique(y[tr])) < 2: continue
        pb.extend((fcv[te] >= thr).astype(int)); sc.extend(fcv[te]); yt.extend(y[te])
    return np.array(yt), np.array(pb), np.array(sc)


def reg_preds(X, omin, y, od, thr):                       # TabPFN 회귀: X[tr]->omin 학습, X[te] 예측 -> THR 판정
    yt = []; pb = []; sc = []
    for tr, te in folds(len(y), od):
        if len(tr) < 40 or len(np.unique(y[tr])) < 2: continue
        m = TabPFNRegressor(device=DEV); m.fit(np.nan_to_num(X[tr]), omin[tr])
        pred = np.asarray(m.predict(np.nan_to_num(X[te])))
        pb.extend((pred >= thr).astype(int)); sc.extend(pred); yt.extend(y[te])
    return np.array(yt), np.array(pb), np.array(sc)


def boot(yt, pa, pb_):                                    # ΔF1 (모델 - 기준)
    rng = np.random.default_rng(0); dd = []
    for _ in range(NB):
        ix = rng.integers(0, len(yt), len(yt))
        if len(np.unique(yt[ix])) > 1: dd.append(f1_score(yt[ix], pa[ix], zero_division=0) - f1_score(yt[ix], pb_[ix], zero_division=0))
    dd = np.array(dd); return np.percentile(dd, 2.5), np.percentile(dd, 97.5), np.mean(dd > 0)


REG = {}
for R in REGIONS:
    if not os.path.exists(f"opdenom_full_{R}.npz"): print(f"[{R}] opdenom_full 없음 -> skip"); continue
    d = np.load(f"opdenom_full_{R}.npz")
    D2 = d["D2"]; ENV = d["ENV"]; fcv = d["fcv"]; y = d["y"]; omin = d["omin"]; oday = d["oday"]; THR = float(d["THR"])
    Xivw = np.column_stack([D2, ENV[:, IVW]])             # 예보9 + IVT웨이블릿6 = 15
    yt0, p0, s0 = raw_preds(fcv, y, oday, THR); f0 = f1_score(yt0, p0, zero_division=0)
    yt1, p1, s1 = reg_preds(D2, omin, y, oday, THR); f1v = f1_score(yt1, p1, zero_division=0)
    yt2, p2, s2 = reg_preds(Xivw, omin, y, oday, THR); f2 = f1_score(yt2, p2, zero_division=0)
    lo1, hi1, pp1 = boot(yt0, p1, p0); lo2, hi2, pp2 = boot(yt0, p2, p0)
    print(f"[{R}] n={len(y)} 지속률={y.mean():.3f} THR={THR:.1f}")
    print(f"  기준 raw          : F1={f0:.3f} AUC={roc_auc_score(yt0, s0):.3f}")
    print(f"  예보9(TabPFN)     : F1={f1v:.3f} AUC={roc_auc_score(yt1, s1):.3f} AUPRC={average_precision_score(yt1, s1):.3f}  ΔF1={f1v - f0:+.3f} CI[{lo1:+.3f},{hi1:+.3f}] P={pp1:.3f}")
    print(f"  예보9+IVT웨이블릿6: F1={f2:.3f} AUC={roc_auc_score(yt2, s2):.3f} AUPRC={average_precision_score(yt2, s2):.3f}  ΔF1={f2 - f0:+.3f} CI[{lo2:+.3f},{hi2:+.3f}] P={pp2:.3f}")
    REG[R] = {"_raw": (yt0, p0), "예보9": (yt1, p1), "예보9+ivw": (yt2, p2)}

# ===== 통합검정: 예보9 / 예보9+IVT웨이블릿 각각 vs raw, 3지역 평균 ΔF1 (지역 부트스트랩) =====
print("\n[통합 ΔF1(vs raw), 지역 평균]")
for name in ["예보9", "예보9+ivw"]:
    rng = np.random.default_rng(0); md = []
    for _ in range(NB):
        ds = []
        for R in REG:
            yt0, p0 = REG[R]["_raw"]; _, pm = REG[R][name]; ix = rng.integers(0, len(yt0), len(yt0))
            if len(np.unique(yt0[ix])) < 2: ds = None; break
            ds.append(f1_score(yt0[ix], pm[ix], zero_division=0) - f1_score(yt0[ix], p0[ix], zero_division=0))
        if ds is not None: md.append(np.mean(ds))
    md = np.array(md)
    obs = np.mean([f1_score(REG[R]["_raw"][0], REG[R][name][1], zero_division=0) - f1_score(REG[R]["_raw"][0], REG[R]["_raw"][1], zero_division=0) for R in REG])
    print(f"  {name:10s} 평균ΔF1={obs:+.3f}  95%CI[{np.percentile(md,2.5):+.3f},{np.percentile(md,97.5):+.3f}]  P(Δ>0)={np.mean(md>0):.3f}")
