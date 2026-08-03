# ===== Colab: 18h 블록 빈칸 채우기 — TabICL·TabNet =====
# 목적: 표 9(모델별 통합 ΔF1, 지속 기준별)의 18시간 열에서 비어 있는 두 모델을 채운다.
#       나머지 5개 모델(LR/LGBM/LSTM/1d-CNN/TabPFN)의 18h 값은 이미 산출됨.
# 조건은 기존 실행과 동일: 피처 B(고정리드 8), 회귀 타깃 omin, >=THR 판정, F1,
#       워크포워드 5폴드(64일 임베고), 부트스트랩 2,000회, 2지역(CA·Chile).
# 준비: opdenom_full_{ca,chile}_18h.npz (git pull). !pip install tabicl pytorch-tabnet -q
import warnings, logging
warnings.filterwarnings("ignore"); logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
import numpy as np, torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from tabicl import TabICLRegressor
from pytorch_tabnet.tab_model import TabNetRegressor
DEV = "cuda" if torch.cuda.is_available() else "cpu"; NB = 2000; REGIONS = ["ca", "chile"]


def p_tabicl(Xtr, ytr, Xte):
    m = TabICLRegressor(); m.fit(np.nan_to_num(Xtr), ytr); return np.asarray(m.predict(np.nan_to_num(Xte)))


def p_tabnet(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr); bs = max(16, len(Xtr) // 4)
    m = TabNetRegressor(verbose=0, device_name=DEV, seed=0)
    m.fit(sc.transform(Xtr).astype("float32"), ytr.reshape(-1, 1).astype("float32"),
          max_epochs=150, batch_size=bs, virtual_batch_size=bs, drop_last=True)
    return m.predict(sc.transform(Xte).astype("float32")).ravel()


MODELS = {"TabICL": p_tabicl, "TabNet": p_tabnet}


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


def boot_pooled(REG, name):
    rng = np.random.default_rng(0); md = []
    for _ in range(NB):
        ds = []
        for R in REG:
            yt = REG[R]["_yt"]; ix = rng.integers(0, len(yt), len(yt))
            if len(np.unique(yt[ix])) < 2: ds = None; break
            ds.append(f1_score(yt[ix], REG[R][name][ix], zero_division=0)
                      - f1_score(yt[ix], REG[R]["_raw"][ix], zero_division=0))
        if ds is not None: md.append(np.mean(ds))
    md = np.array(md)
    obs = np.mean([f1_score(REG[R]["_yt"], REG[R][name], zero_division=0)
                   - f1_score(REG[R]["_yt"], REG[R]["_raw"], zero_division=0) for R in REG])
    return obs, np.percentile(md, 2.5), np.percentile(md, 97.5), np.mean(md > 0)


print("=" * 66)
print("지속 기준 18h — TabICL·TabNet (피처 B, 2지역)")
print("=" * 66)
REG = {}
for R in REGIONS:
    d = np.load(f"opdenom_full_{R}_18h.npz")
    D8 = d["D8"]; fcv = d["fcv"]; y = d["y"]; omin = d["omin"]; oday = d["oday"]; THR = float(d["THR"])
    FL = [(tr, te) for tr, te in folds(len(y), oday) if len(tr) >= 40 and len(np.unique(y[tr])) > 1]
    yt = np.concatenate([y[te] for _, te in FL])
    p0 = np.concatenate([(fcv[te] >= THR).astype(int) for _, te in FL])
    f0 = f1_score(yt, p0, zero_division=0)
    print(f"[{R}] n={len(y)} 지속={int(y.sum())} THR={THR:.1f} | raw F1={f0:.3f}")
    REG[R] = {"_yt": yt, "_raw": p0}
    for name, pf in MODELS.items():
        pb = np.concatenate([(np.asarray(pf(D8[tr], omin[tr], D8[te])) >= THR).astype(int) for tr, te in FL])
        ff = f1_score(yt, pb, zero_division=0); lo, hi, pp = boot_ci(yt, pb, p0)
        print(f"  {name:7s}: F1={ff:.3f}  ΔF1={ff - f0:+.3f} CI[{lo:+.3f},{hi:+.3f}] P={pp:.3f}")
        REG[R][name] = pb

print("[통합 (2지역 평균 ΔF1) — 18h]")
for name in MODELS:
    obs, lo, hi, pp = boot_pooled(REG, name)
    print(f"  {name:7s} 평균ΔF1={obs:+.3f}  95%CI[{lo:+.3f},{hi:+.3f}]  P(Δ>0)={pp:.3f}")