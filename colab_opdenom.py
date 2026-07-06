# ===== Colab: 0.5THR 완전분모(TT+TF+FT+FF)에서 모델 비교 — 3지역 × 지속 horizon 24h·30h =====
# 회귀 omin -> THR 판정 -> F1. 입력 opdenom_full_{r}.npz(24h) / opdenom_full_{r}_30h.npz(30h).
# 준비: opdenom_full_{ca,uk,chile}{,_30h}.npz (git pull). !pip install tabpfn tabicl lightgbm pytorch-tabnet -q
import numpy as np, torch, torch.nn as nn, os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMRegressor
from tabpfn import TabPFNRegressor
from tabicl import TabICLRegressor   # 완전 오픈 tabular foundation model (키 불필요, HF 자동 다운로드)
from pytorch_tabnet.tab_model import TabNetRegressor
DEV = "cuda" if torch.cuda.is_available() else "cpu"; NB = 2000; REGIONS = ["ca", "uk", "chile"]


def p_lr(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr); return LinearRegression().fit(sc.transform(Xtr), ytr).predict(sc.transform(Xte))
def p_lgbm(Xtr, ytr, Xte):
    return LGBMRegressor(subsample=0.8, verbose=-1, **LGBM_HP).fit(Xtr, ytr).predict(Xte)   # LGBM_HP = 블록0 튜닝값(아래)
def p_tabpfn(Xtr, ytr, Xte):
    m = TabPFNRegressor(device=DEV); m.fit(np.nan_to_num(Xtr), ytr); return np.asarray(m.predict(np.nan_to_num(Xte)))
def p_tabicl(Xtr, ytr, Xte):
    m = TabICLRegressor(); m.fit(np.nan_to_num(Xtr), ytr); return np.asarray(m.predict(np.nan_to_num(Xte)))
def p_tabnet(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr); bs = max(16, len(Xtr) // 4)
    m = TabNetRegressor(verbose=0, device_name=DEV, seed=0)
    m.fit(sc.transform(Xtr).astype("float32"), ytr.reshape(-1, 1).astype("float32"), max_epochs=150, batch_size=bs, virtual_batch_size=bs, drop_last=True)
    return m.predict(sc.transform(Xte).astype("float32")).ravel()
class LSTMreg(nn.Module):
    def __init__(s, h=16): super().__init__(); s.lstm = nn.LSTM(1, h, batch_first=True); s.fc = nn.Linear(h, 1)
    def forward(s, x): o, _ = s.lstm(x); return s.fc(o[:, -1]).squeeze(-1)
def p_lstm(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr[:, :5])
    xtr = torch.tensor(sc.transform(Xtr[:, :5])[:, :, None], dtype=torch.float32, device=DEV)
    xte = torch.tensor(sc.transform(Xte[:, :5])[:, :, None], dtype=torch.float32, device=DEV)
    ym, ys = float(ytr.mean()), float(ytr.std() + 1e-8); ytn = torch.tensor((ytr - ym) / ys, dtype=torch.float32, device=DEV)
    torch.manual_seed(0); net = LSTMreg().to(DEV); opt = torch.optim.Adam(net.parameters(), lr=1e-2); lf = nn.MSELoss()
    rng = np.random.default_rng(0); n = len(ytr)
    for ep in range(200):
        net.train(); perm = rng.permutation(n)
        for b in range(0, n, 128):
            bi = torch.as_tensor(perm[b:b + 128], device=DEV); opt.zero_grad(); lf(net(xtr[bi]), ytn[bi]).backward(); opt.step()
    net.eval()
    with torch.no_grad(): return net(xte).cpu().numpy() * ys + ym
MODELS = {"LR": p_lr, "LGBM": p_lgbm, "LSTM": p_lstm, "TabPFN": p_tabpfn, "TabICL": p_tabicl, "TabNet": p_tabnet}


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
def model_preds(pf, X, omin, y, od, thr):
    yt = []; pb = []; sc = []
    for tr, te in folds(len(y), od):
        if len(tr) < 40 or len(np.unique(y[tr])) < 2: continue
        pred = np.asarray(pf(X[tr], omin[tr], X[te]))
        pb.extend((pred >= thr).astype(int)); sc.extend(pred); yt.extend(y[te])
    return np.array(yt), np.array(pb), np.array(sc)
def boot(yt, pa, pb_):
    rng = np.random.default_rng(0); dd = []
    for _ in range(NB):
        ix = rng.integers(0, len(yt), len(yt))
        if len(np.unique(yt[ix])) > 1: dd.append(f1_score(yt[ix], pa[ix], zero_division=0) - f1_score(yt[ix], pb_[ix], zero_division=0))
    dd = np.array(dd); return np.percentile(dd, 2.5), np.percentile(dd, 97.5), np.mean(dd > 0)


# ===== LGBM HP 튜닝: 블록0(항상 train, test 절대 아님)에서만 선택 -> leak-free. 나머지 모델은 튜닝 대상 아님 =====
LGBM_GRID = [dict(num_leaves=nl, learning_rate=lr, n_estimators=ne, min_child_samples=mc)
             for nl in (15, 31) for lr in (0.03, 0.05) for ne in (200, 400) for mc in (20, 40)]
def tune_lgbm(regions, suf="", n_splits=3):
    # 블록0(항상 train, test 절대 아님) '전체'를 시간순 CV(expanding TimeSeriesSplit)로 사용 -> leak-free.
    # 선택지표 = omin 회귀 RMSE(THR 정규화). F1 대신 회귀오차인 이유: 블록0 양성이 극소(UK)라 F1 불안정/정의불가.
    #   블록0 전체 활용 + 여러 시간순 스플릿 평균 -> 단일 슬라이스 운(예: 특정 슬라이스 양성 0)에 안 흔들림. 풀링 없음(정규화 후 평균).
    packs = []
    for R in regions:
        d = np.load(f"opdenom_full_{R}{suf}.npz"); D2 = d["D2"]; omin = d["omin"]; THR = float(d["THR"]); N = len(d["y"])
        f = N // 6
        packs.append((D2[:f], omin[:f], THR))                      # 블록0 전체
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best = None; best_s = 1e18
    for hp in LGBM_GRID:
        es = []                                                    # 지역 × 시간순 스플릿별 정규화 RMSE
        for X0, o0, THR in packs:
            for tr, val in tscv.split(X0):
                m = LGBMRegressor(subsample=0.8, verbose=-1, **hp).fit(X0[tr], o0[tr])
                es.append(float(np.sqrt(np.mean((m.predict(X0[val]) - o0[val]) ** 2)) / THR))
        s = float(np.mean(es))
        if s < best_s: best_s = s; best = hp
    return best, best_s
LGBM_HP, _lgbm_dev = tune_lgbm(REGIONS)                             # 24h 블록0 시간순 CV로 HP 고정 -> 전 horizon 공통 적용
print(f"[LGBM HP: 블록0 시간순CV 튜닝(leak-free, 지표=정규화RMSE)] {LGBM_HP}  지역·스플릿평균 devRMSE/THR={_lgbm_dev:.3f}\n")

HORIZONS = [("_18h", "18h"), ("", "24h"), ("_30h", "30h")]   # 지속 임계 18/24/30h (24h=opdenom_full_{r}.npz)
for suf, hlab in HORIZONS:
    print(f"\n========== 지속 horizon {hlab} ==========")
    REG = {}
    for R in REGIONS:
        fn = f"opdenom_full_{R}{suf}.npz"
        if not os.path.exists(fn): print(f"[{R} {hlab}] {fn} 없음 -> skip"); continue
        d = np.load(fn); D2 = d["D2"]; fcv = d["fcv"]; y = d["y"]; omin = d["omin"]; oday = d["oday"]; THR = float(d["THR"])
        yt0, p0, s0 = raw_preds(fcv, y, oday, THR); f0 = f1_score(yt0, p0, zero_division=0)
        print(f"[{R} {hlab}] n={len(y)} 지속률={y.mean():.3f} THR={THR:.1f} | 기준 raw F1={f0:.3f} AUC={roc_auc_score(yt0, s0):.3f}")
        REG[R] = {"_raw": (yt0, p0)}
        for name, pf in MODELS.items():
            yt, pb, sc = model_preds(pf, D2, omin, y, oday, THR)
            lo, hi, pp = boot(yt0, pb, p0)
            print(f"  {name:7s}: F1={f1_score(yt, pb, zero_division=0):.3f} AUC={roc_auc_score(yt, sc):.3f} AUPRC={average_precision_score(yt, sc):.3f}  ΔF1={f1_score(yt, pb, zero_division=0)-f0:+.3f} CI[{lo:+.3f},{hi:+.3f}] P={pp:.3f}")
            REG[R][name] = (yt, pb)

    # ===== 통합검정: 모델별 평균 ΔF1(지역 부트스트랩) =====
    print(f"[통합 ΔF1(vs raw), 지역 평균 — {hlab}]")
    for name in MODELS:
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
        print(f"  {name:7s} 평균ΔF1={obs:+.3f}  95%CI[{np.percentile(md,2.5):+.3f},{np.percentile(md,97.5):+.3f}]  P(Δ>0)={np.mean(md>0):.3f}")
