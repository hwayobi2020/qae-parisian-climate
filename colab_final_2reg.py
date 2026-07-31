# ===== Colab: 최종 2지역(CA·Chile) 헤드라인 — B피처, 7모델, 18/24/30h =====
# 목적: (1) 논문 헤드라인 수치 확정 — 2지역 통합 ΔF1의 CI·P (초록 +0.137 뒷받침)
#       (2) 1d-CNN 추가 (초록 비교모델), LSTM과 동일한 학습 조건으로 공정하게.
# 피처 = B(고정리드 8: 발표+48~90h 예보 IVT 원값). 회귀 타깃 = 관측 지속창 최소 IVT(omin) -> >=THR 판정 -> F1.
# 기준선 = raw 예보(fcv >= THR). 검증 = 워크포워드 5폴드(64일 임베고) + 부트스트랩(지역별/2지역 통합).
# 준비: opdenom_full_{ca,chile}{,_18h,_30h}.npz (git pull). !pip install tabpfn tabicl lightgbm pytorch-tabnet -q
import warnings, logging, os
warnings.filterwarnings("ignore"); logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
os.environ.setdefault("PYTHONWARNINGS", "ignore")
import numpy as np, torch, torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMRegressor
from tabpfn import TabPFNRegressor
from tabicl import TabICLRegressor
from pytorch_tabnet.tab_model import TabNetRegressor
DEV = "cuda" if torch.cuda.is_available() else "cpu"; NB = 2000; REGIONS = ["ca", "chile"]
HORIZONS = [("_18h", "18h"), ("", "24h"), ("_30h", "30h")]


# ---------- 모델 ----------
def p_lr(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr); return LinearRegression().fit(sc.transform(Xtr), ytr).predict(sc.transform(Xte))
def p_lgbm(Xtr, ytr, Xte):
    return LGBMRegressor(subsample=0.8, verbose=-1, **LGBM_HP).fit(Xtr, ytr).predict(Xte)
def p_tabpfn(Xtr, ytr, Xte):
    m = TabPFNRegressor(device=DEV); m.fit(np.nan_to_num(Xtr), ytr); return np.asarray(m.predict(np.nan_to_num(Xte)))
def p_tabicl(Xtr, ytr, Xte):
    m = TabICLRegressor(); m.fit(np.nan_to_num(Xtr), ytr); return np.asarray(m.predict(np.nan_to_num(Xte)))
def p_tabnet(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr); bs = max(16, len(Xtr) // 4)
    m = TabNetRegressor(verbose=0, device_name=DEV, seed=0)
    m.fit(sc.transform(Xtr).astype("float32"), ytr.reshape(-1, 1).astype("float32"), max_epochs=150, batch_size=bs, virtual_batch_size=bs, drop_last=True)
    return m.predict(sc.transform(Xte).astype("float32")).ravel()


def _train_torch(net, xtr, ytr, xte):
    """LSTM/CNN 공통 학습 루틴 — 동일 조건(Adam 1e-2, 200ep, batch128, seed0, 타깃 표준화)."""
    ym, ys = float(ytr.mean()), float(ytr.std() + 1e-8)
    ytn = torch.tensor((ytr - ym) / ys, dtype=torch.float32, device=DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2); lf = nn.MSELoss()
    rng = np.random.default_rng(0); n = len(ytr)
    for ep in range(200):
        net.train(); perm = rng.permutation(n)
        for b in range(0, n, 128):
            bi = torch.as_tensor(perm[b:b + 128], device=DEV); opt.zero_grad(); lf(net(xtr[bi]), ytn[bi]).backward(); opt.step()
    net.eval()
    with torch.no_grad(): return net(xte).cpu().numpy() * ys + ym


class LSTMreg(nn.Module):
    def __init__(s, h=16): super().__init__(); s.lstm = nn.LSTM(1, h, batch_first=True); s.fc = nn.Linear(h, 1)
    def forward(s, x): o, _ = s.lstm(x); return s.fc(o[:, -1]).squeeze(-1)
def p_lstm(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    xtr = torch.tensor(sc.transform(Xtr)[:, :, None], dtype=torch.float32, device=DEV)
    xte = torch.tensor(sc.transform(Xte)[:, :, None], dtype=torch.float32, device=DEV)
    torch.manual_seed(0); return _train_torch(LSTMreg().to(DEV), xtr, ytr, xte)


class CNN1d(nn.Module):
    """리드 8점 궤적용 1d-CNN 회귀 (LSTM과 동급 소형)."""
    def __init__(s):
        super().__init__()
        s.c1 = nn.Conv1d(1, 16, 3, padding=1); s.c2 = nn.Conv1d(16, 32, 3, padding=1); s.fc = nn.Linear(32, 1)
    def forward(s, x):                       # x: (N, 8, 1) -> (N, 1, 8)
        h = x.transpose(1, 2)
        h = torch.relu(s.c1(h)); h = torch.relu(s.c2(h))
        return s.fc(h.mean(dim=2)).squeeze(-1)   # 전역 평균 풀링
def p_cnn(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    xtr = torch.tensor(sc.transform(Xtr)[:, :, None], dtype=torch.float32, device=DEV)
    xte = torch.tensor(sc.transform(Xte)[:, :, None], dtype=torch.float32, device=DEV)
    torch.manual_seed(0); return _train_torch(CNN1d().to(DEV), xtr, ytr, xte)


MODELS = {"LR": p_lr, "LGBM": p_lgbm, "LSTM": p_lstm, "CNN1d": p_cnn, "TabPFN": p_tabpfn, "TabICL": p_tabicl, "TabNet": p_tabnet}


# ---------- 폴드/평가 ----------
def folds(N, od, Nf=5, emb=64):
    f = N // (Nf + 1); out = []
    for k in range(1, Nf + 1):
        ts = k * f; te = (k + 1) * f if k < Nf else N
        out.append((np.array([j for j in range(0, ts) if od[j] <= od[ts] - emb]), np.arange(ts, te)))
    return out


def boot_ci(yt, pa, pb_):
    rng = np.random.default_rng(0); dd = []
    for _ in range(NB):
        ix = rng.integers(0, len(yt), len(yt))
        if len(np.unique(yt[ix])) > 1: dd.append(f1_score(yt[ix], pa[ix], zero_division=0) - f1_score(yt[ix], pb_[ix], zero_division=0))
    dd = np.array(dd); return np.percentile(dd, 2.5), np.percentile(dd, 97.5), np.mean(dd > 0)


# ---------- LGBM HP: 블록0(항상 train) 시간순 CV — leak-free (기존과 동일 절차, 2지역) ----------
LGBM_GRID = [dict(num_leaves=nl, learning_rate=lr, n_estimators=ne, min_child_samples=mc)
             for nl in (15, 31) for lr in (0.03, 0.05) for ne in (200, 400) for mc in (20, 40)]
def tune_lgbm():
    packs = []
    for R in REGIONS:
        d = np.load(f"opdenom_full_{R}.npz"); D8 = d["D8"]; omin = d["omin"]; THR = float(d["THR"]); N = len(d["y"])
        f = N // 6; packs.append((D8[:f], omin[:f], THR))
    tscv = TimeSeriesSplit(n_splits=3); best = None; best_s = 1e18
    for hp in LGBM_GRID:
        es = []
        for X0, o0, THR in packs:
            for tr, val in tscv.split(X0):
                m = LGBMRegressor(subsample=0.8, verbose=-1, **hp).fit(X0[tr], o0[tr])
                es.append(float(np.sqrt(np.mean((m.predict(X0[val]) - o0[val]) ** 2)) / THR))
        s = float(np.mean(es))
        if s < best_s: best_s = s; best = hp
    return best
LGBM_HP = tune_lgbm()
print(f"[LGBM HP(블록0 CV, 2지역)] {LGBM_HP}\n")


# ---------- 본 평가 ----------
for suf, hlab in HORIZONS:
    print(f"\n========== {hlab} : 2지역 최종 (피처 B, 7모델) ==========")
    REG = {}
    for R in REGIONS:
        d = np.load(f"opdenom_full_{R}{suf}.npz")
        D8 = d["D8"]; fcv = d["fcv"]; y = d["y"]; omin = d["omin"]; oday = d["oday"]; THR = float(d["THR"])
        FL = [(tr, te) for tr, te in folds(len(y), oday) if len(tr) >= 40 and len(np.unique(y[tr])) > 1]
        yt = np.concatenate([y[te] for _, te in FL])
        p0 = np.concatenate([(fcv[te] >= THR).astype(int) for _, te in FL])
        f0 = f1_score(yt, p0, zero_division=0)
        print(f"[{R}] n={len(y)} 지속={int(y.sum())} base={y.mean():.3f} THR={THR:.1f} | raw F1={f0:.3f}")
        REG[R] = {"_yt": yt, "_raw": p0}
        for name, pf in MODELS.items():
            pb = np.concatenate([(np.asarray(pf(D8[tr], omin[tr], D8[te])) >= THR).astype(int) for tr, te in FL])
            ff = f1_score(yt, pb, zero_division=0); lo, hi, pp = boot_ci(yt, pb, p0)
            print(f"  {name:7s}: F1={ff:.3f}  ΔF1={ff - f0:+.3f} CI[{lo:+.3f},{hi:+.3f}] P={pp:.3f}")
            REG[R][name] = pb

    print(f"[통합 (2지역 평균 ΔF1) — {hlab}]")
    for name in MODELS:
        rng = np.random.default_rng(0); md = []
        for _ in range(NB):
            ds = []
            for R in REG:
                yt = REG[R]["_yt"]; p0 = REG[R]["_raw"]; pm = REG[R][name]; ix = rng.integers(0, len(yt), len(yt))
                if len(np.unique(yt[ix])) < 2: ds = None; break
                ds.append(f1_score(yt[ix], pm[ix], zero_division=0) - f1_score(yt[ix], p0[ix], zero_division=0))
            if ds is not None: md.append(np.mean(ds))
        md = np.array(md)
        obs = np.mean([f1_score(REG[R]["_yt"], REG[R][name], zero_division=0) - f1_score(REG[R]["_yt"], REG[R]["_raw"], zero_division=0) for R in REG])
        print(f"  {name:7s} 평균ΔF1={obs:+.3f}  95%CI[{np.percentile(md,2.5):+.3f},{np.percentile(md,97.5):+.3f}]  P(Δ>0)={np.mean(md>0):.3f}")