# ===== Colab: Result 빈칸 채우기 — (1) 18h 블록, (2) feature ablation =====
# 목적 1 (4.5.1): 지속 기준 18h 결과 → 18/24/30h 세 기준 강건성 완성.
# 목적 2 (4.5.2): "예보 원값 8개로 충분한가" — 궤적 요약통계 / 관측 환경변수를 추가해도 개선되는지.
#   피처셋: B(예보 8개, 메인) / B+요약4 / B+IVT웨이블릿6 / B+관측환경44(전체)
#   ※ 관측 환경변수를 인코더로 압축하지 않고 원값 그대로 추가 — 압축 과정의 과적합 영향을 배제.
# 준비: opdenom_full_{ca,chile}{,_18h,_30h}.npz (git pull). !pip install tabpfn lightgbm -q
import warnings, logging, os
warnings.filterwarnings("ignore"); logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
import numpy as np, torch, torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from lightgbm import LGBMRegressor
from tabpfn import TabPFNRegressor
DEV = "cuda" if torch.cuda.is_available() else "cpu"; NB = 2000; REGIONS = ["ca", "chile"]
LGBM_HP = dict(num_leaves=15, learning_rate=0.03, n_estimators=200, min_child_samples=20)


def folds(N, od, Nf=5, emb=64):
    f = N // (Nf + 1); out = []
    for k in range(1, Nf + 1):
        ts = k * f; te = (k + 1) * f if k < Nf else N
        out.append((np.array([j for j in range(0, ts) if od[j] <= od[ts] - emb]), np.arange(ts, te)))
    return out


def boot_ci(yt, pa, pb_):
    """pa 와 pb_ 의 F1 차이에 대한 95% CI 와 P(Δ>0)."""
    rng = np.random.default_rng(0); dd = []
    for _ in range(NB):
        ix = rng.integers(0, len(yt), len(yt))
        if len(np.unique(yt[ix])) > 1:
            dd.append(f1_score(yt[ix], pa[ix], zero_division=0) - f1_score(yt[ix], pb_[ix], zero_division=0))
    dd = np.array(dd); return np.percentile(dd, 2.5), np.percentile(dd, 97.5), np.mean(dd > 0)


def boot_pooled(REG, key_a, key_b):
    """2지역 평균 F1 차이(key_a - key_b)의 CI·P."""
    rng = np.random.default_rng(0); md = []
    for _ in range(NB):
        ds = []
        for R in REG:
            yt = REG[R]["_yt"]; pa = REG[R][key_a]; pb = REG[R][key_b]
            ix = rng.integers(0, len(yt), len(yt))
            if len(np.unique(yt[ix])) < 2: ds = None; break
            ds.append(f1_score(yt[ix], pa[ix], zero_division=0) - f1_score(yt[ix], pb[ix], zero_division=0))
        if ds is not None: md.append(np.mean(ds))
    md = np.array(md)
    obs = np.mean([f1_score(REG[R]["_yt"], REG[R][key_a], zero_division=0)
                   - f1_score(REG[R]["_yt"], REG[R][key_b], zero_division=0) for R in REG])
    return obs, np.percentile(md, 2.5), np.percentile(md, 97.5), np.mean(md > 0)


# ---------- 모델 ----------
def p_lr(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr); return LinearRegression().fit(sc.transform(Xtr), ytr).predict(sc.transform(Xte))
def p_lgbm(Xtr, ytr, Xte):
    return LGBMRegressor(subsample=0.8, verbose=-1, **LGBM_HP).fit(Xtr, ytr).predict(Xte)
def p_tabpfn(Xtr, ytr, Xte):
    m = TabPFNRegressor(device=DEV); m.fit(np.nan_to_num(Xtr), ytr); return np.asarray(m.predict(np.nan_to_num(Xte)))


def _train_torch(net, xtr, ytr, xte):
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
    def __init__(s):
        super().__init__()
        s.c1 = nn.Conv1d(1, 16, 3, padding=1); s.c2 = nn.Conv1d(16, 32, 3, padding=1); s.fc = nn.Linear(32, 1)
    def forward(s, x):
        h = x.transpose(1, 2); h = torch.relu(s.c1(h)); h = torch.relu(s.c2(h))
        return s.fc(h.mean(dim=2)).squeeze(-1)
def p_cnn(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    xtr = torch.tensor(sc.transform(Xtr)[:, :, None], dtype=torch.float32, device=DEV)
    xte = torch.tensor(sc.transform(Xte)[:, :, None], dtype=torch.float32, device=DEV)
    torch.manual_seed(0); return _train_torch(CNN1d().to(DEV), xtr, ytr, xte)


# ================= (1) 18h 블록 =================
print("=" * 70)
print("[1] 지속 기준 18h — 4.5.1 보완 (피처 B, 2지역)")
print("=" * 70)
MODELS_18 = {"LR": p_lr, "LGBM": p_lgbm, "LSTM": p_lstm, "CNN1d": p_cnn, "TabPFN": p_tabpfn}
REG18 = {}
for R in REGIONS:
    d = np.load(f"opdenom_full_{R}_18h.npz")
    D8 = d["D8"]; fcv = d["fcv"]; y = d["y"]; omin = d["omin"]; oday = d["oday"]; THR = float(d["THR"])
    FL = [(tr, te) for tr, te in folds(len(y), oday) if len(tr) >= 40 and len(np.unique(y[tr])) > 1]
    yt = np.concatenate([y[te] for _, te in FL])
    p0 = np.concatenate([(fcv[te] >= THR).astype(int) for _, te in FL])
    f0 = f1_score(yt, p0, zero_division=0)
    print(f"[{R}] n={len(y)} 지속={int(y.sum())} base={y.mean():.3f} THR={THR:.1f} | raw F1={f0:.3f}")
    REG18[R] = {"_yt": yt, "_raw": p0}
    for name, pf in MODELS_18.items():
        pb = np.concatenate([(np.asarray(pf(D8[tr], omin[tr], D8[te])) >= THR).astype(int) for tr, te in FL])
        ff = f1_score(yt, pb, zero_division=0); lo, hi, pp = boot_ci(yt, pb, p0)
        print(f"  {name:7s}: F1={ff:.3f}  ΔF1={ff - f0:+.3f} CI[{lo:+.3f},{hi:+.3f}] P={pp:.3f}")
        REG18[R][name] = pb
print("[통합 (2지역 평균 ΔF1) — 18h]")
for name in MODELS_18:
    obs, lo, hi, pp = boot_pooled(REG18, name, "_raw")
    print(f"  {name:7s} 평균ΔF1={obs:+.3f}  95%CI[{lo:+.3f},{hi:+.3f}]  P(Δ>0)={pp:.3f}")


# ================= (2) feature ablation =================
print("\n" + "=" * 70)
print("[2] Feature ablation — 4.5.2 (24h, 2지역)")
print("    B=예보 원값 8개(메인) / +요약4 / +IVT웨이블릿6 / +관측환경44(전체)")
print("=" * 70)
FEATSETS = ["B", "B+요약4", "B+웨이블릿6", "B+환경44"]
for MNAME, PF in [("TabPFN", p_tabpfn), ("LGBM", p_lgbm)]:
    print(f"\n--- {MNAME} ---")
    REGF = {}
    for R in REGIONS:
        d = np.load(f"opdenom_full_{R}.npz")
        D8 = d["D8"]; D2 = d["D2"]; ENV = d["ENV"]; fcv = d["fcv"]
        y = d["y"]; omin = d["omin"]; oday = d["oday"]; THR = float(d["THR"])
        SETS = {"B": D8,
                "B+요약4": np.column_stack([D8, D2[:, 5:9]]),
                "B+웨이블릿6": np.column_stack([D8, ENV[:, 0:6]]),
                "B+환경44": np.column_stack([D8, ENV])}
        FL = [(tr, te) for tr, te in folds(len(y), oday) if len(tr) >= 40 and len(np.unique(y[tr])) > 1]
        yt = np.concatenate([y[te] for _, te in FL])
        p0 = np.concatenate([(fcv[te] >= THR).astype(int) for _, te in FL])
        f0 = f1_score(yt, p0, zero_division=0)
        REGF[R] = {"_yt": yt, "_raw": p0}
        line = f"  [{R}] raw F1={f0:.3f}"
        for sn in FEATSETS:
            X = SETS[sn]
            pb = np.concatenate([(np.asarray(PF(X[tr], omin[tr], X[te])) >= THR).astype(int) for tr, te in FL])
            REGF[R][sn] = pb
            line += f" | {sn}={f1_score(yt, pb, zero_division=0):.3f}"
        print(line + f"   (피처 수: 8 / 12 / 14 / 52)")
    print(f"  [통합 — vs 원예보]")
    for sn in FEATSETS:
        obs, lo, hi, pp = boot_pooled(REGF, sn, "_raw")
        print(f"    {sn:12s} 평균ΔF1={obs:+.3f}  95%CI[{lo:+.3f},{hi:+.3f}]  P={pp:.3f}")
    print(f"  [통합 — vs B (피처 추가 효과)]")
    for sn in FEATSETS[1:]:
        obs, lo, hi, pp = boot_pooled(REGF, sn, "B")
        print(f"    {sn:12s} 평균Δ(vs B)={obs:+.3f}  95%CI[{lo:+.3f},{hi:+.3f}]  P={pp:.3f}")