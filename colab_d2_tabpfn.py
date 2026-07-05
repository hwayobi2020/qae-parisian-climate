# ===== Colab: D-2(2일전) 예보 프레임 — TabPFN 회귀 =====
# 기준 = raw D-2 예보 min IVT(fcv). 우리모델 = TabPFN회귀(예보피처9 -> 관측 min IVT 예측).
# 둘 다 지역별 THR로 자름: 예측/예보 값 >= THR 이면 지속, < THR 이면 미지속. 지표 = F1. env 없음.
# 준비: transfer_{r}.npz + d2feat_{r}.npz + d2env_{r}.npz (git pull). !pip install tabpfn -q
import numpy as np, torch, torch.nn as nn
from tabpfn import TabPFNRegressor
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
DEV = "cuda"; FEAT = ""; NB = 3000


class MLPenc(nn.Module):                                 # env 인코더 (pre-2000 지속 학습, 8차원 표현)
    def __init__(s, cin, hid=8, w=16):
        super().__init__(); s.f1 = nn.Linear(cin, w); s.dp = nn.Dropout(0.3); s.f2 = nn.Linear(w, hid); s.head = nn.Linear(hid, 1)
    def rep(s, x): return torch.relu(s.f2(s.dp(torch.relu(s.f1(x)))))
    def forward(s, x): return s.head(s.rep(x)).squeeze(-1)


def fit_enc(FE, y, idx, hid=8):                          # pre-2000 학습 후 전체 온셋의 8차원 rep 반환 (leak-free)
    mu = np.nanmean(FE[idx], 0); sd = np.nanstd(FE[idx], 0) + 1e-8; FEs = np.nan_to_num((FE - mu) / sd)
    Xtr = torch.tensor(FEs[idx], dtype=torch.float32, device=DEV); yt = torch.tensor(y[idx], dtype=torch.float32, device=DEV)
    torch.manual_seed(0); net = MLPenc(FE.shape[1], hid).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
    pw = torch.tensor((len(idx) - y[idx].sum()) / max(y[idx].sum(), 1), dtype=torch.float32, device=DEV)
    lf = nn.BCEWithLogitsLoss(pos_weight=pw); rng = np.random.default_rng(0); n = len(idx)
    for ep in range(300):
        net.train(); perm = rng.permutation(n)
        for b in range(0, n, 64):
            bi = torch.as_tensor(perm[b:b + 64], device=DEV); opt.zero_grad(); lf(net(Xtr[bi]), yt[bi]).backward(); opt.step()
    net.eval()
    with torch.no_grad(): return net.rep(torch.tensor(FEs, dtype=torch.float32, device=DEV)).cpu().numpy()


def folds(N, od, Nf=5, emb=64):
    f = N // (Nf + 1); out = []
    for k in range(1, Nf + 1):
        ts = k * f; te = (k + 1) * f if k < Nf else N
        out.append((np.array([j for j in range(0, ts) if od[j] <= od[ts] - emb]), np.arange(ts, te)))
    return out


def raw_preds(fcv, yf, odf, thr):                       # 기준: 예보 min IVT. F1은 >=THR, AUC/AUPRC는 fcv 점수 그대로
    yt = []; pb = []; sc = []
    for tr, te in folds(len(yf), odf):
        if len(tr) < 40 or len(np.unique(yf[tr])) < 2: continue
        pb.extend((fcv[te] >= thr).astype(int)); sc.extend(fcv[te]); yt.extend(yf[te])
    return np.array(yt), np.array(pb), np.array(sc)


def reg_preds(X, omin, yf, odf, thr):                   # 우리모델: 예보피처->관측 min IVT 회귀예측. F1은 >=THR, AUC/AUPRC는 예측값 점수
    yt = []; pb = []; sc = []
    for tr, te in folds(len(yf), odf):
        if len(tr) < 40 or len(np.unique(yf[tr])) < 2: continue
        m = TabPFNRegressor(device=DEV); m.fit(np.nan_to_num(X[tr]), omin[tr])
        pred = np.asarray(m.predict(np.nan_to_num(X[te])))
        pb.extend((pred >= thr).astype(int)); sc.extend(pred); yt.extend(yf[te])
    return np.array(yt), np.array(pb), np.array(sc)


def boot(yt, pa, pb_):                                  # ΔF1 (모델 - 기준)
    rng = np.random.default_rng(0); dd = []
    for _ in range(NB):
        ix = rng.integers(0, len(yt), len(yt))
        if len(np.unique(yt[ix])) > 1:
            dd.append(f1_score(yt[ix], pa[ix], zero_division=0) - f1_score(yt[ix], pb_[ix], zero_division=0))
    dd = np.array(dd); return np.percentile(dd, 2.5), np.percentile(dd, 97.5), np.mean(dd > 0)


REG = {}
for R in ["ca", "uk", "chile"]:
    d = np.load(FEAT + f"transfer_{R}.npz"); y, oday, S = d["y"], d["oday"], d["s"].astype(int)
    de = np.load(FEAT + f"d2env_{R}.npz"); THR = float(de["THR"]); OMIN = de["omin"]; FE = de["FE"]
    assert np.array_equal(de["s"].astype(int), S), "d2env-transfer 온셋 불일치"
    z = np.load(FEAT + f"d2feat_{R}.npz"); s2 = z["s"].astype(int); D2 = z["D2"]; D2min = z["D2min"]
    d2map = {int(s2[k]): k for k in range(len(s2))}
    keep = [i for i in range(len(S)) if int(S[i]) in d2map and not np.isnan(OMIN[i])]   # D-2예보 있고 회귀타깃 유효
    ridx = np.array([d2map[int(S[i])] for i in keep])
    fcv = D2min[ridx]; D2s = D2[ridx]; y_s = y[keep]; od_s = oday[keep]; omin_s = OMIN[keep]
    print(f"[{R}] n={len(keep)} 양성률={y_s.mean():.2f} THR={THR:.1f} fcv-y상관={np.corrcoef(fcv, y_s)[0,1]:+.2f}")
    yt0, p0, s0 = raw_preds(fcv, y_s, od_s, THR); f0 = f1_score(yt0, p0, zero_division=0)
    yt1, p1, s1 = reg_preds(D2s, omin_s, y_s, od_s, THR); f1v = f1_score(yt1, p1, zero_division=0)
    a0 = roc_auc_score(yt0, s0); a1 = roc_auc_score(yt1, s1); ap0 = average_precision_score(yt0, s0); ap1 = average_precision_score(yt1, s1)
    lo, hi, pp = boot(yt0, p1, p0)
    pre = oday < min(oday[keep]); ENC = fit_enc(FE, y, np.where(pre)[0])[keep]        # env 인코더(pre-2000, D-2컷) 8차원
    yt2, p2, sc2 = reg_preds(np.column_stack([D2s, ENC]), omin_s, y_s, od_s, THR); f2 = f1_score(yt2, p2, zero_division=0)
    a2 = roc_auc_score(yt2, sc2); ap2 = average_precision_score(yt2, sc2); lo2, hi2, pp2 = boot(yt0, p2, p0)
    print(f"  기준 raw       : F1={f0:.3f} AUC={a0:.3f} AUPRC={ap0:.3f}")
    print(f"  우리모델(예보)  : F1={f1v:.3f} AUC={a1:.3f} AUPRC={ap1:.3f}  ΔF1={f1v - f0:+.3f}(CI[{lo:+.3f},{hi:+.3f}] P={pp:.3f})")
    print(f"  예보+인코더     : F1={f2:.3f} AUC={a2:.3f} AUPRC={ap2:.3f}  ΔF1={f2 - f0:+.3f}(CI[{lo2:+.3f},{hi2:+.3f}] P={pp2:.3f})")
    REG[R] = (yt0, p0, p1, p2)

# ===== 통합검정: 3지역 평균 ΔF1 부트스트랩 (REG=(yt,p_raw,p_예보,p_인코더)) =====
def pooled(idx, tag):
    rng = np.random.default_rng(0); md = []
    for _ in range(NB):
        ds = []
        for R in REG:
            t = REG[R]; ix = rng.integers(0, len(t[0]), len(t[0]))
            if len(np.unique(t[0][ix])) < 2: ds = None; break
            ds.append(f1_score(t[0][ix], t[idx][ix], zero_division=0) - f1_score(t[0][ix], t[1][ix], zero_division=0))
        if ds is not None: md.append(np.mean(ds))
    md = np.array(md)
    obs = np.mean([f1_score(REG[R][0], REG[R][idx], zero_division=0) - f1_score(REG[R][0], REG[R][1], zero_division=0) for R in REG])
    print(f"[통합 {tag}] 평균 ΔF1={obs:+.3f}  95%CI[{np.percentile(md, 2.5):+.3f},{np.percentile(md, 97.5):+.3f}]  P(Δ>0)={np.mean(md > 0):.3f}")


print()
pooled(2, "예보 vs 기준")          # idx2 = p_예보
pooled(3, "예보+인코더 vs 기준")   # idx3 = p_인코더
