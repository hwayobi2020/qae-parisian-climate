# ===== Colab: 0.5THR 완전분모 — D-2 env 인코더(MLP 44->8, 3클래스) 얹기 =====
# 질문: D-2 시점 관측 env 를 폴드별 MLP 인코더로 8차원 압축해 예보9 위에 얹으면 raw/예보9 대비 F1 오르나.
# 누수방지: 인코더도 TabPFN 도 매 폴드 train 에서만 학습 -> test 평가 (walk-forward 5폴드).
# 인코더 라벨 y3: 비온셋0 / 온셋<24h 1 / 온셋>=24h 2 (전체 후보일에 정의).
# 준비: opdenom_full_{ca,uk,chile}.npz (git pull; ENV·y3 포함). !pip install tabpfn -q
import numpy as np, torch, torch.nn as nn, os
from tabpfn import TabPFNRegressor
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
DEV = "cuda" if torch.cuda.is_available() else "cpu"; NB = 2000; REGIONS = ["ca", "uk", "chile"]


class MLPenc(nn.Module):                                  # env 인코더: 44 -> (16) -> 8차원 병목 -> 3클래스 헤드
    def __init__(s, cin=44, hid=8, w=16, ncls=3):
        super().__init__(); s.f1 = nn.Linear(cin, w); s.dp = nn.Dropout(0.3); s.f2 = nn.Linear(w, hid); s.head = nn.Linear(hid, ncls)
    def rep(s, x): return torch.relu(s.f2(s.dp(torch.relu(s.f1(x)))))
    def forward(s, x): return s.head(s.rep(x))


def fit_enc(ENVtr, y3tr, hid=8, epochs=300):              # 폴드 train 에서만 인코더 학습 -> 임의 행의 8차원 rep 반환
    mu = np.nanmean(ENVtr, 0); sd = np.nanstd(ENVtr, 0) + 1e-8
    Xtr = torch.tensor(np.nan_to_num((ENVtr - mu) / sd), dtype=torch.float32, device=DEV)
    yt = torch.tensor(y3tr, dtype=torch.long, device=DEV)
    cnt = np.bincount(y3tr, minlength=3).astype(float); cw = len(y3tr) / (3 * np.maximum(cnt, 1))   # 역빈도 클래스가중
    lf = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device=DEV))
    torch.manual_seed(0); net = MLPenc(ENVtr.shape[1], hid).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
    rng = np.random.default_rng(0); n = len(y3tr)
    for ep in range(epochs):
        net.train(); perm = rng.permutation(n)
        for b in range(0, n, 64):
            bi = torch.as_tensor(perm[b:b + 64], device=DEV); opt.zero_grad(); lf(net(Xtr[bi]), yt[bi]).backward(); opt.step()
    net.eval()
    def rep(E):
        with torch.no_grad(): return net.rep(torch.tensor(np.nan_to_num((E - mu) / sd), dtype=torch.float32, device=DEV)).cpu().numpy()
    return rep


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


def reg_preds(D2, ENV, y3, omin, y, od, thr, use_enc):    # TabPFN 회귀. use_enc=True 면 폴드별 인코더8 을 예보9 뒤에 붙임
    yt = []; pb = []; sc = []
    for tr, te in folds(len(y), od):
        if len(tr) < 40 or len(np.unique(y[tr])) < 2: continue
        if use_enc:
            rep = fit_enc(ENV[tr], y3[tr])                 # train 에서만 인코더 학습
            Xtr = np.column_stack([D2[tr], rep(ENV[tr])]); Xte = np.column_stack([D2[te], rep(ENV[te])])
        else:
            Xtr = D2[tr]; Xte = D2[te]
        m = TabPFNRegressor(device=DEV); m.fit(np.nan_to_num(Xtr), omin[tr])
        pred = np.asarray(m.predict(np.nan_to_num(Xte)))
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
    D2 = d["D2"]; ENV = d["ENV"]; fcv = d["fcv"]; y = d["y"]; y3 = d["y3"].astype(int); omin = d["omin"]; oday = d["oday"]; THR = float(d["THR"])
    yt0, p0, s0 = raw_preds(fcv, y, oday, THR); f0 = f1_score(yt0, p0, zero_division=0)
    yt1, p1, s1 = reg_preds(D2, ENV, y3, omin, y, oday, THR, use_enc=False); f1v = f1_score(yt1, p1, zero_division=0)
    yt2, p2, s2 = reg_preds(D2, ENV, y3, omin, y, oday, THR, use_enc=True); f2 = f1_score(yt2, p2, zero_division=0)
    lo1, hi1, pp1 = boot(yt0, p1, p0); lo2, hi2, pp2 = boot(yt0, p2, p0)
    print(f"[{R}] n={len(y)} 지속률={y.mean():.3f} THR={THR:.1f}")
    print(f"  기준 raw        : F1={f0:.3f} AUC={roc_auc_score(yt0, s0):.3f}")
    print(f"  예보9(TabPFN)   : F1={f1v:.3f} AUC={roc_auc_score(yt1, s1):.3f} AUPRC={average_precision_score(yt1, s1):.3f}  ΔF1={f1v - f0:+.3f} CI[{lo1:+.3f},{hi1:+.3f}] P={pp1:.3f}")
    print(f"  예보9+인코더8   : F1={f2:.3f} AUC={roc_auc_score(yt2, s2):.3f} AUPRC={average_precision_score(yt2, s2):.3f}  ΔF1={f2 - f0:+.3f} CI[{lo2:+.3f},{hi2:+.3f}] P={pp2:.3f}")
    REG[R] = {"_raw": (yt0, p0), "예보9": (yt1, p1), "예보9+enc": (yt2, p2)}

# ===== 통합검정: 예보9 / 예보9+인코더 각각 vs raw, 3지역 평균 ΔF1 (지역 부트스트랩) =====
print("\n[통합 ΔF1(vs raw), 지역 평균]")
for name in ["예보9", "예보9+enc"]:
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
