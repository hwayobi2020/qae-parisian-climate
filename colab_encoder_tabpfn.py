# ===== Colab: 기준=raw 기상청 예보(min IVT). 우리모델(예보재보정 / 예보+인코더) 이 기준을 넘나. =====
# 준비: transfer_{ca,uk,chile}.npz (git pull). tabpfn 설치 셀: !pip install tabpfn -q
# 인코더=pre-2000 MLP(BCE학습, MCC 선택), leak-free. 결합=TabPFN. 지표 MCC(+AUC/F1).
import numpy as np, torch, torch.nn as nn, copy
from tabpfn import TabPFNClassifier
from sklearn.metrics import roc_auc_score, matthews_corrcoef, f1_score
DEV = "cuda" if torch.cuda.is_available() else "cpu"; FEAT_DIR = ""; HIDS = [4, 8, 16]; W = 16


class MLPenc(nn.Module):
    def __init__(s, cin, hid, w=W):
        super().__init__(); s.f1 = nn.Linear(cin, w); s.dp = nn.Dropout(0.3); s.f2 = nn.Linear(w, hid); s.head = nn.Linear(hid, 1)
    def rep(s, x): return torch.relu(s.f2(s.dp(torch.relu(s.f1(x)))))
    def forward(s, x): return s.head(s.rep(x)).squeeze(-1)


def folds(N, od, Nf=5, emb=64):
    f = N // (Nf + 1); out = []
    for k in range(1, Nf + 1):
        ts = k * f; te = (k + 1) * f if k < Nf else N
        out.append((np.array([j for j in range(0, ts) if od[j] <= od[ts] - emb]), np.arange(ts, te)))
    return out


def bt(yy, pp):
    b = (-2., .5)
    for t in np.unique(pp):
        m = matthews_corrcoef(yy, (pp >= t).astype(int))
        if m > b[0]: b = (m, t)
    return b[1]


def train_enc(FE, y, pre, hid):
    idx = np.where(pre)[0]; cut = int(len(idx) * 0.8)
    tr_i, va_i = idx[:cut], idx[cut:]
    mu = np.nanmean(FE[tr_i], 0); sd = np.nanstd(FE[tr_i], 0) + 1e-8; FEs = np.nan_to_num((FE - mu) / sd)
    Xtr = torch.tensor(FEs[tr_i], dtype=torch.float32, device=DEV); ytr = torch.tensor(y[tr_i], dtype=torch.float32, device=DEV)
    Xva = torch.tensor(FEs[va_i], dtype=torch.float32, device=DEV); yva = y[va_i]
    torch.manual_seed(0); net = MLPenc(FE.shape[1], hid).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
    pw = torch.tensor((len(tr_i) - float(ytr.sum())) / max(float(ytr.sum()), 1), dtype=torch.float32, device=DEV)
    lf = nn.BCEWithLogitsLoss(pos_weight=pw); rng = np.random.default_rng(0); nt = len(tr_i); best = (-2.0, None)
    for ep in range(300):
        net.train(); perm = rng.permutation(nt)
        for b in range(0, nt, 64):
            bi = torch.as_tensor(perm[b:b + 64], device=DEV)
            opt.zero_grad(); lf(net(Xtr[bi]), ytr[bi]).backward(); opt.step()
        if ep % 10 == 0 and len(np.unique(yva)) > 1:
            net.eval()
            with torch.no_grad(): pv = torch.sigmoid(net(Xva)).cpu().numpy()
            th = bt(yva, pv); mc = matthews_corrcoef(yva, (pv >= th).astype(int))
            if mc > best[0]: best = (mc, copy.deepcopy(net.state_dict()))
    if best[1] is not None: net.load_state_dict(best[1])
    net.eval()
    with torch.no_grad(): return net.rep(torch.tensor(FEs, dtype=torch.float32, device=DEV)).cpu().numpy()


def base_raw(fcv, yf, odf):        # 기준: raw 예보 min IVT 직접 임계 (모델 없음)
    ytb = []; pb = []; pp = np.full(len(yf), np.nan)
    for tr, te in folds(len(yf), odf):
        if len(tr) < 40 or len(np.unique(yf[tr])) < 2: continue
        th = bt(yf[tr], fcv[tr]); pp[te] = fcv[te]; pb.extend((fcv[te] >= th).astype(int)); ytb.extend(yf[te])
    ok = ~np.isnan(pp); ytb = np.array(ytb); pb = np.array(pb)
    return roc_auc_score(yf[ok], pp[ok]), matthews_corrcoef(ytb, pb), f1_score(ytb, pb, zero_division=0)


def tpcomb(X, yf, odf):            # 우리 모델: TabPFN
    ytb = []; pb = []; pp = np.full(len(yf), np.nan)
    for tr, te in folds(len(yf), odf):
        if len(tr) < 40 or len(np.unique(yf[tr])) < 2: continue
        m = TabPFNClassifier(device=DEV); m.fit(np.nan_to_num(X[tr]), yf[tr])
        ptr = m.predict_proba(np.nan_to_num(X[tr]))[:, 1]; pte = m.predict_proba(np.nan_to_num(X[te]))[:, 1]
        pp[te] = pte; th = bt(yf[tr], ptr); pb.extend((pte >= th).astype(int)); ytb.extend(yf[te])
    ok = ~np.isnan(pp); ytb = np.array(ytb); pb = np.array(pb)
    return roc_auc_score(yf[ok], pp[ok]), matthews_corrcoef(ytb, pb), f1_score(ytb, pb, zero_division=0)


for r in ["ca", "uk", "chile"]:
    d = np.load(FEAT_DIR + f"transfer_{r}.npz")
    FE, FC, y, oday, fmask = d["FE"], d["FC"], d["y"], d["oday"], d["fmask"].astype(bool)
    pre = oday < oday[fmask].min()
    sel = fmask; FCf = FC[sel]; yf = y[sel]; odf = oday[sel]; fcv = FCf.mean(1)
    ab, mb, fb = base_raw(fcv, yf, odf)
    print(f"[{r}] pre-2000 {int(pre.sum())} | 예보온셋 {int(sel.sum())} 양성 {int(yf.sum())}")
    print(f"  [기준] raw 예보(min IVT): AUC={ab:.3f} MCC={mb:.3f} F1={fb:.3f}")
    a0, m0, f0 = tpcomb(fcv.reshape(-1, 1), yf, odf)
    print(f"  우리모델(예보만)       : AUC={a0:.3f} MCC={m0:.3f} F1={f0:.3f} | vs기준 {m0 - mb:+.3f}")
    for hid in HIDS:
        ENCf = train_enc(FE, y, pre, hid)[sel]
        a, m, f = tpcomb(np.column_stack([fcv, ENCf]), yf, odf)
        print(f"  우리모델(예보+인코더 hid={hid:2d}): AUC={a:.3f} MCC={m:.3f} F1={f:.3f} | vs기준 {m - mb:+.3f}")
