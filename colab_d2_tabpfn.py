# ===== Colab: D-2(2일전) 예보 프레임 TabPFN =====
# 3단: 기준(raw D-2 min) / 예보풍부(TabPFN D2 9피처) / 예보풍부+인코더(TabPFN D2+ENC). 인코더 pre-2000(leak-free).
# 준비: transfer_{r}.npz + d2feat_{r}.npz (git pull). !pip install tabpfn -q
import numpy as np, torch, torch.nn as nn
from tabpfn import TabPFNClassifier
from sklearn.metrics import matthews_corrcoef
DEV = "cuda" if torch.cuda.is_available() else "cpu"; FEAT = ""; W = 16; NB = 3000; HIDS = [4, 8, 16]


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


def fit_enc(FE, y, idx, hid):
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
    net.eval(); return net, FEs


def prob(net, FEs, ii):
    with torch.no_grad(): return torch.sigmoid(net(torch.tensor(FEs[ii], dtype=torch.float32, device=DEV))).cpu().numpy()


def rep(net, FEs):
    with torch.no_grad(): return net.rep(torch.tensor(FEs, dtype=torch.float32, device=DEV)).cpu().numpy()


def raw_preds(fcv, yf, odf):
    yt = []; pb = []
    for tr, te in folds(len(yf), odf):
        if len(tr) < 40 or len(np.unique(yf[tr])) < 2: continue
        th = bt(yf[tr], fcv[tr]); pb.extend((fcv[te] >= th).astype(int)); yt.extend(yf[te])
    return np.array(yt), np.array(pb)


def tp_preds(X, yf, odf):
    yt = []; pb = []
    for tr, te in folds(len(yf), odf):
        if len(tr) < 40 or len(np.unique(yf[tr])) < 2: continue
        m = TabPFNClassifier(device=DEV); m.fit(np.nan_to_num(X[tr]), yf[tr])
        ptr = m.predict_proba(np.nan_to_num(X[tr]))[:, 1]; pte = m.predict_proba(np.nan_to_num(X[te]))[:, 1]
        th = bt(yf[tr], ptr); pb.extend((pte >= th).astype(int)); yt.extend(yf[te])
    return np.array(yt), np.array(pb)


def boot(yt, pa, pb_):   # P(a>b)
    rng = np.random.default_rng(0); dd = []
    for _ in range(NB):
        ix = rng.integers(0, len(yt), len(yt))
        if len(np.unique(yt[ix])) > 1: dd.append(matthews_corrcoef(yt[ix], pa[ix]) - matthews_corrcoef(yt[ix], pb_[ix]))
    dd = np.array(dd); return np.percentile(dd, 2.5), np.percentile(dd, 97.5), np.mean(dd > 0)


for R in ["ca", "uk", "chile"]:
    d = np.load(FEAT + f"transfer_{R}.npz"); FE, y, oday, S = d["FE"], d["y"], d["oday"], d["s"].astype(int)
    z = np.load(FEAT + f"d2feat_{R}.npz"); s2 = z["s"].astype(int); D2 = z["D2"]; D2min = z["D2min"]
    d2map = {int(s2[k]): k for k in range(len(s2))}
    sel = np.array([int(S[i]) in d2map for i in range(len(S))])
    ridx = np.array([d2map[int(S[i])] for i in range(len(S)) if int(S[i]) in d2map])
    fcv = D2min[ridx]; D2s = D2[ridx]; y_s = y[sel]; od_s = oday[sel]
    # 인코더: pre-2000 val로 hid 선택
    pre = oday < oday[sel].min(); pidx = np.where(pre)[0]; cut = int(len(pidx) * 0.8); tri, vai = pidx[:cut], pidx[cut:]
    best = (-2., None)
    for hid in HIDS:
        net, FEs = fit_enc(FE, y, tri, hid); th = bt(y[tri], prob(net, FEs, tri))
        mv = matthews_corrcoef(y[vai], (prob(net, FEs, vai) >= th).astype(int))
        if mv > best[0]: best = (mv, hid)
    sel_hid = best[1]; net, FEs = fit_enc(FE, y, pidx, sel_hid); ENC = rep(net, FEs)[sel]
    print(f"[{R}] n={int(sel.sum())} 양성률={y_s.mean():.2f} fcv-y상관={np.corrcoef(fcv, y_s)[0,1]:+.2f} 선택hid={sel_hid}")
    yt0, p0 = raw_preds(fcv, y_s, od_s); m0 = matthews_corrcoef(yt0, p0)
    yt1, p1 = tp_preds(D2s, y_s, od_s); m1 = matthews_corrcoef(yt1, p1)
    yt2, p2 = tp_preds(np.column_stack([D2s, ENC]), y_s, od_s); m2 = matthews_corrcoef(yt2, p2)
    lo, hi, pp = boot(yt0, p2, p0)
    print(f"  기준 raw={m0:.3f} | 예보풍부(TabPFN)={m1:.3f} | 예보풍부+인코더={m2:.3f}")
    print(f"  (풍부+인코더 vs 기준) Δ={m2-m0:+.3f} 95%CI[{lo:+.3f},{hi:+.3f}] P(Δ>0)={pp:.3f}")
