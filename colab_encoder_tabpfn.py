# ===== Colab: MLP인코더(pre-2000) 크기 4/8/16 + 예보 -> TabPFN 결합. leak-free. MCC. =====
# 준비: transfer_{ca,uk,chile}.npz (git pull). tabpfn 설치 셀: !pip install tabpfn -q
import numpy as np, torch, torch.nn as nn
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
    mu = np.nanmean(FE[pre], 0); sd = np.nanstd(FE[pre], 0) + 1e-8; FEs = np.nan_to_num((FE - mu) / sd)
    Xtr = torch.tensor(FEs[pre], dtype=torch.float32, device=DEV); yt = torch.tensor(y[pre], dtype=torch.float32, device=DEV)
    torch.manual_seed(0); net = MLPenc(FE.shape[1], hid).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
    pw = torch.tensor((len(y[pre]) - y[pre].sum()) / max(y[pre].sum(), 1), dtype=torch.float32, device=DEV)
    lf = nn.BCEWithLogitsLoss(pos_weight=pw); rng = np.random.default_rng(0); npr = int(pre.sum())
    for ep in range(300):
        net.train(); perm = rng.permutation(npr)
        for b in range(0, npr, 64):
            bi = torch.as_tensor(perm[b:b + 64], device=DEV)
            opt.zero_grad(); lf(net(Xtr[bi]), yt[bi]).backward(); opt.step()
    net.eval()
    with torch.no_grad(): return net.rep(torch.tensor(FEs, dtype=torch.float32, device=DEV)).cpu().numpy()


def tpcomb(X, yf, odf):
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
    sel = fmask; FCf = FC[sel]; yf = y[sel]; odf = oday[sel]
    ab, mb, fb = tpcomb(FCf, yf, odf)
    print(f"[{r}] pre-2000 {int(pre.sum())} | 예보온셋 {int(sel.sum())} 양성 {int(yf.sum())}")
    print(f"  예보단독        : AUC={ab:.3f} MCC={mb:.3f} F1={fb:.3f}")
    for hid in HIDS:
        ENCf = train_enc(FE, y, pre, hid)[sel]
        a, m, f = tpcomb(np.hstack([FCf, ENCf]), yf, odf)
        print(f"  예보+인코더(hid={hid:2d}): AUC={a:.3f} MCC={m:.3f} F1={f:.3f} | env효과 {m - mb:+.3f}")
