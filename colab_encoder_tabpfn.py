# ===== Colab: 기준=raw 예보(min IVT). 우리모델=예보+LSTM인코더(pre-2000 시퀀스). leak-free. MCC. =====
# 준비: transfer_{ca,uk,chile}.npz (git pull). tabpfn 설치 셀: !pip install tabpfn -q
# LSTM 인코더: pre-2000 IVT+웨이블릿 시퀀스(64x7) BCE학습 -> 풀링 히든(2*hid) 인코더피처 -> 예보와 TabPFN 결합.
import numpy as np, torch, torch.nn as nn
from tabpfn import TabPFNClassifier
from sklearn.metrics import roc_auc_score, matthews_corrcoef, f1_score
DEV = "cuda" if torch.cuda.is_available() else "cpu"; FEAT_DIR = ""; HIDS = [4, 8, 16]


class LSTMenc(nn.Module):
    def __init__(s, cin, hid):
        super().__init__(); s.lstm = nn.LSTM(cin, hid, batch_first=True, bidirectional=True); s.dp = nn.Dropout(0.3); s.head = nn.Linear(hid * 2, 1)
    def rep(s, x): o, _ = s.lstm(x); return o.mean(1)
    def forward(s, x): return s.head(s.dp(s.rep(x))).squeeze(-1)


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


def train_enc(XSEQ, y, pre, hid):
    cmu = XSEQ[pre].reshape(-1, XSEQ.shape[-1]).mean(0); csd = XSEQ[pre].reshape(-1, XSEQ.shape[-1]).std(0) + 1e-8
    Xs = (XSEQ - cmu) / csd
    Xtr = torch.tensor(Xs[pre], dtype=torch.float32, device=DEV); yt = torch.tensor(y[pre], dtype=torch.float32, device=DEV)
    torch.manual_seed(0); net = LSTMenc(XSEQ.shape[-1], hid).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    pw = torch.tensor((len(y[pre]) - y[pre].sum()) / max(y[pre].sum(), 1), dtype=torch.float32, device=DEV)
    lf = nn.BCEWithLogitsLoss(pos_weight=pw); rng = np.random.default_rng(0); npr = int(pre.sum())
    for ep in range(150):
        net.train(); perm = rng.permutation(npr)
        for b in range(0, npr, 64):
            bi = torch.as_tensor(perm[b:b + 64], device=DEV)
            opt.zero_grad(); lf(net(Xtr[bi]), yt[bi]).backward(); opt.step()
    net.eval()
    with torch.no_grad(): return net.rep(torch.tensor(Xs, dtype=torch.float32, device=DEV)).cpu().numpy()


def base_raw(fcv, yf, odf):
    ytb = []; pb = []; pp = np.full(len(yf), np.nan)
    for tr, te in folds(len(yf), odf):
        if len(tr) < 40 or len(np.unique(yf[tr])) < 2: continue
        th = bt(yf[tr], fcv[tr]); pp[te] = fcv[te]; pb.extend((fcv[te] >= th).astype(int)); ytb.extend(yf[te])
    ok = ~np.isnan(pp); ytb = np.array(ytb); pb = np.array(pb)
    return roc_auc_score(yf[ok], pp[ok]), matthews_corrcoef(ytb, pb), f1_score(ytb, pb, zero_division=0)


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
    XSEQ, FC, y, oday, fmask = d["XSEQ"], d["FC"], d["y"], d["oday"], d["fmask"].astype(bool)
    pre = oday < oday[fmask].min()
    sel = fmask; FCf = FC[sel]; yf = y[sel]; odf = oday[sel]; fcv = FCf.mean(1)
    ab, mb, fb = base_raw(fcv, yf, odf)
    print(f"[{r}] pre-2000 {int(pre.sum())} | 예보온셋 {int(sel.sum())} 양성 {int(yf.sum())}")
    print(f"  [기준] raw 예보(min IVT): AUC={ab:.3f} MCC={mb:.3f} F1={fb:.3f}")
    a0, m0, f0 = tpcomb(fcv.reshape(-1, 1), yf, odf)
    print(f"  우리모델(예보만)          : AUC={a0:.3f} MCC={m0:.3f} F1={f0:.3f} | vs기준 {m0 - mb:+.3f}")
    for hid in HIDS:
        ENCf = train_enc(XSEQ, y, pre, hid)[sel]
        a, m, f = tpcomb(np.column_stack([fcv, ENCf]), yf, odf)
        print(f"  우리모델(예보+LSTM인코더 hid={hid:2d}): AUC={a:.3f} MCC={m:.3f} F1={f:.3f} | vs기준 {m - mb:+.3f}")
