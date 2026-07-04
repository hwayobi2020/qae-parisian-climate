# ===== Colab: 전이 검증 — env(전체 온셋 학습) OOF -> 예보와 결합. TabPFN & LSTM. MCC. =====
# 준비: transfer_{ca,uk,chile}.npz 를 Colab에 (git pull). tabpfn 설치 셀에서: !pip install tabpfn -q
import numpy as np, torch, torch.nn as nn
from tabpfn import TabPFNClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, matthews_corrcoef, f1_score
DEV = "cuda" if torch.cuda.is_available() else "cpu"; FEAT_DIR = ""


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


class LSTMc(nn.Module):
    def __init__(s, cin, hid=48):
        super().__init__(); s.l = nn.LSTM(cin, hid, batch_first=True, bidirectional=True); s.d = nn.Dropout(0.3); s.h = nn.Linear(hid * 2, 1)
    def forward(s, x):
        o, _ = s.l(x); return s.h(s.d(o.mean(1))).squeeze(-1)


def env_oof(kind, FE, XSEQ, y, oday):
    N = len(y); pred = np.full(N, np.nan); rng = np.random.default_rng(0)
    for tr, te in folds(N, oday):
        if len(tr) < 60 or len(np.unique(y[tr])) < 2: continue
        if kind == "tabpfn":
            m = TabPFNClassifier(device=DEV); m.fit(np.nan_to_num(FE[tr]), y[tr]); pred[te] = m.predict_proba(np.nan_to_num(FE[te]))[:, 1]
        else:
            cmu = XSEQ[tr].reshape(-1, XSEQ.shape[-1]).mean(0); csd = XSEQ[tr].reshape(-1, XSEQ.shape[-1]).std(0) + 1e-8
            Xtr = torch.tensor((XSEQ[tr] - cmu) / csd, dtype=torch.float32, device=DEV)
            Xte = torch.tensor((XSEQ[te] - cmu) / csd, dtype=torch.float32, device=DEV)
            yt = torch.tensor(y[tr], dtype=torch.float32, device=DEV)
            torch.manual_seed(0); net = LSTMc(XSEQ.shape[-1]).to(DEV)
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2); lf = nn.BCEWithLogitsLoss(); nt = len(tr)
            for ep in range(60):
                net.train(); perm = rng.permutation(nt)
                for b in range(0, nt, 64):
                    bi = torch.as_tensor(perm[b:b + 64], device=DEV)
                    opt.zero_grad(); lf(net(Xtr[bi]), yt[bi]).backward(); opt.step()
            net.eval()
            with torch.no_grad(): pred[te] = torch.sigmoid(net(Xte)).cpu().numpy()
    return pred


def combine(FC, envcol, yf, odf):
    cols = [FC] if envcol is None else [FC, envcol.reshape(-1, 1)]
    X = np.column_stack(cols); ytb = []; pb = []; pp = np.full(len(yf), np.nan)
    for tr, te in folds(len(yf), odf):
        if len(tr) < 40 or len(np.unique(yf[tr])) < 2: continue
        Xtr = np.nan_to_num(X[tr]); Xte = np.nan_to_num(X[te]); sc = StandardScaler().fit(Xtr)
        m = LogisticRegression(max_iter=3000, C=0.3, class_weight="balanced").fit(sc.transform(Xtr), yf[tr])
        ptr = m.predict_proba(sc.transform(Xtr))[:, 1]; pte = m.predict_proba(sc.transform(Xte))[:, 1]
        pp[te] = pte; th = bt(yf[tr], ptr); pb.extend((pte >= th).astype(int)); ytb.extend(yf[te])
    ok = ~np.isnan(pp); ytb = np.array(ytb); pb = np.array(pb)
    return roc_auc_score(yf[ok], pp[ok]), matthews_corrcoef(ytb, pb), f1_score(ytb, pb, zero_division=0)


for r in ["ca", "uk", "chile"]:
    d = np.load(FEAT_DIR + f"transfer_{r}.npz")
    FE, XSEQ, FC, y, oday, fmask = d["FE"], d["XSEQ"], d["FC"], d["y"], d["oday"], d["fmask"].astype(bool)
    etp = env_oof("tabpfn", FE, XSEQ, y, oday)
    els = env_oof("lstm", FE, XSEQ, y, oday)
    sel = fmask; FCf = FC[sel]; yf = y[sel]; odf = oday[sel]
    print(f"[{r}] 예보온셋 {int(sel.sum())} 양성 {int(yf.sum())} | AUC / MCC / F1")
    for tag, col in [("예보단독", None), ("예보+env(TabPFN)", etp[sel]), ("예보+env(LSTM)", els[sel])]:
        a, m, f = combine(FCf, col, yf, odf)
        print(f"  {tag:20s}: AUC={a:.3f} MCC={m:.3f} F1={f:.3f}")
