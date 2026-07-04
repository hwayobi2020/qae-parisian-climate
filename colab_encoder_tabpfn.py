# ===== Colab: 기준=raw 예보. 인코더 입력 피처수 k 훑기(pre-2000 SelectKBest). MLP hid=8. leak-free. =====
# 준비: transfer_{ca,uk,chile}.npz (git pull). tabpfn 설치 셀: !pip install tabpfn -q
import numpy as np, torch, torch.nn as nn
from tabpfn import TabPFNClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import matthews_corrcoef
DEV = "cuda" if torch.cuda.is_available() else "cpu"; FEAT_DIR = ""; HID = 8; W = 16; KS = [8, 16, 24, 32, 44]


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


def train_enc(FE, y, pre, hid, k):
    FEz = np.nan_to_num(FE)
    if k < FE.shape[1]:
        sk = SelectKBest(f_classif, k=k).fit(FEz[pre], y[pre]); FEz = sk.transform(FEz)   # 선택은 pre-2000만
    mu = FEz[pre].mean(0); sd = FEz[pre].std(0) + 1e-8; FEs = (FEz - mu) / sd
    Xtr = torch.tensor(FEs[pre], dtype=torch.float32, device=DEV); yt = torch.tensor(y[pre], dtype=torch.float32, device=DEV)
    torch.manual_seed(0); net = MLPenc(FEz.shape[1], hid).to(DEV)
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


def base_raw(fcv, yf, odf):
    ytb = []; pb = []
    for tr, te in folds(len(yf), odf):
        if len(tr) < 40 or len(np.unique(yf[tr])) < 2: continue
        th = bt(yf[tr], fcv[tr]); pb.extend((fcv[te] >= th).astype(int)); ytb.extend(yf[te])
    return matthews_corrcoef(np.array(ytb), np.array(pb))


def tpcomb(X, yf, odf):
    ytb = []; pb = []
    for tr, te in folds(len(yf), odf):
        if len(tr) < 40 or len(np.unique(yf[tr])) < 2: continue
        m = TabPFNClassifier(device=DEV); m.fit(np.nan_to_num(X[tr]), yf[tr])
        ptr = m.predict_proba(np.nan_to_num(X[tr]))[:, 1]; pte = m.predict_proba(np.nan_to_num(X[te]))[:, 1]
        th = bt(yf[tr], ptr); pb.extend((pte >= th).astype(int)); ytb.extend(yf[te])
    return matthews_corrcoef(np.array(ytb), np.array(pb))


for r in ["ca", "uk", "chile"]:
    d = np.load(FEAT_DIR + f"transfer_{r}.npz")
    FE, FC, y, oday, fmask = d["FE"], d["FC"], d["y"], d["oday"], d["fmask"].astype(bool)
    pre = oday < oday[fmask].min()
    sel = fmask; FCf = FC[sel]; yf = y[sel]; odf = oday[sel]; fcv = FCf.mean(1)
    mb = base_raw(fcv, yf, odf)
    print(f"[{r}] 양성 {int(yf.sum())} | [기준] raw 예보 MCC={mb:.3f} | MLP hid={HID}, 피처수 k 훑기")
    for k in KS:
        ENCf = train_enc(FE, y, pre, HID, k)[sel]
        m = tpcomb(np.column_stack([fcv, ENCf]), yf, odf)
        print(f"  k={k:2d}: 예보+인코더 MCC={m:.3f} | vs기준 {m - mb:+.3f}")
