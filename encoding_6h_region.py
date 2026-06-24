"""6시간 정의, 시계열 인코딩/분류기 ablation.
축1 인코딩: wavelet vs PCA vs raw-downsample (분류기 TabPFN 고정)
축2 분류기: TabPFN vs MLP vs Logistic (입력 wavelet 고정)
구간은 동일(IVT 16일, 제트/블로킹 64일), onset 직전 연속 구간.
사용: python encoding_6h_region.py [ca|uk|chile]   (TABPFN_TOKEN 필요)"""
import torch
import os, sys, numpy as np, pywt, warnings; warnings.filterwarnings("ignore")
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
REGION = sys.argv[1] if len(sys.argv) > 1 else "ca"
IVT_FILE = {"ca": "ivt_sf_1980_2023.npy"}.get(REGION, f"ivt_{REGION}_1980_2023.npy")
CIRC_FILE = {"ca": "circ_indices.npz"}.get(REGION, f"circ_indices_{REGION}.npz")
def find(name):
    for p in [os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw", name),
              os.path.join("data/raw", name), "/content/" + name,
              os.path.join("/content/qae-parisian-climate/data/raw", name)]:
        if os.path.exists(p): return p
    raise FileNotFoundError(name)
ivt = np.load(find(IVT_FILE)).astype("float64")
ci = np.load(find(CIRC_FILE)); jet = ci["jet"].astype("float64"); blk = ci["blocking"].astype("float64")
dmax = ivt.reshape(-1, 4).max(1); ND = len(dmax); THR = np.percentile(dmax, 85)
T = len(ivt); ar6 = ivt > THR
runs = []; i = 0
while i < T:
    if ar6[i]:
        j = i
        while j < T and ar6[j]: j += 1
        runs.append((i, j)); i = j
    else: i += 1
IVTw = []; JETw = []; BLKw = []; base = []; steps = []
for s, e in runs:
    o = s // 4
    if s - 63 >= 0 and o - 63 >= 0 and o < ND and not (np.isnan(jet[o]) or np.isnan(blk[o])):
        IVTw.append(ivt[s - 63:s + 1]); JETw.append(jet[o - 63:o + 1]); BLKw.append(blk[o - 63:o + 1])
        base.append([dmax[o], jet[o]]); steps.append(e - s)
IVTw, JETw, BLKw, base, steps = map(np.array, (IVTw, JETw, BLKw, base, steps))
n = len(steps); i1 = int(n * 0.6); i2 = int(n * 0.8)
def wavenc(W): return np.array([[c[-1] for c in pywt.swt(w, 'db2', level=5, trim_approx=True, norm=True)] for w in W])
def pcaenc(W): p = PCA(6).fit(W[:i1]); return p.transform(W)
def rawenc(W): idx = np.linspace(0, W.shape[1] - 1, 6).astype(int); return W[:, idx]
def build(enc): return np.hstack([base, enc(IVTw), enc(JETw), enc(BLKw)])
Xwav, Xpca, Xraw = build(wavenc), build(pcaenc), build(rawenc)
from tabpfn import TabPFNClassifier
def auc_tp(X, y):
    m = TabPFNClassifier(random_state=0, ignore_pretraining_limits=True).fit(X[:i1], y[:i1])
    return roc_auc_score(y[i2:], m.predict_proba(X[i2:])[:, 1])
def auc_mlp(X, y):
    sc = StandardScaler().fit(X[:i1])
    m = MLPClassifier((64,), max_iter=500, early_stopping=True, random_state=0).fit(sc.transform(X[:i1]), y[:i1])
    return roc_auc_score(y[i2:], m.predict_proba(sc.transform(X[i2:]))[:, 1])
def auc_lr(X, y):
    sc = StandardScaler().fit(X[:i1])
    m = LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced").fit(sc.transform(X[:i1]), y[:i1])
    return roc_auc_score(y[i2:], m.predict_proba(sc.transform(X[i2:]))[:, 1])
import torch.nn as nn, copy
def auc_cnn(y):    # raw 3채널 시계열(IVT/제트/블로킹) -> 1D CNN end-to-end
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X = np.stack([IVTw, JETw, BLKw], 1)                       # (n,3,64)
    mu = X[:i1].mean((0, 2), keepdims=True); sd = X[:i1].std((0, 2), keepdims=True) + 1e-8
    X = (X - mu) / sd
    b = (base - base[:i1].mean(0)) / (base[:i1].std(0) + 1e-8)
    Xt = torch.tensor(X, dtype=torch.float32, device=dev); bt = torch.tensor(b, dtype=torch.float32, device=dev)
    yt = torch.tensor(y, dtype=torch.float32, device=dev)
    class CNN(nn.Module):
        def __init__(s):
            super().__init__()
            s.c1 = nn.Conv1d(3, 16, 5, padding=2); s.c2 = nn.Conv1d(16, 32, 5, padding=2)
            s.fc = nn.Sequential(nn.Linear(34, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1))
        def forward(s, x, bb):
            h = torch.relu(s.c1(x)); h = torch.relu(s.c2(h)).mean(-1)
            return s.fc(torch.cat([h, bb], 1)).squeeze(-1)
    torch.manual_seed(0); net = CNN().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
    pw = torch.tensor((len(y[:i1]) - y[:i1].sum()) / max(y[:i1].sum(), 1), dtype=torch.float32, device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw); best = -1; bs = None
    for ep in range(200):
        net.train(); opt.zero_grad(); lossf(net(Xt[:i1], bt[:i1]), yt[:i1]).backward(); opt.step()
        if ep % 5 == 0 and len(np.unique(y[i1:i2])) > 1:
            net.eval()
            with torch.no_grad(): pv = torch.sigmoid(net(Xt[i1:i2], bt[i1:i2])).cpu().numpy()
            v = roc_auc_score(y[i1:i2], pv)
            if v > best: best = v; bs = copy.deepcopy(net.state_dict())
    if bs is not None: net.load_state_dict(bs)
    net.eval()
    with torch.no_grad(): pt = torch.sigmoid(net(Xt[i2:], bt[i2:])).cpu().numpy()
    return roc_auc_score(y[i2:], pt)
def auc_lstm(y):    # raw 3채널 시계열 -> LSTM 인코더 end-to-end
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X = np.stack([IVTw, JETw, BLKw], 2)                       # (n,64,3) seq,channel
    mu = X[:i1].mean((0, 1), keepdims=True); sd = X[:i1].std((0, 1), keepdims=True) + 1e-8
    X = (X - mu) / sd
    b = (base - base[:i1].mean(0)) / (base[:i1].std(0) + 1e-8)
    Xt = torch.tensor(X, dtype=torch.float32, device=dev); bt = torch.tensor(b, dtype=torch.float32, device=dev)
    yt = torch.tensor(y, dtype=torch.float32, device=dev)
    class L(nn.Module):
        def __init__(s):
            super().__init__(); s.lstm = nn.LSTM(3, 32, batch_first=True); s.fc = nn.Linear(34, 1)
        def forward(s, x, bb):
            o, _ = s.lstm(x); return s.fc(torch.cat([o[:, -1], bb], 1)).squeeze(-1)
    torch.manual_seed(0); net = L().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
    pw = torch.tensor((len(y[:i1]) - y[:i1].sum()) / max(y[:i1].sum(), 1), dtype=torch.float32, device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw); best = -1; bs = None
    for ep in range(200):
        net.train(); opt.zero_grad(); lossf(net(Xt[:i1], bt[:i1]), yt[:i1]).backward(); opt.step()
        if ep % 5 == 0 and len(np.unique(y[i1:i2])) > 1:
            net.eval()
            with torch.no_grad(): pv = torch.sigmoid(net(Xt[i1:i2], bt[i1:i2])).cpu().numpy()
            v = roc_auc_score(y[i1:i2], pv)
            if v > best: best = v; bs = copy.deepcopy(net.state_dict())
    if bs is not None: net.load_state_dict(bs)
    net.eval()
    with torch.no_grad(): pt = torch.sigmoid(net(Xt[i2:], bt[i2:])).cpu().numpy()
    return roc_auc_score(y[i2:], pt)
def auc_mlp_raw(y):    # raw 시계열 flatten -> MLP (순서 무시)
    X = np.hstack([np.stack([IVTw, JETw, BLKw], 1).reshape(len(steps), -1), base])
    sc = StandardScaler().fit(X[:i1])
    m = MLPClassifier((64,), max_iter=500, early_stopping=True, random_state=0).fit(sc.transform(X[:i1]), y[:i1])
    return roc_auc_score(y[i2:], m.predict_proba(sc.transform(X[i2:]))[:, 1])
print(f"\n[{REGION}] 6h n={n}, AR임계(85th)={THR:.0f}")
print(f"\n  === 축1 인코딩 (분류기 TabPFN 고정) ===")
print(f"  {'지속':>5} {'양성%':>5} | {'wavelet':>8} {'PCA':>7} {'raw':>7} | {'test양성':>6}")
for k in [4, 6, 8]:
    y = (steps >= k).astype(int)
    if len(np.unique(y[i2:])) < 2: print(f"  {k*6}h 클래스부족"); continue
    print(f"  {k*6:>4}h {y.mean()*100:>4.0f}% | {auc_tp(Xwav,y):>8.3f} {auc_tp(Xpca,y):>7.3f} {auc_tp(Xraw,y):>7.3f} | {int(y[i2:].sum()):>6}")
print(f"\n  === 축2 방법 (wavelet+TabPFN/MLP/Logistic vs raw+CNN end-to-end) ===")
print(f"  {'지속':>5} {'양성%':>5} | {'wav+TabPFN':>10} {'wav+MLP':>8} {'wav+LR':>7} {'raw+CNN':>8} | {'test양성':>6}")
for k in [4, 6, 8]:
    y = (steps >= k).astype(int)
    if len(np.unique(y[i2:])) < 2: print(f"  {k*6}h 클래스부족"); continue
    print(f"  {k*6:>4}h {y.mean()*100:>4.0f}% | {auc_tp(Xwav,y):>10.3f} {auc_mlp(Xwav,y):>8.3f} {auc_lr(Xwav,y):>7.3f} {auc_cnn(y):>8.3f} | {int(y[i2:].sum()):>6}")
print(f"\n  === 축3 학습형 인코더 (raw 시계열 end-to-end): wavelet+TabPFN vs CNN/LSTM/MLP ===")
print(f"  {'지속':>5} {'양성%':>5} | {'wav+TabPFN':>10} {'raw+CNN':>8} {'raw+LSTM':>9} {'raw+MLP':>8} | {'test양성':>6}")
for k in [4, 6, 8]:
    y = (steps >= k).astype(int)
    if len(np.unique(y[i2:])) < 2: print(f"  {k*6}h 클래스부족"); continue
    print(f"  {k*6:>4}h {y.mean()*100:>4.0f}% | {auc_tp(Xwav,y):>10.3f} {auc_cnn(y):>8.3f} {auc_lstm(y):>9.3f} {auc_mlp_raw(y):>8.3f} | {int(y[i2:].sum()):>6}")
