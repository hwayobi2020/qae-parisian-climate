"""6시간 정의, 인코더 비교 (시계열 -> 고정 표현).
비학습 인코더: wavelet / PCA / raw-downsample  (-> TabPFN 분류 고정)
학습형 인코더: CNN / LSTM / MLP                (raw 시계열 end-to-end)
구간 동일(IVT 16일, 제트/블로킹 64일). 분류기 비교(메인모델 ablation)는 별도 스크립트.
사용: python encoding_6h_region.py [ca|uk|chile]   (TABPFN_TOKEN)"""
import torch
import os, sys, numpy as np, pywt, copy, warnings; warnings.filterwarnings("ignore")
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
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
# --- 비학습 인코더: 시계열 -> 6계수 + onset 스칼라 ---
def wavenc(W): return np.array([[c[-1] for c in pywt.swt(w, 'db2', level=5, trim_approx=True, norm=True)] for w in W])
def pcaenc(W): return PCA(6).fit(W[:i1]).transform(W)
def rawenc(W): idx = np.linspace(0, W.shape[1] - 1, 6).astype(int); return W[:, idx]
def build(enc): return np.hstack([base, enc(IVTw), enc(JETw), enc(BLKw)])
Xwav, Xpca, Xraw = build(wavenc), build(pcaenc), build(rawenc)
from tabpfn import TabPFNClassifier
def auc_tp(X, y):
    m = TabPFNClassifier(random_state=0, ignore_pretraining_limits=True).fit(X[:i1], y[:i1])
    return roc_auc_score(y[i2:], m.predict_proba(X[i2:])[:, 1])
# --- 학습형 인코더: raw 3채널 시계열 end-to-end ---
def _train_torch(net, Xt, bt, yt):
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
    pw = torch.tensor((len(yt[:i1]) - yt[:i1].sum()) / max(yt[:i1].sum().item(), 1), dtype=torch.float32, device=yt.device)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw); best = -1; bs = None; yv = yt[i1:i2].cpu().numpy()
    for ep in range(200):
        net.train(); opt.zero_grad(); lossf(net(Xt[:i1], bt[:i1]), yt[:i1]).backward(); opt.step()
        if ep % 5 == 0 and len(np.unique(yv)) > 1:
            net.eval()
            with torch.no_grad(): pv = torch.sigmoid(net(Xt[i1:i2], bt[i1:i2])).cpu().numpy()
            v = roc_auc_score(yv, pv)
            if v > best: best = v; bs = copy.deepcopy(net.state_dict())
    if bs is not None: net.load_state_dict(bs)
    net.eval()
    with torch.no_grad(): return torch.sigmoid(net(Xt[i2:], bt[i2:])).cpu().numpy()
def _prep(seq_first):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X = np.stack([IVTw, JETw, BLKw], 2 if seq_first else 1)        # (n,64,3) or (n,3,64)
    ax = (0, 1) if seq_first else (0, 2)
    mu = X[:i1].mean(ax, keepdims=True); sd = X[:i1].std(ax, keepdims=True) + 1e-8
    X = (X - mu) / sd
    b = (base - base[:i1].mean(0)) / (base[:i1].std(0) + 1e-8)
    return (torch.tensor(X, dtype=torch.float32, device=dev), torch.tensor(b, dtype=torch.float32, device=dev), dev)
def auc_cnn(y):
    Xt, bt, dev = _prep(seq_first=False)
    yt = torch.tensor(y, dtype=torch.float32, device=dev)
    class M(nn.Module):
        def __init__(s):
            super().__init__(); s.c1 = nn.Conv1d(3, 16, 5, padding=2); s.c2 = nn.Conv1d(16, 32, 5, padding=2)
            s.fc = nn.Sequential(nn.Linear(34, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1))
        def forward(s, x, bb):
            h = torch.relu(s.c1(x)); h = torch.relu(s.c2(h)).mean(-1); return s.fc(torch.cat([h, bb], 1)).squeeze(-1)
    torch.manual_seed(0)
    return roc_auc_score(y[i2:], _train_torch(M().to(dev), Xt, bt, yt))
def auc_lstm(y):
    Xt, bt, dev = _prep(seq_first=True)
    yt = torch.tensor(y, dtype=torch.float32, device=dev)
    class M(nn.Module):
        def __init__(s):
            super().__init__(); s.lstm = nn.LSTM(3, 32, batch_first=True); s.fc = nn.Linear(34, 1)
        def forward(s, x, bb):
            o, _ = s.lstm(x); return s.fc(torch.cat([o[:, -1], bb], 1)).squeeze(-1)
    torch.manual_seed(0)
    return roc_auc_score(y[i2:], _train_torch(M().to(dev), Xt, bt, yt))
def auc_mlp(y):
    X = np.hstack([np.stack([IVTw, JETw, BLKw], 1).reshape(n, -1), base])
    sc = StandardScaler().fit(X[:i1])
    m = MLPClassifier((64,), max_iter=500, early_stopping=True, random_state=0).fit(sc.transform(X[:i1]), y[:i1])
    return roc_auc_score(y[i2:], m.predict_proba(sc.transform(X[i2:]))[:, 1])
print(f"\n[{REGION}] 6h n={n}, AR임계(85th)={THR:.0f}  인코더 비교")
print(f"  {'지속':>5} {'양성%':>5} | {'wavelet':>8} {'PCA':>7} {'raw':>7} | {'CNN':>7} {'LSTM':>7} {'MLP':>7} | {'test양성':>6}")
print(f"  {'':5} {'':5} |   ----- 비학습 +TabPFN -----   |  ---- 학습형 end-to-end ----  |")
for k in [4, 6, 8]:
    y = (steps >= k).astype(int)
    if len(np.unique(y[i2:])) < 2: print(f"  {k*6}h 클래스부족"); continue
    print(f"  {k*6:>4}h {y.mean()*100:>4.0f}% | {auc_tp(Xwav,y):>8.3f} {auc_tp(Xpca,y):>7.3f} {auc_tp(Xraw,y):>7.3f} | "
          f"{auc_cnn(y):>7.3f} {auc_lstm(y):>7.3f} {auc_mlp(y):>7.3f} | {int(y[i2:].sum()):>6}")
