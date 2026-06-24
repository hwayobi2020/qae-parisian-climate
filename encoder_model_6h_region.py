"""6h 정의, 2D ablation: encoder x main-model.
encoder: raw / wavelet (비학습) + mlp / cnn / lstm (학습형, end-to-end 학습 후 penultimate hidden 추출)
main model: Logistic / TabPFN
모든 인코더 -> 시계열을 고정 표현으로 -> 같은 두 분류기로 비교.
사용: python encoder_model_6h_region.py [ca|uk|chile]   (TABPFN_TOKEN)"""
import torch, torch.nn as nn, copy
import os, sys, numpy as np, pywt, warnings; warnings.filterwarnings("ignore")
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
# --- 비학습 인코더 표현 (임계 무관) ---
def wavenc(W): return np.array([[c[-1] for c in pywt.swt(w, 'db2', level=5, trim_approx=True, norm=True)] for w in W])
def rawenc(W): idx = np.linspace(0, W.shape[1] - 1, 6).astype(int); return W[:, idx]
def build(enc): return np.hstack([base, enc(IVTw), enc(JETw), enc(BLKw)])
R_raw, R_wav = build(rawenc), build(wavenc)
# --- 학습형 인코더 (시계열 -> hidden 32) ---
bt_np = (base - base[:i1].mean(0)) / (base[:i1].std(0) + 1e-8)
def _prep(seq_first):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X = np.stack([IVTw, JETw, BLKw], 2 if seq_first else 1)
    ax = (0, 1) if seq_first else (0, 2)
    mu = X[:i1].mean(ax, keepdims=True); sd = X[:i1].std(ax, keepdims=True) + 1e-8
    return torch.tensor((X - mu) / sd, dtype=torch.float32, device=dev), dev
class CNNenc(nn.Module):
    def __init__(s): super().__init__(); s.c1 = nn.Conv1d(3, 16, 5, padding=2); s.c2 = nn.Conv1d(16, 32, 5, padding=2)
    def forward(s, x): return torch.relu(s.c2(torch.relu(s.c1(x)))).mean(-1)        # (n,32)
class LSTMenc(nn.Module):
    def __init__(s): super().__init__(); s.lstm = nn.LSTM(3, 32, batch_first=True)
    def forward(s, x): o, _ = s.lstm(x); return o[:, -1]                            # (n,32)
class MLPenc(nn.Module):
    def __init__(s): super().__init__(); s.net = nn.Sequential(nn.Linear(192, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 32))
    def forward(s, x): return s.net(x.reshape(x.shape[0], -1))                      # (n,3,64)->192->32
def deep_repr(EncClass, seq_first, y):
    Xt, dev = _prep(seq_first); bt = torch.tensor(bt_np, dtype=torch.float32, device=dev); yt = torch.tensor(y, dtype=torch.float32, device=dev)
    enc = EncClass().to(dev); head = nn.Linear(34, 1).to(dev)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=2e-3, weight_decay=1e-3)
    pw = torch.tensor((len(y[:i1]) - y[:i1].sum()) / max(y[:i1].sum(), 1), dtype=torch.float32, device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    def fwd(sl): return head(torch.cat([enc(Xt[sl]), bt[sl]], 1)).squeeze(-1)
    best = -1; bs = None
    for ep in range(200):
        enc.train(); head.train(); opt.zero_grad(); lossf(fwd(slice(0, i1)), yt[:i1]).backward(); opt.step()
        if ep % 5 == 0 and len(np.unique(y[i1:i2])) > 1:
            enc.eval(); head.eval()
            with torch.no_grad(): pv = torch.sigmoid(fwd(slice(i1, i2))).cpu().numpy()
            v = roc_auc_score(y[i1:i2], pv)
            if v > best: best = v; bs = copy.deepcopy(enc.state_dict())
    if bs is not None: enc.load_state_dict(bs)
    enc.eval()
    with torch.no_grad(): H = enc(Xt).cpu().numpy()
    return np.hstack([H, base])           # penultimate hidden + onset 스칼라
from tabpfn import TabPFNClassifier
def auc_lr(X, y):
    sc = StandardScaler().fit(X[:i1])
    m = LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced").fit(sc.transform(X[:i1]), y[:i1])
    return roc_auc_score(y[i2:], m.predict_proba(sc.transform(X[i2:]))[:, 1])
def auc_tp(X, y):
    m = TabPFNClassifier(random_state=0, ignore_pretraining_limits=True).fit(X[:i1], y[:i1])
    return roc_auc_score(y[i2:], m.predict_proba(X[i2:])[:, 1])
print(f"\n[{REGION}] 6h n={n}, AR임계(85th)={THR:.0f}   2D ablation: encoder x main-model")
for k in [4, 6, 8]:
    y = (steps >= k).astype(int)
    if len(np.unique(y[i2:])) < 2: print(f"  {k*6}h 클래스부족"); continue
    reprs = {"raw": R_raw, "wavelet": R_wav,
             "mlp": deep_repr(MLPenc, False, y), "cnn": deep_repr(CNNenc, False, y), "lstm": deep_repr(LSTMenc, True, y)}
    print(f"\n  [{k*6}h] test양성={int(y[i2:].sum())}/{len(y[i2:])}")
    print(f"  {'encoder':>9} | {'Logistic':>9} {'TabPFN':>8}")
    for en in ["raw", "wavelet", "mlp", "cnn", "lstm"]:
        X = reprs[en]
        print(f"  {en:>9} | {auc_lr(X, y):>9.3f} {auc_tp(X, y):>8.3f}")
