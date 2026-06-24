"""6h 정의, 시간순 확장창 K-fold 교차검증 + OOF 풀링 AUC (encoder=lstm, main=TabPFN).

목적: 단일 60/20 홀드아웃이 긴 horizon에서 test 양성을 6개까지 떨어뜨려 AUC가 불안정한 문제를,
확장창(expanding-window) 시간순 K-fold로 전체 이벤트를 out-of-sample 채점해 해결한다.
- 각 폴드: train = 그 폴드 test 블록 "이전" 이벤트 전부(확장창), test = 해당 블록.
- 누수 방지: train과 test 사이 EMBARGO_DAYS(=최장 피처 창 64일) 캘린더 갭으로 train 꼬리를 잘라낸다.
- lstm 인코더는 폴드마다 재학습하며, 각 폴드 train 꼬리 20%를 내부 val로 early-stopping.
- 표준화는 폴드별 train 통계로만(train-only). TabPFN은 lstm penultimate hidden + onset 스칼라(raw)로 학습.
- 모든 폴드의 OOF 예측을 모아(pooled) horizon별 AUC 1개 + 층화 부트스트랩 95% CI.
- 검증: 폴드별 AUC도 같이 출력해 pooled ≈ 폴드평균 인지(=풀링 점수 비교가능성 가정) 확인.

사용: python cv_6h_region.py [ca|uk|chile]   (TABPFN_TOKEN 필요)
"""
import torch, torch.nn as nn, copy
import os, sys, numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score

REGION = sys.argv[1] if len(sys.argv) > 1 else "uk"
N_SPLITS = 5            # 시간순 폴드 수 (확장창)
EMBARGO_DAYS = 64       # 최장 피처 창(jet/blocking 64일)과 동일한 train/test 갭
VAL_FRAC = 0.20         # 폴드 train 꼬리 중 내부 val 비율(early-stopping)
N_BOOT = 1000           # 층화 부트스트랩 반복
SEED = 0
HORIZONS = [4, 6, 8]    # 24h, 36h, 48h  (steps 는 6h 단위)

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

# --- 이벤트(연속 6h AR run) 추출 + 시계열 창/onset 스칼라/라벨/onset일 ---
runs = []; i = 0
while i < T:
    if ar6[i]:
        j = i
        while j < T and ar6[j]: j += 1
        runs.append((i, j)); i = j
    else: i += 1
IVTw = []; JETw = []; BLKw = []; base = []; steps = []; onset_day = []
for s, e in runs:
    o = s // 4
    if s - 63 >= 0 and o - 63 >= 0 and o < ND and not (np.isnan(jet[o]) or np.isnan(blk[o])):
        IVTw.append(ivt[s - 63:s + 1]); JETw.append(jet[o - 63:o + 1]); BLKw.append(blk[o - 63:o + 1])
        base.append([dmax[o], jet[o]]); steps.append(e - s); onset_day.append(o)
IVTw, JETw, BLKw, base, steps, onset_day = map(np.array, (IVTw, JETw, BLKw, base, steps, onset_day))
n = len(steps)
Xseq_all = np.stack([IVTw, JETw, BLKw], axis=2)   # (n, 64, 3)  lstm: (batch, seq, feat)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

class LSTMenc(nn.Module):
    def __init__(s): super().__init__(); s.lstm = nn.LSTM(3, 32, batch_first=True)
    def forward(s, x): o, _ = s.lstm(x); return o[:, -1]      # (n, 32)

def lstm_hidden(tr_core, tr_val, y):
    """tr_core 로 lstm+head 학습, tr_val AUC 로 early-stop. 전체 이벤트 hidden(n,32) 반환."""
    mu = Xseq_all[tr_core].mean((0, 1), keepdims=True); sd = Xseq_all[tr_core].std((0, 1), keepdims=True) + 1e-8
    Xt = torch.tensor((Xseq_all - mu) / sd, dtype=torch.float32, device=DEV)
    bmu = base[tr_core].mean(0); bsd = base[tr_core].std(0) + 1e-8
    bt = torch.tensor((base - bmu) / bsd, dtype=torch.float32, device=DEV)
    yt = torch.tensor(y, dtype=torch.float32, device=DEV)
    enc = LSTMenc().to(DEV); head = nn.Linear(34, 1).to(DEV)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=2e-3, weight_decay=1e-3)
    npos = max(int(y[tr_core].sum()), 1)
    pw = torch.tensor((len(tr_core) - y[tr_core].sum()) / npos, dtype=torch.float32, device=DEV)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    tc = torch.tensor(tr_core, dtype=torch.long, device=DEV)
    def fwd(idx_t): return head(torch.cat([enc(Xt[idx_t]), bt[idx_t]], 1)).squeeze(-1)
    can_val = len(tr_val) > 0 and len(np.unique(y[tr_val])) > 1
    best = -1; bs = None
    for ep in range(200):
        enc.train(); head.train(); opt.zero_grad(); lossf(fwd(tc), yt[tc]).backward(); opt.step()
        if can_val and ep % 5 == 0:
            enc.eval(); head.eval()
            with torch.no_grad():
                pv = torch.sigmoid(fwd(torch.tensor(tr_val, dtype=torch.long, device=DEV))).cpu().numpy()
            v = roc_auc_score(y[tr_val], pv)
            if v > best: best = v; bs = (copy.deepcopy(enc.state_dict()), copy.deepcopy(head.state_dict()))
    if bs is not None: enc.load_state_dict(bs[0]); head.load_state_dict(bs[1])
    enc.eval()
    with torch.no_grad(): H = enc(Xt).cpu().numpy()
    return H

from tabpfn import TabPFNClassifier

def make_folds():
    """확장창: test 블록 N_SPLITS개. 각 폴드 train = test 시작 이전 + EMBARGO 캘린더 갭."""
    fold = n // (N_SPLITS + 1)
    out = []
    for k in range(1, N_SPLITS + 1):
        ts = k * fold
        te = (k + 1) * fold if k < N_SPLITS else n
        test_idx = np.arange(ts, te)
        cutoff = onset_day[ts] - EMBARGO_DAYS
        train_idx = np.array([j for j in range(0, ts) if onset_day[j] <= cutoff])
        out.append((train_idx, test_idx))
    return out

def strat_boot_ci(y, p, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    if len(pos) < 1 or len(neg) < 1: return (float("nan"), float("nan"))
    aucs = []
    for _ in range(n_boot):
        bp = rng.choice(pos, len(pos), replace=True); bn = rng.choice(neg, len(neg), replace=True)
        idx = np.concatenate([bp, bn]); aucs.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))

folds = make_folds()
print(f"\n[{REGION}] 6h n={n}, AR임계(85th)={THR:.0f} | 확장창 {N_SPLITS}-fold CV, embargo {EMBARGO_DAYS}d, "
      f"encoder=lstm, main=TabPFN, OOF 풀링 + 부트스트랩 CI(B={N_BOOT})")
for fi, (tr, te) in enumerate(folds):
    print(f"  fold{fi+1}: train n={len(tr)} (onset일 {onset_day[tr[0]]}~{onset_day[tr[-1]]}), "
          f"test n={len(te)} (onset일 {onset_day[te[0]]}~{onset_day[te[-1]]})")

for k in HORIZONS:
    y = (steps >= k).astype(int)
    oof_p = np.full(n, np.nan); oof_used = np.zeros(n, dtype=bool)
    fold_aucs = []
    for (tr, te) in folds:
        if len(tr) < 30 or len(np.unique(y[tr])) < 2:    # 학습 불가 폴드는 건너뜀
            continue
        cut = int(len(tr) * (1 - VAL_FRAC))
        tr_core, tr_val = tr[:cut], tr[cut:]
        if len(np.unique(y[tr_core])) < 2: tr_core, tr_val = tr, np.array([], dtype=int)
        H = lstm_hidden(tr_core, tr_val, y)
        feat = np.hstack([H, base])                       # hidden + onset 스칼라(raw)
        m = TabPFNClassifier(random_state=0, ignore_pretraining_limits=True).fit(feat[tr_core], y[tr_core])
        pte = m.predict_proba(feat[te])[:, 1]
        oof_p[te] = pte; oof_used[te] = True
        if len(np.unique(y[te])) > 1:
            fold_aucs.append(roc_auc_score(y[te], pte))
    mask = oof_used & ~np.isnan(oof_p)
    yp, pp = y[mask], oof_p[mask]
    if len(np.unique(yp)) < 2:
        print(f"\n  [{k*6}h] OOF 클래스부족 (양성={int(yp.sum())}/{len(yp)})"); continue
    pooled = roc_auc_score(yp, pp); lo, hi = strat_boot_ci(yp, pp)
    fa = np.array(fold_aucs)
    favg = fa.mean() if len(fa) else float("nan")
    print(f"\n  [{k*6}h] OOF 양성={int(yp.sum())}/{len(yp)} (단일홀드아웃 대비 양성 확대)")
    print(f"    pooled AUC = {pooled:.3f}  95%CI[{lo:.3f}, {hi:.3f}]")
    print(f"    폴드별 AUC = [{', '.join(f'{a:.3f}' for a in fa)}]  평균 {favg:.3f}  "
          f"(pooled≈평균이면 풀링 비교가능성 OK)")
