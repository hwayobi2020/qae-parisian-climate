"""6h 정의, 시간순 확장창 K-fold CV + OOF 풀링 — "강도 anchored 증분 ablation".

핵심 질문(리뷰어 방어): "그냥 onset 강도만 알면 되는 것 아니냐?"
→ onset 강도(IVT)를 baseline으로 깔고, 거기에 환경 채널이 얼마나 더 보태는지(ΔAUC)를 측정한다.
   기여 = 절대 AUC가 아니라 "강도 위 증분".

피처셋 (bootstrap_6h_region.py 의 Fa/Fb/Fc/Fd 구조 그대로):
  A intensity      : [dmax]                                  (onset IVT 강도만)
  B +IVT-temporal  : IVT 16일 wavelet + [dmax]               (강도 + IVT 시간구조)
  C +circulation   : [dmax, jet] + jet wavelet + blk wavelet (강도 + 순환/제트 상태)
  D full           : 위 전부
  → Δwav = B−A, Δcirc = C−A, Δfull = D−A  (강도 위 증분)

분류기 (main model 축): Logistic / TabPFN(v2) / LGBM
CV: 확장창(expanding-window) walk-forward 5-fold, embargo 64일(최장 피처 창), train-only 표준화(Logistic).
평가: 모든 폴드 OOF 예측을 풀링 → 피처셋별 pooled AUC + 강도 대비 ΔAUC 의 paired 층화 부트스트랩 95% CI, P(Δ>0).

TabPFN 버전: 기본 v2(Apache, 상업가능). 환경변수 TABPFN_MODEL 로 override.
사용: python cv_6h_region.py [ca|uk|chile]   (TABPFN_TOKEN 필요)
"""
import os, sys, numpy as np, pywt, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

REGION = sys.argv[1] if len(sys.argv) > 1 else "uk"
N_SPLITS = 5
EMBARGO_DAYS = 64
N_BOOT = 1000
SEED = 0
HORIZONS = [4, 6, 8]    # 24h, 36h, 48h
TPMODEL = os.environ.get("TABPFN_MODEL", "tabpfn-v2-classifier-v2_default.ckpt")  # 기본 v2

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

def wl(a, lvl=5): return [c[-1] for c in pywt.swt(a, 'db2', level=lvl, trim_approx=True, norm=True)]

# --- 이벤트(연속 6h AR run) 추출 + 강도-앵커 피처셋 4종 + 라벨 + onset일 ---
runs = []; i = 0
while i < T:
    if ar6[i]:
        j = i
        while j < T and ar6[j]: j += 1
        runs.append((i, j)); i = j
    else: i += 1
FA, FB, FC, FD = [], [], [], []; steps = []; onset_day = []
for s, e in runs:
    o = s // 4
    if s - 63 >= 0 and o - 63 >= 0 and o < ND and not (np.isnan(jet[o]) or np.isnan(blk[o])):
        ivw = wl(ivt[s - 63:s + 1]); jw = wl(jet[o - 63:o + 1]); bw = wl(blk[o - 63:o + 1])
        FA.append([dmax[o]])
        FB.append(ivw + [dmax[o]])
        FC.append([dmax[o], jet[o]] + jw + bw)
        FD.append(ivw + [dmax[o], jet[o]] + jw + bw)
        steps.append(e - s); onset_day.append(o)
FA, FB, FC, FD = map(np.array, (FA, FB, FC, FD))
steps = np.array(steps); onset_day = np.array(onset_day); n = len(steps)
FS = {"A": FA, "B": FB, "C": FC, "D": FD}
FS_LABEL = {"A": "intensity     ", "B": "+IVT-temporal ", "C": "+circulation  ", "D": "full          "}
CLFS = ["Logistic", "TabPFN", "LGBM"]

from tabpfn import TabPFNClassifier
try:
    from lightgbm import LGBMClassifier; HAS_LGBM = True
except Exception:
    from sklearn.ensemble import HistGradientBoostingClassifier; HAS_LGBM = False

def fit_predict(clf, Xtr, ytr, Xte):
    if clf == "Logistic":
        sc = StandardScaler().fit(Xtr)
        m = LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced").fit(sc.transform(Xtr), ytr)
        return m.predict_proba(sc.transform(Xte))[:, 1]
    if clf == "TabPFN":
        kw = {} if TPMODEL == "auto" else {"model_path": TPMODEL}
        m = TabPFNClassifier(random_state=0, ignore_pretraining_limits=True, **kw).fit(Xtr, ytr)
        return m.predict_proba(Xte)[:, 1]
    # LGBM (없으면 sklearn HGB로 대체, 동일 규제 수준)
    if HAS_LGBM:
        m = LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=4, max_depth=2,
                           min_child_samples=6, reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8,
                           class_weight="balanced", verbose=-1).fit(Xtr, ytr)
    else:
        m = HistGradientBoostingClassifier(max_depth=2, max_iter=100, learning_rate=0.05,
                                           l2_regularization=1.0, min_samples_leaf=6,
                                           class_weight="balanced").fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]

def make_folds():
    fold = n // (N_SPLITS + 1); out = []
    for k in range(1, N_SPLITS + 1):
        ts = k * fold; te = (k + 1) * fold if k < N_SPLITS else n
        cutoff = onset_day[ts] - EMBARGO_DAYS
        train_idx = np.array([j for j in range(0, ts) if onset_day[j] <= cutoff])
        out.append((train_idx, np.arange(ts, te)))
    return out

def paired_boot(y, preds, n_boot=N_BOOT, seed=SEED):
    """같은 리샘플에서 A/B/C/D AUC 동시 계산(paired) → 부트스트랩 분포 반환."""
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    dist = {k: [] for k in preds}
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
        yy = y[idx]
        for k in preds: dist[k].append(roc_auc_score(yy, preds[k][idx]))
    return {k: np.array(v) for k, v in dist.items()}

folds = make_folds()
tpver = "auto(v3)" if TPMODEL == "auto" else TPMODEL
print(f"\n[{REGION}] 6h n={n}, AR임계(85th)={THR:.0f} | 확장창 {N_SPLITS}-fold CV, embargo {EMBARGO_DAYS}d, "
      f"TabPFN={tpver}, LGBM={'lightgbm' if HAS_LGBM else 'sklearn-HGB'}, 강도-앵커 증분 ablation")
for fi, (tr, te) in enumerate(folds):
    print(f"  fold{fi+1}: train n={len(tr)} (onset {onset_day[tr[0]]}~{onset_day[tr[-1]]}), "
          f"test n={len(te)} (onset {onset_day[te[0]]}~{onset_day[te[-1]]})")

for k in HORIZONS:
    y = (steps >= k).astype(int)
    valid = [(tr, te) for (tr, te) in folds if len(tr) >= 30 and len(np.unique(y[tr])) > 1]
    used = np.zeros(n, bool)
    for (_, te) in valid: used[te] = True
    yk = y[used]
    print(f"\n[{k*6}h] OOF 양성={int(yk.sum())}/{len(yk)}")
    if len(np.unique(yk)) < 2:
        print("  OOF 클래스부족"); continue
    # OOF 채우기: (분류기 x 피처셋)
    oof = {c: {f: np.full(n, np.nan) for f in FS} for c in CLFS}
    for (tr, te) in valid:
        for f, X in FS.items():
            for c in CLFS:
                oof[c][f][te] = fit_predict(c, X[tr], y[tr], X[te])
    for c in CLFS:
        preds = {f: oof[c][f][used] for f in FS}
        dist = paired_boot(yk, preds)
        aA = roc_auc_score(yk, preds["A"])
        loA, hiA = np.percentile(dist["A"], [2.5, 97.5])
        print(f"  [{c}]  A intensity AUC={aA:.3f} 95%CI[{loA:.3f},{hiA:.3f}]")
        for f in ["B", "C", "D"]:
            aX = roc_auc_score(yk, preds[f]); d = dist[f] - dist["A"]
            lo, hi = np.percentile(d, [2.5, 97.5]); pg = (d > 0).mean()
            flag = "*유의*" if pg >= 0.975 else ("동급" if pg > 0.5 else "효과없음")
            print(f"        {FS_LABEL[f]} AUC={aX:.3f}  Δ(강도위)={np.median(d):+.3f} "
                  f"95%CI[{lo:+.3f},{hi:+.3f}] P(>0)={pg:.2f} {flag}")
