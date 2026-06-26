"""6h 정의, 시간순 확장창 CV — "강도(intensity)" 변형 비교 전용.

질문: UK 24h 의 예측력(~0.72)이 (1) 진짜 onset 예측인가 (2) intra-day look-ahead/동어반복인가
      (3) 단일 점인가 강도의 흐름(궤적)인가.

강도 피처 4종 (모두 IVT만, 환경 채널 제외):
  peak_LA   : [dmax[o]]                = onset '당일' 최대  → onset 이후 같은 날 시간 포함(LOOK-AHEAD)
  onset_pt  : [ivt[s]]                 = onset 순간 값      → 엄격 causal, 단일 점
  peak_rec  : [max(ivt[s-3:s+1])]      = onset 직전 24h 최대 → 엄격 causal, 단일 '최근 수준'
  flow6     : ivt 6점 over [s-63:s+1]  = 16일 궤적(6점), onset까지 → 엄격 causal, '흐름'

읽는 법:
  - peak_LA vs peak_rec : LOOK-AHEAD 가 AUC 를 얼마나 부풀렸나 (둘 차이 = 당일 엿보기 효과)
  - flow6 vs onset_pt   : 강도의 '흐름'이 단일 점보다 나은가
  - causal 버전(onset_pt/peak_rec/flow6)이 ~0.7 유지 → 강도가 진짜 onset 예측 / ~0.5~0.6 붕괴 → artifact

분류기: Logistic / TabPFN(v2) / LGBM. CV: 확장창 5-fold, embargo 64d, OOF 풀링 + 층화 부트스트랩 CI.
사용: python cv_intensity_6h_region.py [ca|uk|chile]   (TABPFN_TOKEN 필요)
"""
import os, sys, numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

REGION = sys.argv[1] if len(sys.argv) > 1 else "uk"
N_SPLITS = 5
EMBARGO_DAYS = 64
N_BOOT = 1000
SEED = 0
HORIZONS = [4, 6, 8]
TPMODEL = os.environ.get("TABPFN_MODEL", "tabpfn-v2-classifier-v2_default.ckpt")

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

# 이벤트 추출 (jet/blk nan 필터는 기존 실행과 동일 이벤트셋 유지를 위해 그대로 둠)
runs = []; i = 0
while i < T:
    if ar6[i]:
        j = i
        while j < T and ar6[j]: j += 1
        runs.append((i, j)); i = j
    else: i += 1
F_la, F_pt, F_rec, F_fl, steps, onset_day = [], [], [], [], [], []
for s, e in runs:
    o = s // 4
    if s - 63 >= 0 and o - 63 >= 0 and o < ND and not (np.isnan(jet[o]) or np.isnan(blk[o])):
        idx6 = (s - 63 + np.linspace(0, 63, 6)).astype(int)
        F_la.append([dmax[o]])
        F_pt.append([ivt[s]])
        F_rec.append([ivt[s - 3:s + 1].max()])
        F_fl.append(list(ivt[idx6]))
        steps.append(e - s); onset_day.append(o)
F_la, F_pt, F_rec, F_fl = map(np.array, (F_la, F_pt, F_rec, F_fl))
steps = np.array(steps); onset_day = np.array(onset_day); n = len(steps)
FS = {"peak_LA": F_la, "onset_pt": F_pt, "peak_rec": F_rec, "flow6": F_fl}
FS_LABEL = {"peak_LA": "peak_LA (당일max,엿봄)", "onset_pt": "onset_pt (onset점)",
            "peak_rec": "peak_rec (직전24h)   ", "flow6": "flow6 (16일궤적6점) "}
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
        tr = np.array([j for j in range(0, ts) if onset_day[j] <= cutoff])
        out.append((tr, np.arange(ts, te)))
    return out

def paired_boot(y, preds, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    dist = {k: [] for k in preds}
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
        yy = y[idx]
        for k in preds: dist[k].append(roc_auc_score(yy, preds[k][idx]))
    return {k: np.array(v) for k, v in dist.items()}

folds = make_folds()
print(f"\n[{REGION}] 6h n={n}, AR임계(85th)={THR:.0f} | 확장창 {N_SPLITS}-fold CV, embargo {EMBARGO_DAYS}d, "
      f"강도 변형 비교 (IVT only), TabPFN={TPMODEL}")
for k in HORIZONS:
    y = (steps >= k).astype(int)
    valid = [(tr, te) for (tr, te) in folds if len(tr) >= 30 and len(np.unique(y[tr])) > 1]
    used = np.zeros(n, bool)
    for (_, te) in valid: used[te] = True
    yk = y[used]
    print(f"\n[{k*6}h] OOF 양성={int(yk.sum())}/{len(yk)}")
    if len(np.unique(yk)) < 2:
        print("  OOF 클래스부족"); continue
    oof = {c: {f: np.full(n, np.nan) for f in FS} for c in CLFS}
    for (tr, te) in valid:
        for f, X in FS.items():
            for c in CLFS:
                oof[c][f][te] = fit_predict(c, X[tr], y[tr], X[te])
    for c in CLFS:
        preds = {f: oof[c][f][used] for f in FS}
        dist = paired_boot(yk, preds)
        print(f"  [{c}]")
        for f in ["peak_LA", "onset_pt", "peak_rec", "flow6"]:
            a = roc_auc_score(yk, preds[f]); lo, hi = np.percentile(dist[f], [2.5, 97.5])
            print(f"        {FS_LABEL[f]} AUC={a:.3f} 95%CI[{lo:.3f},{hi:.3f}]")
        # 핵심 두 비교 (paired)
        dLA = dist["peak_LA"] - dist["peak_rec"]      # 당일 엿보기 효과
        dFL = dist["flow6"] - dist["onset_pt"]        # 흐름이 점보다 나은가
        for nm, d in [("look-ahead 효과 (peak_LA−peak_rec)", dLA), ("흐름 효과 (flow6−onset_pt)", dFL)]:
            lo, hi = np.percentile(d, [2.5, 97.5]); pg = (d > 0).mean()
            flag = "*유의*" if pg >= 0.975 or pg <= 0.025 else "동급"
            print(f"          Δ {nm}: 중앙{np.median(d):+.3f} 95%CI[{lo:+.3f},{hi:+.3f}] P(>0)={pg:.2f} {flag}")
