"""6h 정의, 시간순 확장창 CV — LEAK 제거(완전 causal) 강도-앵커 증분 ablation.

배경: 기존 피처는 onset '당일(day o)' 일별값(dmax[o], jet[o], blk[o] 및 onset일 포함 wavelet)을 써서
      onset 순간 이후 같은 날 최대 ~18h를 엿봤다(look-ahead). 이를 전부 제거한 causal 버전.

leak 제거:
  강도 A      : dmax[o] -> max(ivt[s-3:s+1])   (직전 24h, onset 포함, 미래 없음)
  순환 onset  : jet[o]/blk[o] -> jet[o-1]/blk[o-1]  (직전 완전히 아는 날)
  순환 wavelet: jet[o-63:o+1] -> jet[o-64:o]   (o-1 에서 끝)
  IVT wavelet : ivt[s-63:s+1]                  (onset s 에서 끝나 이미 causal, 유지)

피처셋(강도 anchored, cv_6h_region.py 와 동일 구조):
  A intensity     : [intens]
  B +IVT-temporal : IVT wavelet + [intens]
  C +circulation  : [intens, jet[o-1]] + jet wavelet + blk wavelet
  D full          : 전부
  → Δwav=B-A, Δcirc=C-A, Δfull=D-A  (causal 강도 위 증분)

분류기: Logistic / TabPFN(v2) / LGBM. CV: 확장창 5-fold, embargo 64d, OOF 풀링 + paired 부트스트랩 CI.
주의: AR임계 THR=전기간 85백분위(기후 정의, per-event target leak 아님). train-only 임계는 별도 사안.
사용: python cv_causal_6h_region.py [ca|uk|chile]   (TABPFN_TOKEN 필요)
"""
import os, sys, numpy as np, pywt, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

REGION = sys.argv[1] if len(sys.argv) > 1 else "uk"
N_SPLITS = 5; EMBARGO_DAYS = 64; N_BOOT = 1000; SEED = 0
HORIZONS = [4, 6, 8]
TPMODEL = os.environ.get("TABPFN_MODEL", "tabpfn-v2-classifier-v2_default.ckpt")

IVT_FILE = {"ca": "ivt_sf_1980_2023.npy"}.get(REGION, f"ivt_{REGION}_1980_2023.npy")
CIRC_FILE = {"ca": "circ_indices.npz"}.get(REGION, f"circ_indices_{REGION}.npz")
TIMES_FILE = {"ca": "times_sf_1980_2023.npy"}.get(REGION, f"times_{REGION}_1980_2023.npy")

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

# --- 신규 피처셋 E 재료 (선별 통과 변수): 해면기압/추세, IVT방향, 직전AR활동, 운량, 다중스케일IVT ---
# 전부 causal: 지표·IVT는 ≤ s, 강수규약상 기압/운량 순간값도 ≤ s.
# Colab 전달용 사전계산 efeat_{region}.npz 가 있으면 그걸 쓰고(큰 raw 불필요),
# 없으면 로컬에서 raw(openmeteo + era5 nc)로 직접 빌드.
def find_opt(name):
    try: return find(name)
    except FileNotFoundError: return None
_ef = find_opt(f"efeat_{REGION}.npz")
if _ef is not None:
    _z = np.load(_ef)
    pmsl = _z["pmsl"].astype("float64"); cloud = _z["cloud"].astype("float64")
    dir_sin = _z["dir_sin"].astype("float64"); dir_cos = _z["dir_cos"].astype("float64")
else:
    times = np.load(find(TIMES_FILE))
    from screen_features_region import load_aux_6h, load_ivt_dir
    aux = load_aux_6h(REGION, T)                     # openmeteo + enso + sst (T 정렬)
    dir_sin, dir_cos = load_ivt_dir(REGION, times)   # IVT 방향 (없으면 None)
    pmsl = aux["pressure_msl"]; cloud = aux["cloud_cover"]
    if dir_sin is None:
        dir_sin = np.zeros(T); dir_cos = np.zeros(T)

def wl(a, lvl=5): return [c[-1] for c in pywt.swt(a, 'db2', level=lvl, trim_approx=True, norm=True)]

runs = []; i = 0
while i < T:
    if ar6[i]:
        j = i
        while j < T and ar6[j]: j += 1
        runs.append((i, j)); i = j
    else: i += 1
FA, FB, FC, FD, FE, steps, onset_day = [], [], [], [], [], [], []
for s, e in runs:
    o = s // 4
    if s - 63 >= 0 and o - 64 >= 0 and o < ND and not (np.isnan(jet[o - 1]) or np.isnan(blk[o - 1])):
        intens = ivt[s - 3:s + 1].max()       # causal 강도(직전 24h 최대)
        ivw = wl(ivt[s - 63:s + 1])           # causal IVT 16일 wavelet (s 에서 끝)
        jw = wl(jet[o - 64:o]); bw = wl(blk[o - 64:o])   # causal 순환 wavelet (o-1 에서 끝)
        d_row = ivw + [intens, jet[o - 1]] + jw + bw     # = full(D)
        extra = [pmsl[s], pmsl[s] - pmsl[s - 4],          # 해면기압 + 24h 추세
                 dir_sin[s], dir_cos[s],                  # IVT 방향
                 ar6[max(0, s - 119):s + 1].mean(),       # 직전 30d AR활동
                 ar6[max(0, s - 239):s + 1].mean(),       # 직전 60d AR활동
                 cloud[s],                                # 운량
                 ivt[s - 7:s + 1].max(), ivt[s - 11:s + 1].max(), ivt[s - 27:s + 1].max(),
                 ivt[s - 27:s + 1].mean(), ivt[s - 27:s + 1].std()]   # 다중스케일 IVT
        FA.append([intens])
        FB.append(ivw + [intens])
        FC.append([intens, jet[o - 1]] + jw + bw)
        FD.append(d_row)
        FE.append(d_row + extra)              # E = full + 신규 선별 변수
        steps.append(e - s); onset_day.append(o)
FA, FB, FC, FD, FE = map(np.array, (FA, FB, FC, FD, FE))
steps = np.array(steps); onset_day = np.array(onset_day); n = len(steps)
FS = {"A": FA, "B": FB, "C": FC, "D": FD, "E": FE}
FS_LABEL = {"A": "intensity     ", "B": "+IVT-temporal ", "C": "+circulation  ",
            "D": "full          ", "E": "+신규선별(E)   "}
try:
    from tabpfn import TabPFNClassifier; HAS_TP = True
except Exception:
    HAS_TP = False                      # 로컬(torch 깨짐)에서는 TabPFN 건너뜀, Colab 에서만 포함
CLFS = ["Logistic"] + (["TabPFN"] if HAS_TP else []) + ["LGBM"]
try:
    from lightgbm import LGBMClassifier; HAS_LGBM = True
except Exception:
    from sklearn.ensemble import HistGradientBoostingClassifier; HAS_LGBM = False

def fit_predict(clf, Xtr, ytr, Xte):
    med = np.nanmedian(Xtr, axis=0); med = np.where(np.isnan(med), 0.0, med)   # train-only 결측 대치
    Xtr = np.where(np.isnan(Xtr), med, Xtr); Xte = np.where(np.isnan(Xte), med, Xte)
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
print(f"\n[{REGION}] 6h n={n}, AR임계(85th)={THR:.0f} | LEAK제거 causal | 확장창 {N_SPLITS}-fold CV, "
      f"embargo {EMBARGO_DAYS}d, TabPFN={TPMODEL}, LGBM={'lightgbm' if HAS_LGBM else 'sklearn-HGB'}")
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
        aA = roc_auc_score(yk, preds["A"]); loA, hiA = np.percentile(dist["A"], [2.5, 97.5])
        print(f"  [{c}]  A intensity(causal) AUC={aA:.3f} 95%CI[{loA:.3f},{hiA:.3f}]")
        for f in ["B", "C", "D", "E"]:
            aX = roc_auc_score(yk, preds[f]); d = dist[f] - dist["A"]
            lo, hi = np.percentile(d, [2.5, 97.5]); pg = (d > 0).mean()
            flag = "*유의*" if pg >= 0.975 else ("동급" if pg > 0.5 else "효과없음")
            print(f"        {FS_LABEL[f]} AUC={aX:.3f}  Δ(강도위)={np.median(d):+.3f} "
                  f"95%CI[{lo:+.3f},{hi:+.3f}] P(>0)={pg:.2f} {flag}")
        # 핵심 비교: 신규셋 E 가 기존 full(D) 를 실제로 넘는가
        dED = dist["E"] - dist["D"]; loE, hiE = np.percentile(dED, [2.5, 97.5]); pgE = (dED > 0).mean()
        flagE = "*유의*" if pgE >= 0.975 else ("동급" if pgE > 0.5 else "효과없음")
        print(f"        >>> E vs D(full)  Δ={np.median(dED):+.3f} 95%CI[{loE:+.3f},{hiE:+.3f}] "
              f"P(>0)={pgE:.2f} {flagE}")
