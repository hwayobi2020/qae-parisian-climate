"""하이브리드 CV: E(스냅샷) / N(NWP 예보) / E+N — CA 2000-2019, leak-free 확장창.
핵심 질문: E+N 이 N 을 넘나(스냅샷이 NWP에 추가되나). lr/TabPFN/LGBM 3분류기.
피처는 hybrid_feats_ca.npz 로 precompute(로컬) → Colab에서 그걸 로드해 TabPFN 포함 실행.
사용: 로컬(빌드+lr/lgbm) python cv_hybrid.py / Colab(+TabPFN) git pull 후 동일.
"""
import os, numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
TPMODEL = os.environ.get("TABPFN_MODEL", "tabpfn-v2-classifier-v2_default.ckpt")
HERE = os.path.dirname(os.path.abspath(__file__))


def find(name, must=True):
    for p in [os.path.join(HERE, "data", "raw", name), os.path.join(HERE, name),
              "data/raw/" + name, "/content/" + name,
              "/content/qae-parisian-climate/data/raw/" + name, "/content/qae-parisian-climate/" + name]:
        if os.path.exists(p): return p
    if must: raise FileNotFoundError(name)
    return None


FEAT = find("hybrid_feats_ca.npz", must=False)
if FEAT is not None:                       # Colab: precompute 된 피처 로드
    d = np.load(FEAT); FE, FN, steps, oday = d["FE"], d["FN"], d["steps"], d["oday"]
else:                                       # 로컬: raw 에서 빌드 후 저장
    import pywt
    ivt = np.load(find("ivt_sf_1980_2023.npy")).astype("float64")
    ci = np.load(find("circ_indices.npz"), allow_pickle=True); jet = ci["jet"].astype("float64"); blk = ci["blocking"].astype("float64")
    ef = np.load(find("efeat_ca.npz")); pmsl = ef["pmsl"].astype("float64"); cloud = ef["cloud"].astype("float64")
    dsin = ef["dir_sin"].astype("float64"); dcos = ef["dir_cos"].astype("float64"); ND = len(jet)
    THR = np.percentile(ivt.reshape(-1, 4).max(1), 85); ar6 = ivt > THR
    z = np.load(find("gefs_pwat_ca.npz")); S = z["s"]; Ee = z["e"]; OH = z["onset_hour"]; leads = list(z["leads"]); pw = z["pwat"]
    keep = ~np.isnan(pw[:, 0, :]).all(1); S, Ee, OH, pw = S[keep], Ee[keep], OH[keep], pw[keep]
    Lmap = {int(L): i for i, L in enumerate(leads)}
    def wl(a): return [c[-1] for c in pywt.swt(a, 'db2', level=5, trim_approx=True, norm=True)]
    def nwin(oi, a, b):
        idx = [Lmap[x] for x in range(OH[oi] + a, OH[oi] + b + 1, 6) if x in Lmap]
        return np.nanmean(pw[oi, :, idx]) if idx else np.nan
    FE = []; FN = []; steps = []; oday = []
    for k in range(len(S)):
        s = int(S[k]); e = int(Ee[k]); o = s // 4
        if not (s - 63 >= 0 and o - 64 >= 0 and o < ND and not (np.isnan(jet[o - 1]) or np.isnan(blk[o - 1]))):
            continue
        FE.append(wl(ivt[s - 63:s + 1]) + [ivt[s - 3:s + 1].max(), jet[o - 1]] + wl(jet[o - 64:o]) + wl(blk[o - 64:o]) +
                  [pmsl[s], pmsl[s] - pmsl[s - 4], dsin[s], dcos[s],
                   ar6[max(0, s - 119):s + 1].mean(), ar6[max(0, s - 239):s + 1].mean(), cloud[s],
                   ivt[s - 7:s + 1].max(), ivt[s - 11:s + 1].max(), ivt[s - 27:s + 1].max(),
                   ivt[s - 27:s + 1].mean(), ivt[s - 27:s + 1].std()])
        FN.append([nwin(k, 6, 24), nwin(k, 24, 72), nwin(k, 48, 96), nwin(k, 0, 6), nwin(k, 6, 48) - nwin(k, 48, 96)])
        steps.append(e - s); oday.append(o)
    FE = np.array(FE); FN = np.array(FN); steps = np.array(steps); oday = np.array(oday)
    np.savez(os.path.join(HERE, "hybrid_feats_ca.npz"), FE=FE, FN=FN, steps=steps, oday=oday)
    print(f"saved hybrid_feats_ca.npz  n={len(steps)}", flush=True)

m = len(steps); FEN = np.hstack([FE, FN])
FS = {"E": FE, "N": FN, "E+N": FEN}
try:
    from tabpfn import TabPFNClassifier; HAS_TP = True
except Exception:
    HAS_TP = False
CLFS = ["lr"] + (["tabpfn"] if HAS_TP else []) + ["lgbm"]
try:
    from lightgbm import LGBMClassifier; HAS_LGBM = True
except Exception:
    from sklearn.ensemble import HistGradientBoostingClassifier; HAS_LGBM = False


def fp(clf, X, y, Xte):
    md = np.nanmedian(X, 0); md = np.where(np.isnan(md), 0, md)
    X = np.where(np.isnan(X), md, X); Xte = np.where(np.isnan(Xte), md, Xte)
    if clf == "lr":
        sc = StandardScaler().fit(X)
        return LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced").fit(sc.transform(X), y).predict_proba(sc.transform(Xte))[:, 1]
    if clf == "tabpfn":
        kw = {} if TPMODEL == "auto" else {"model_path": TPMODEL}
        return TabPFNClassifier(random_state=0, ignore_pretraining_limits=True, **kw).fit(X, y).predict_proba(Xte)[:, 1]
    if HAS_LGBM:
        return LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=4, max_depth=2, min_child_samples=6,
                              reg_lambda=1.0, class_weight="balanced", verbose=-1).fit(X, y).predict_proba(Xte)[:, 1]
    return HistGradientBoostingClassifier(max_depth=2, max_iter=100, learning_rate=0.05, l2_regularization=1.0,
                                          class_weight="balanced").fit(X, y).predict_proba(Xte)[:, 1]


fold = m // 6; folds = []
for k in range(1, 6):
    ts = k * fold; te = (k + 1) * fold if k < 5 else m; cut = oday[ts] - 64
    folds.append((np.array([j for j in range(ts) if oday[j] <= cut]), np.arange(ts, te)))
print(f"하이브리드 CV (CA 2000-2019, n={m}) | CLFS={CLFS}")
for K, lab in [(4, "24h"), (5, "30h"), (8, "48h")]:
    y = (steps >= K).astype(int); used = np.zeros(m, bool)
    for _, te in folds: used[te] = True
    print(f"\n[≥{lab}] OOF양성={int(y[used].sum())}/{int(used.sum())}")
    for clf in CLFS:
        oof = {k: np.full(m, np.nan) for k in FS}
        for tr, te in folds:
            if len(tr) < 30 or len(np.unique(y[tr])) < 2: continue
            for k, X in FS.items(): oof[k][te] = fp(clf, X[tr], y[tr], X[te])
        au = {k: roc_auc_score(y[used], oof[k][used]) for k in FS}
        print(f"  [{clf:>7}] E={au['E']:.3f}  N={au['N']:.3f}  E+N={au['E+N']:.3f}   (E+N−N={au['E+N']-au['N']:+.3f})")
