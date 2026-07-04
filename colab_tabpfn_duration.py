# ===== Colab: TabPFN로 24h 지속 판별 — env / 예보 / env+예보 (+피처선택 규제) =====
# 준비: feats_ca.npz, feats_uk.npz, feats_chile.npz 를 Colab에 (git pull).
# TabPFN엔 C 규제 없음 -> "규제"=train에서 top-k 피처선택(SelectKBest)로 노이즈 env 억제.
# 워크포워드 CV(임베고 64) + MCC 최대화 임계(train에서 골라 test 적용 = out-of-sample).
# (tabpfn 설치는 노트북 셀에서: !pip install tabpfn -q)
import numpy as np, torch
from tabpfn import TabPFNClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score, matthews_corrcoef
DEV = "cuda" if torch.cuda.is_available() else "cpu"
FEAT_DIR = ""   # repo 폴더 안에서 실행하면 빈 문자열


def make_folds(N, od, Nf=5, emb=64):
    fold = N // (Nf + 1); out = []
    for k in range(1, Nf + 1):
        ts = k * fold; te = (k + 1) * fold if k < Nf else N
        out.append((np.array([j for j in range(0, ts) if od[j] <= od[ts] - emb]), np.arange(ts, te)))
    return out


def best_thr(yy, pp):
    best = (-2.0, 0.5)
    for t in np.unique(pp):
        m = matthews_corrcoef(yy, (pp >= t).astype(int))
        if m > best[0]: best = (m, t)
    return best[1]


def evalX(X, y, folds, k=None):
    yt = []; pb = []
    for tr, te in folds:
        if len(tr) < 40 or len(np.unique(y[tr])) < 2: continue
        Xtr = X[tr].astype(float).copy(); Xte = X[te].astype(float).copy()
        med = np.nanmedian(Xtr, 0); med = np.where(np.isnan(med), 0.0, med)
        Xtr = np.where(np.isnan(Xtr), med, Xtr); Xte = np.where(np.isnan(Xte), med, Xte)
        if k is not None and k < Xtr.shape[1]:
            sel = SelectKBest(f_classif, k=k).fit(Xtr, y[tr])
            Xtr = sel.transform(Xtr); Xte = sel.transform(Xte)
        clf = TabPFNClassifier(device=DEV); clf.fit(Xtr, y[tr])
        ptr = clf.predict_proba(Xtr)[:, 1]; pte = clf.predict_proba(Xte)[:, 1]
        thr = best_thr(y[tr], ptr)
        pb.extend((pte >= thr).astype(int).tolist()); yt.extend(y[te].tolist())
    yt = np.array(yt); pb = np.array(pb)
    return f1_score(yt, pb, zero_division=0), matthews_corrcoef(yt, pb), len(yt)


for r in ["ca", "uk", "chile"]:
    d = np.load(FEAT_DIR + f"feats_{r}.npz")
    FE, FC, y, oday = d["FE"], d["FC"], d["y"], d["oday"]
    XA = np.hstack([FE, FC])
    folds = make_folds(len(y), oday)
    print(f"[{r}] N={len(y)} 지속={int(y.sum())} ({y.mean()*100:.0f}%)")
    for tag, X, k in [("예보단독", FC, None), ("env만", FE, None),
                      ("env+예보(전체37)", XA, None), ("env+예보 k=8", XA, 8), ("env+예보 k=5", XA, 5)]:
        f1, mcc, n = evalX(X, y, folds, k)
        print(f"  {tag:16s}: F1={f1:.3f} MCC={mcc:.3f} (n={n})")
