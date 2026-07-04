# ===== Colab: TabPFN — 24h 지속 예측. F1/MCC + CSI(늑대소년 관점). =====
# 준비: feats_{ca,uk,chile}.npz 를 Colab에 (git pull). tabpfn 설치는 노트북 셀에서: !pip install tabpfn -q
# "규제"=피처선택(SelectKBest). 임계=best-F1(train), out-of-sample. CSI=TP/(TP+FP+FN) (TN 제외).
import numpy as np, torch
from tabpfn import TabPFNClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score, matthews_corrcoef
DEV = "cuda" if torch.cuda.is_available() else "cpu"
FEAT_DIR = ""


def make_folds(N, od, Nf=5, emb=64):
    fold = N // (Nf + 1); out = []
    for k in range(1, Nf + 1):
        ts = k * fold; te = (k + 1) * fold if k < Nf else N
        out.append((np.array([j for j in range(0, ts) if od[j] <= od[ts] - emb]), np.arange(ts, te)))
    return out


def best_f1_thr(yy, pp):
    best = (-1.0, 0.5)
    for t in np.unique(pp):
        f = f1_score(yy, (pp >= t).astype(int), zero_division=0)
        if f > best[0]: best = (f, t)
    return best[1]


def evalX(X, y, folds, k=None):
    yt = []; pb = []
    for tr, te in folds:
        if len(tr) < 40 or len(np.unique(y[tr])) < 2: continue
        Xtr = X[tr].astype(float).copy(); Xte = X[te].astype(float).copy()
        med = np.nanmedian(Xtr, 0); med = np.where(np.isnan(med), 0.0, med)
        Xtr = np.where(np.isnan(Xtr), med, Xtr); Xte = np.where(np.isnan(Xte), med, Xte)
        if k is not None and k < Xtr.shape[1]:
            sel = SelectKBest(f_classif, k=k).fit(Xtr, y[tr]); Xtr = sel.transform(Xtr); Xte = sel.transform(Xte)
        clf = TabPFNClassifier(device=DEV); clf.fit(Xtr, y[tr])
        ptr = clf.predict_proba(Xtr)[:, 1]; pte = clf.predict_proba(Xte)[:, 1]
        th = best_f1_thr(y[tr], ptr)
        pb.extend((pte >= th).astype(int).tolist()); yt.extend(y[te].tolist())
    yt = np.array(yt); pb = np.array(pb)
    TP = int(((pb == 1) & (yt == 1)).sum()); FP = int(((pb == 1) & (yt == 0)).sum())
    FN = int(((pb == 0) & (yt == 1)).sum())
    csi = TP / (TP + FP + FN) if (TP + FP + FN) else float("nan")
    return f1_score(yt, pb, zero_division=0), matthews_corrcoef(yt, pb), csi, TP, FP, FN


for r in ["ca", "uk", "chile"]:
    d = np.load(FEAT_DIR + f"feats_{r}.npz"); FE, FC, y, oday = d["FE"], d["FC"], d["y"], d["oday"]
    XA = np.hstack([FE, FC]); folds = make_folds(len(y), oday)
    print(f"[{r}] N={len(y)} 실제지속={int(y.sum())} ({y.mean()*100:.0f}%)")
    for tag, X, k in [("예보단독", FC, None), ("env+예보 k=8", XA, 8), ("env+예보(전체37)", XA, None)]:
        f1, mcc, csi, TP, FP, FN = evalX(X, y, folds, k)
        print(f"  {tag:16s}: F1={f1:.3f} MCC={mcc:.3f} CSI={csi:.3f} | TP={TP} 헛경보={FP} 놓침={FN}")
