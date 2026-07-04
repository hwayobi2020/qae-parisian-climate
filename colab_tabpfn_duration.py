# ===== Colab: TabPFN로 24h 지속 판별 — env / 예보 / env+예보 비교 =====
# 준비: feats_ca.npz, feats_uk.npz, feats_chile.npz 를 Colab에 올림
#   (git push 후 !git clone/pull, 또는 좌측 파일창에 직접 업로드, 또는 Drive 마운트)
# 워크포워드 CV(임베고 64) + MCC 최대화 임계(train에서 골라 test 적용 = out-of-sample).
!pip -q install tabpfn
import numpy as np, torch
from tabpfn import TabPFNClassifier
from sklearn.metrics import f1_score, matthews_corrcoef
DEV = "cuda" if torch.cuda.is_available() else "cpu"
FEAT_DIR = ""   # 예: "/content/" 또는 "/content/drive/MyDrive/Colab Notebooks/"


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


def evalX(X, y, folds):
    yt = []; pb = []
    for tr, te in folds:
        if len(tr) < 40 or len(np.unique(y[tr])) < 2: continue
        Xtr = X[tr].astype(float).copy(); Xte = X[te].astype(float).copy()
        med = np.nanmedian(Xtr, 0); med = np.where(np.isnan(med), 0.0, med)
        Xtr = np.where(np.isnan(Xtr), med, Xtr); Xte = np.where(np.isnan(Xte), med, Xte)
        clf = TabPFNClassifier(device=DEV)
        clf.fit(Xtr, y[tr])
        ptr = clf.predict_proba(Xtr)[:, 1]; pte = clf.predict_proba(Xte)[:, 1]
        thr = best_thr(y[tr], ptr)
        pb.extend((pte >= thr).astype(int).tolist()); yt.extend(y[te].tolist())
    yt = np.array(yt); pb = np.array(pb)
    return f1_score(yt, pb, zero_division=0), matthews_corrcoef(yt, pb), len(yt)


for r in ["ca", "uk", "chile"]:
    d = np.load(FEAT_DIR + f"feats_{r}.npz")
    FE, FC, y, oday = d["FE"], d["FC"], d["y"], d["oday"]
    folds = make_folds(len(y), oday)
    print(f"[{r}] N={len(y)} 지속={int(y.sum())} ({y.mean()*100:.0f}%)")
    for tag, X in [("예보단독", FC), ("env만", FE), ("env+예보", np.hstack([FE, FC]))]:
        f1, mcc, n = evalX(X, y, folds)
        print(f"  {tag:10s}: F1={f1:.3f} MCC={mcc:.3f} (n={n})")
