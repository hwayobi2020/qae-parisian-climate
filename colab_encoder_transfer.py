# ===== Colab: 인코더 전이 — pre-2000(예보없음) TabPFN 학습 -> 예측을 인코더피처로 -> 2000+ 결합 =====
# 준비: transfer_{ca,uk,chile}.npz (git pull). tabpfn 설치 셀: !pip install tabpfn -q
# pre-2000 TabPFN은 2000+ 를 절대 안 봄 -> 인코더피처는 완전 OOS(누수 0). 최종=[env+예보+인코더].
import numpy as np, torch
from tabpfn import TabPFNClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef, roc_auc_score, f1_score
DEV = "cuda" if torch.cuda.is_available() else "cpu"; FEAT_DIR = ""


def folds(N, od, Nf=5, emb=64):
    f = N // (Nf + 1); out = []
    for k in range(1, Nf + 1):
        ts = k * f; te = (k + 1) * f if k < Nf else N
        out.append((np.array([j for j in range(0, ts) if od[j] <= od[ts] - emb]), np.arange(ts, te)))
    return out


def bt(yy, pp):
    b = (-2., .5)
    for t in np.unique(pp):
        m = matthews_corrcoef(yy, (pp >= t).astype(int))
        if m > b[0]: b = (m, t)
    return b[1]


def combine(X, yf, odf):
    ytb = []; pb = []; pp = np.full(len(yf), np.nan)
    for tr, te in folds(len(yf), odf):
        if len(tr) < 40 or len(np.unique(yf[tr])) < 2: continue
        Xtr = np.nan_to_num(X[tr]); Xte = np.nan_to_num(X[te]); sc = StandardScaler().fit(Xtr)
        m = LogisticRegression(max_iter=3000, C=0.3, class_weight="balanced").fit(sc.transform(Xtr), yf[tr])
        ptr = m.predict_proba(sc.transform(Xtr))[:, 1]; pte = m.predict_proba(sc.transform(Xte))[:, 1]
        pp[te] = pte; th = bt(yf[tr], ptr); pb.extend((pte >= th).astype(int)); ytb.extend(yf[te])
    ok = ~np.isnan(pp); ytb = np.array(ytb); pb = np.array(pb)
    return roc_auc_score(yf[ok], pp[ok]), matthews_corrcoef(ytb, pb), f1_score(ytb, pb, zero_division=0)


for r in ["ca", "uk", "chile"]:
    d = np.load(FEAT_DIR + f"transfer_{r}.npz")
    FE, FC, y, oday, fmask = d["FE"], d["FC"], d["y"], d["oday"], d["fmask"].astype(bool)
    fstart = oday[fmask].min()                 # 예보 시작(2000년경)
    pre = oday < fstart                        # pre-2000 (예보없는 과거)
    mE = TabPFNClassifier(device=DEV); mE.fit(np.nan_to_num(FE[pre]), y[pre])
    enc = mE.predict_proba(np.nan_to_num(FE))[:, 1]   # 인코더피처(pre2000 모델 예측, 2000+엔 완전 OOS)
    sel = fmask; FEf = FE[sel]; FCf = FC[sel]; encf = enc[sel]; yf = y[sel]; odf = oday[sel]
    print(f"[{r}] pre-2000 학습 {int(pre.sum())} | 예보온셋 {int(sel.sum())} 양성 {int(yf.sum())} | AUC/MCC/F1")
    for tag, X in [("예보단독", FCf),
                   ("예보+env", np.hstack([FCf, FEf])),
                   ("예보+env+인코더", np.hstack([FCf, FEf, encf.reshape(-1, 1)]))]:
        a, m, f = combine(X, yf, odf)
        print(f"  {tag:16s}: AUC={a:.3f} MCC={m:.3f} F1={f:.3f}")
