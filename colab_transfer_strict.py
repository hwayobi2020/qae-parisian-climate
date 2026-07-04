# ===== Colab: strict-nested 누수차단 — env=TabPFN, 각 테스트폴드 '이전'만으로 학습. MCC. =====
# 준비: transfer_{ca,uk,chile}.npz (git pull). tabpfn 설치 셀에서: !pip install tabpfn -q
import numpy as np, torch
from tabpfn import TabPFNClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef, f1_score
DEV = "cuda" if torch.cuda.is_available() else "cpu"; FEAT_DIR = ""; EMB = 64


def bt(yy, pp):
    b = (-2., .5)
    for t in np.unique(pp):
        m = matthews_corrcoef(yy, (pp >= t).astype(int))
        if m > b[0]: b = (m, t)
    return b[1]


def fitlr(X, yy, C=0.3):
    md = np.nanmedian(X, 0); md = np.where(np.isnan(md), 0.0, md)
    Xi = np.where(np.isnan(X), md, X); sc = StandardScaler().fit(Xi)
    return sc, LogisticRegression(max_iter=3000, C=C, class_weight="balanced").fit(sc.transform(Xi), yy), md


def plr(sc, m, md, X):
    Xi = np.where(np.isnan(X), md, X); return m.predict_proba(sc.transform(Xi))[:, 1]


for r in ["ca", "uk", "chile"]:
    d = np.load(FEAT_DIR + f"transfer_{r}.npz")
    FE, FC, y, oday, fmask = d["FE"], d["FC"], d["y"], d["oday"], d["fmask"].astype(bool)
    N = len(y); fc_idx = np.where(fmask)[0]; nf = len(fc_idx); fold = nf // 6
    yb = []; pb = []; ye = []; pe = []
    for k in range(1, 6):
        ts = k * fold; te = (k + 1) * fold if k < 5 else nf
        trF = fc_idx[:ts]; teF = fc_idx[ts:te]; cut = oday[fc_idx[ts]]
        if len(np.unique(y[trF])) < 2: continue
        envtrain = np.array([j for j in range(N) if oday[j] <= cut - EMB])
        if len(envtrain) < 60 or len(np.unique(y[envtrain])) < 2: continue
        mE = TabPFNClassifier(device=DEV); mE.fit(np.nan_to_num(FE[envtrain]), y[envtrain])
        e_tr = mE.predict_proba(np.nan_to_num(FE[trF]))[:, 1]; e_te = mE.predict_proba(np.nan_to_num(FE[teF]))[:, 1]
        scB, mB2, mdB = fitlr(FC[trF], y[trF]); pbtr = plr(scB, mB2, mdB, FC[trF]); pbte = plr(scB, mB2, mdB, FC[teF])
        thb = bt(y[trF], pbtr); pb.extend((pbte >= thb).astype(int)); yb.extend(y[teF])
        Xtr = np.column_stack([FC[trF], e_tr]); Xte = np.column_stack([FC[teF], e_te])
        scC, mC2, mdC = fitlr(Xtr, y[trF]); petr = plr(scC, mC2, mdC, Xtr); pete = plr(scC, mC2, mdC, Xte)
        thc = bt(y[trF], petr); pe.extend((pete >= thc).astype(int)); ye.extend(y[teF])
    yb = np.array(yb); pb = np.array(pb); ye = np.array(ye); pe = np.array(pe)
    mB = matthews_corrcoef(yb, pb); mE = matthews_corrcoef(ye, pe)
    print(f"[{r}] strict TabPFN | 예보온셋 {nf} 양성 {int(y[fmask].sum())}")
    print(f"  예보단독 MCC={mB:.3f} | 예보+env(TabPFN strict) MCC={mE:.3f} | env효과 {mE - mB:+.3f}")
