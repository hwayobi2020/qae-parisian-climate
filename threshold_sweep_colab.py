"""[Colab] 지속기간 임계값 스윕 + TabPFN. 일별 정의, 동일 20피처, 시간순 60/20분할.
Colab 셀:
  !pip install tabpfn -q
  import os; os.environ["TABPFN_API_KEY"]="여기에_API_KEY"   # 또는 무료 라이선스 수락
  !cd /content/qae-parisian-climate && git pull -q && python threshold_sweep_colab.py
"""
import os, numpy as np, pywt, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
HERE=os.path.dirname(os.path.abspath(__file__)); RAW=os.path.join(HERE,"data","raw")
def find(name):
    for p in [os.path.join(RAW,name), name, "/content/"+name, os.path.join("/content/qae-parisian-climate/data/raw",name)]:
        if os.path.exists(p): return p
    raise FileNotFoundError(name)
ivt=np.load(find("ivt_sf_1980_2023.npy")).astype("float64")
ci=np.load(find("circ_indices.npz")); jet=ci["jet"].astype("float64"); blk=ci["blocking"].astype("float64")
dmax=ivt.reshape(-1,4).max(1); ND=len(dmax); ar=dmax>250
def wl(a,lvl): return [c[-1] for c in pywt.swt(a,'db2',level=lvl,trim_approx=True,norm=True)]
X=[];durs=[];i=0
while i<ND:
    if ar[i]:
        j=i
        while j<ND and ar[j]: j+=1
        o=i; dur=j-i; end6=(o+1)*4
        if end6-64>=0 and o-63>=0:
            X.append(wl(ivt[end6-64:end6],5)+[dmax[o],jet[o]]+wl(jet[o-63:o+1],5)+wl(blk[o-63:o+1],5)); durs.append(dur)
        i=j
    else: i+=1
X=np.array(X); durs=np.array(durs); n=len(durs); i1=int(n*0.6); i2=int(n*0.8)
print(f"일별 정의 이벤트수 n={n}, train={i1} test={n-i2}\n")
try:
    from tabpfn import TabPFNClassifier; HAVE_TP=True
except Exception as e:
    HAVE_TP=False; print(f"[TabPFN 미설치: {e}]\n")
print(f"{'임계값':>7} {'양성%':>6} {'Logistic':>9} {'RF(5시드)':>15} {'TabPFN(5시드)':>17} {'test양성':>8}")
for thr in [2,3,4,5]:
    y=(durs>=thr).astype(int)
    sc=StandardScaler().fit(X[:i1]); Xtr,Xte=sc.transform(X[:i1]),sc.transform(X[i2:]); ytr,yte=y[:i1],y[i2:]
    if len(np.unique(yte))<2 or len(np.unique(ytr))<2:
        print(f"{thr:>6}일 {y.mean()*100:>5.0f}%  클래스부족 스킵(test양성={int(yte.sum())})"); continue
    lr=LogisticRegression(max_iter=3000,C=0.5,class_weight="balanced").fit(Xtr,ytr)
    al=roc_auc_score(yte,lr.predict_proba(Xte)[:,1])
    rf=np.array([roc_auc_score(yte,RandomForestClassifier(n_estimators=400,max_depth=4,min_samples_leaf=20,class_weight="balanced",random_state=sd).fit(Xtr,ytr).predict_proba(Xte)[:,1]) for sd in range(5)])
    if HAVE_TP:
        tp=np.array([roc_auc_score(yte,TabPFNClassifier(random_state=sd).fit(Xtr,ytr).predict_proba(Xte)[:,1]) for sd in range(5)])
        tps=f"{tp.mean():.3f}±{tp.std():.3f}"
    else: tps="(미설치)"
    print(f"{thr:>6}일 {y.mean()*100:>5.0f}%   {al:>7.3f}   {rf.mean():.3f}±{rf.std():.3f}   {tps:>15} {int(yte.sum()):>8}")
