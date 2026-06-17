# ===== TabPFN vs Logistic vs RF : AR onset -> 3일+ 지속 (동일 20피처) =====
# IVT16d wav6 + IVT onset1 + U250 onset1 + U250 64d wav6 + Z500 64d wav6 = 20
# 시간순 train60/test20. (Colab 셀: !pip install tabpfn -q)
import os, numpy as np, pywt, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

HERE=os.path.dirname(os.path.abspath(__file__)); RAW=os.path.join(HERE,"data","raw")
def find(name):
    for p in [os.path.join(RAW,name), name, "/content/"+name]:
        if os.path.exists(p): return p
    raise FileNotFoundError(name)
ivt=np.load(find("ivt_sf_1980_2023.npy")).astype("float64")
ci=np.load(find("circ_indices.npz")); jet=ci["jet"].astype("float64"); blk=ci["blocking"].astype("float64")
dmax=ivt.reshape(-1,4).max(1); ND=len(dmax); ar=dmax>250
def wl(a,lvl): return [c[-1] for c in pywt.swt(a,'db2',level=lvl,trim_approx=True,norm=True)]
X=[];y=[];i=0
while i<ND:
    if ar[i]:
        j=i
        while j<ND and ar[j]: j+=1
        o=i; dur=j-i; end6=(o+1)*4
        if end6-64>=0 and o-63>=0:
            X.append(wl(ivt[end6-64:end6],5)+[dmax[o],jet[o]]+wl(jet[o-63:o+1],5)+wl(blk[o-63:o+1],5)); y.append(int(dur>=3))
        i=j
    else: i+=1
X=np.array(X); y=np.array(y); n=len(y); i1=int(n*0.6); i2=int(n*0.8)
sc=StandardScaler().fit(X[:i1]); Xtr,Xte=sc.transform(X[:i1]),sc.transform(X[i2:]); ytr,yte=y[:i1],y[i2:]
print(f"n={n}, feat={X.shape[1]}, train={i1} test={n-i2}, pos% tr/te={ytr.mean()*100:.0f}/{yte.mean()*100:.0f}")
lr=LogisticRegression(max_iter=3000,C=0.5,class_weight="balanced").fit(Xtr,ytr)
print(f"  Logistic     test AUC={roc_auc_score(yte,lr.predict_proba(Xte)[:,1]):.3f}")
rf=RandomForestClassifier(n_estimators=400,max_depth=4,min_samples_leaf=20,class_weight="balanced",random_state=0).fit(Xtr,ytr)
print(f"  RandomForest test AUC={roc_auc_score(yte,rf.predict_proba(Xte)[:,1]):.3f}")
from tabpfn import TabPFNClassifier
tp=TabPFNClassifier()
tp.fit(Xtr,ytr)
print(f"  TabPFN       test AUC={roc_auc_score(yte,tp.predict_proba(Xte)[:,1]):.3f}")
