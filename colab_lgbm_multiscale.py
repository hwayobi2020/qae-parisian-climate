# ===== 슬림 다중척도 + LGBM : AR onset -> 3일+ 지속 (KAN과 동일 피처) =====
# IVT wavelet32일(7)+IVT onset(1)+U250 jet wavelet64일(6)+U250 jet onset(1)=15
# 시간순 train/val/test(60/20/20), val 조기종료, test AUC.
import os
import numpy as np, pywt
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE=os.path.dirname(os.path.abspath(__file__)); RAW=os.path.join(HERE,"data","raw")
def find(name):
    for p in [os.path.join(RAW,name), name, "/content/"+name]:
        if os.path.exists(p): return p
    raise FileNotFoundError(name)
ivt=np.load(find("ivt_sf_1980_2023.npy")).astype("float64")
jet=np.load(find("circ_indices.npz"))["jet"].astype("float64")
dmax=ivt.reshape(-1,4).max(1); ND=len(dmax); ar=dmax>250
def wlast(arr,lvl): return [c[-1] for c in pywt.swt(arr,'db2',level=lvl,trim_approx=True,norm=True)]
X=[]; y=[]; i=0
while i<ND:
    if ar[i]:
        j=i
        while j<ND and ar[j]: j+=1
        o=i; dur=j-i; end6=(o+1)*4
        if end6-128>=0 and o-63>=0:
            X.append(wlast(ivt[end6-128:end6],6)+[dmax[o]]+[jet[o]]+wlast(jet[o-63:o+1],5))
            y.append(int(dur>=3))
        i=j
    else: i+=1
X=np.array(X); y=np.array(y); n=len(y)
i1=int(n*0.6); i2=int(n*0.8)
Xtr,ytr=X[:i1],y[:i1]; Xva,yva=X[i1:i2],y[i1:i2]; Xte,yte=X[i2:],y[i2:]
print(f"n={n}, feat={X.shape[1]}, tr/va/te={i1}/{i2-i1}/{n-i2}, pos% {ytr.mean()*100:.0f}/{yva.mean()*100:.0f}/{yte.mean()*100:.0f}")
spw=(ytr==0).sum()/max((ytr==1).sum(),1)
clf=LGBMClassifier(n_estimators=600,learning_rate=0.02,num_leaves=15,max_depth=4,
                   min_child_samples=20,subsample=0.8,colsample_bytree=0.8,reg_lambda=1.0,
                   scale_pos_weight=spw,verbose=-1)
clf.fit(Xtr,ytr,eval_set=[(Xva,yva)],eval_metric="auc",
        callbacks=[early_stopping(50,verbose=False),log_evaluation(0)])
pv=clf.predict_proba(Xva)[:,1]; pt=clf.predict_proba(Xte)[:,1]
print(f"LGBM  val AUC={roc_auc_score(yva,pv):.3f}  test AUC={roc_auc_score(yte,pt):.3f}  (best_iter={clf.best_iteration_})")
# 비교용 로지스틱(같은 피처)
sc=StandardScaler().fit(Xtr); lr=LogisticRegression(max_iter=3000,C=0.5).fit(sc.transform(Xtr),ytr)
print(f"Logistic(동일피처) test AUC={roc_auc_score(yte,lr.predict_proba(sc.transform(Xte))[:,1]):.3f}")
