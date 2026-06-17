# ===== 5시드 + 부트스트랩 비교: TabPFN / RF vs Logistic (동일 20피처) =====
import os, numpy as np, pywt, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
rng=np.random.RandomState(0)
HERE=os.path.dirname(os.path.abspath(__file__)); RAW=os.path.join(HERE,"data","raw")
def find(name):
    for p in [os.path.join(RAW,name), name, "/content/"+name]:
        if os.path.exists(p): return p
    raise FileNotFoundError(name)
ivt=np.load(find("ivt_sf_1980_2023.npy")).astype("float64")
sst=np.load(find("sst_anom.npy")).astype("float64")
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
            X.append(wl(ivt[end6-64:end6],5)+[dmax[o],jet[o]]+wl(jet[o-63:o+1],5)+wl(blk[o-63:o+1],5)+[sst[end6-1]]); y.append(int(dur>=3))
        i=j
    else: i+=1
X=np.array(X); y=np.array(y); n=len(y); i1=int(n*0.6); i2=int(n*0.8)
sc=StandardScaler().fit(X[:i1]); Xtr,Xte=sc.transform(X[:i1]),sc.transform(X[i2:]); ytr,yte=y[:i1],y[i2:]
print(f"n={n}, train={i1} test={n-i2}\n")
# Logistic (결정론적)
lr=LogisticRegression(max_iter=3000,C=0.5,class_weight="balanced").fit(Xtr,ytr); plr=lr.predict_proba(Xte)[:,1]
print(f"Logistic     test AUC={roc_auc_score(yte,plr):.3f} (결정론적)")
# RF 5시드
from tabpfn import TabPFNClassifier
def seedrun(make,seeds=5):
    aucs=[]; p0=None
    for s in range(seeds):
        m=make(s); m.fit(Xtr,ytr); p=m.predict_proba(Xte)[:,1]; aucs.append(roc_auc_score(yte,p))
        if s==0: p0=p
    return np.array(aucs),p0
rf_a,prf=seedrun(lambda s: RandomForestClassifier(n_estimators=400,max_depth=4,min_samples_leaf=20,class_weight="balanced",random_state=s))
print(f"RandomForest 5시드 AUC={rf_a.mean():.3f}±{rf_a.std():.3f}  (min {rf_a.min():.3f} max {rf_a.max():.3f})")
tp_a,ptp=seedrun(lambda s: TabPFNClassifier(random_state=s))
print(f"TabPFN       5시드 AUC={tp_a.mean():.3f}±{tp_a.std():.3f}  (min {tp_a.min():.3f} max {tp_a.max():.3f})")
# 부트스트랩 test (차이 CI)
B=2000; dT=[]; dR=[]
for _ in range(B):
    idx=rng.randint(0,len(yte),len(yte))
    if len(np.unique(yte[idx]))<2: continue
    al=roc_auc_score(yte[idx],plr[idx]); at=roc_auc_score(yte[idx],ptp[idx]); arf=roc_auc_score(yte[idx],prf[idx])
    dT.append(at-al); dR.append(arf-al)
dT=np.array(dT); dR=np.array(dR)
print(f"\n부트스트랩 (test 재표집):")
print(f"  TabPFN-Logistic 차이: 중앙{np.median(dT):+.3f} 95%[{np.percentile(dT,2.5):+.3f},{np.percentile(dT,97.5):+.3f}] P(>0)={(dT>0).mean():.2f}")
print(f"  RF-Logistic     차이: 중앙{np.median(dR):+.3f} 95%[{np.percentile(dR,2.5):+.3f},{np.percentile(dR,97.5):+.3f}] P(>0)={(dR>0).mean():.2f}")
print("(차이 CI가 0 넘으면 진짜 우위, 0 포함이면 동급)")
