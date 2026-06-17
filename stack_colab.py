"""[Colab] 스태킹 앙상블: LR/RF/TabPFN 전문가 + 로지스틱 게이트.
임계값 2/3/4/5일, 동일 20피처, 시간순 60/20 홀드아웃, 게이트는 train 내부 5-fold OOF로 학습.
Colab 셀:
  !pip install tabpfn -q
  import os; os.environ["TABPFN_API_KEY"]="키"
  %cd /content/drive/MyDrive/Colab Notebooks/qae-parisian-climate
  !git pull -q
  !python stack_colab.py
"""
import os, numpy as np, pywt, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from tabpfn import TabPFNClassifier
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
makers=[lambda:LogisticRegression(max_iter=3000,C=0.5,class_weight="balanced"),
        lambda:RandomForestClassifier(n_estimators=400,max_depth=4,min_samples_leaf=20,class_weight="balanced",random_state=0),
        lambda:TabPFNClassifier(random_state=0)]
names=["LR","RF","TabPFN"]
def fp(mk,Xa,ya,Xb): m=mk(); m.fit(Xa,ya); return m.predict_proba(Xb)[:,1]
print(f"n={n}, train={i1} test={n-i2}\n")
print(f"{'임계값':>6} {'양성%':>6} {'LR':>6} {'RF':>6} {'TabPFN':>7} {'단순평균':>8} {'스태킹':>8} {'게이트가중(LR/RF/Tab)':>22}")
for thr in [2,3,4,5]:
    y=(durs>=thr).astype(int)
    sc=StandardScaler().fit(X[:i1]); Xall=sc.transform(X); Xtr,Xte=Xall[:i1],Xall[i2:]; ytr,yte=y[:i1],y[i2:]
    if len(np.unique(yte))<2 or len(np.unique(ytr))<2:
        print(f"{thr:>5}일 {y.mean()*100:>5.0f}%  클래스부족 스킵"); continue
    kf=KFold(5,shuffle=True,random_state=0); oof=np.zeros((len(ytr),3))
    for tri,vai in kf.split(Xtr):
        for k,mk in enumerate(makers): oof[vai,k]=fp(mk,Xtr[tri],ytr[tri],Xtr[vai])
    testp=np.column_stack([fp(mk,Xtr,ytr,Xte) for mk in makers])
    ind=[roc_auc_score(yte,testp[:,k]) for k in range(3)]
    avg=roc_auc_score(yte,testp.mean(1))
    gate=LogisticRegression(max_iter=1000).fit(oof,ytr)
    stk=roc_auc_score(yte,gate.predict_proba(testp)[:,1])
    w=gate.coef_[0]
    print(f"{thr:>5}일 {y.mean()*100:>5.0f}% {ind[0]:>6.3f} {ind[1]:>6.3f} {ind[2]:>7.3f} {avg:>8.3f} {stk:>8.3f}   [{w[0]:+.2f}/{w[1]:+.2f}/{w[2]:+.2f}]")
