"""[Colab] 모델 쌍별 paired 부트스트랩: 어느 모델이 유의하게 나은가.
LR/RF/TabPFN, 임계값 2/3/4/5일, 시간순 60/20 홀드아웃 test 재표집(B=2000).
Colab 셀:
  !pip install tabpfn -q
  import os; os.environ["TABPFN_API_KEY"]="키"
  %cd /content/drive/MyDrive/Colab Notebooks/qae-parisian-climate
  !git pull -q
  !python model_compare_ci_colab.py
"""
import os, numpy as np, pywt, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
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
names=["LR","RF","TabPFN"]; pairs=[("RF","LR"),("TabPFN","LR"),("RF","TabPFN")]
def fp(mk,Xa,ya,Xb): m=mk(); m.fit(Xa,ya); return m.predict_proba(Xb)[:,1]
rng=np.random.RandomState(0); B=2000
print(f"n={n}, train={i1} test={n-i2}, paired 부트스트랩 B={B}\n")
for thr in [2,3,4,5]:
    y=(durs>=thr).astype(int)
    sc=StandardScaler().fit(X[:i1]); Xall=sc.transform(X); Xtr,Xte=Xall[:i1],Xall[i2:]; ytr,yte=y[:i1],y[i2:]
    if len(np.unique(yte))<2: print(f"[>= {thr}일] 클래스부족 스킵\n"); continue
    P={names[k]:fp(mk,Xtr,ytr,Xte) for k,mk in enumerate(makers)}
    aucs={nm:[] for nm in names}; diffs={f"{a}-{b}":[] for a,b in pairs}
    for _ in range(B):
        idx=rng.randint(0,len(yte),len(yte))
        if len(np.unique(yte[idx]))<2: continue
        a={nm:roc_auc_score(yte[idx],P[nm][idx]) for nm in names}
        for nm in names: aucs[nm].append(a[nm])
        for u,v in pairs: diffs[f"{u}-{v}"].append(a[u]-a[v])
    print(f"[>= {thr}일]  양성 test={int(yte.sum())}/{len(yte)}")
    for nm in names:
        arr=np.array(aucs[nm]); print(f"  {nm:>7}  AUC={np.median(arr):.3f}  95%CI[{np.percentile(arr,2.5):.3f}, {np.percentile(arr,97.5):.3f}]")
    for u,v in pairs:
        d=np.array(diffs[f"{u}-{v}"]); pg=(d>0).mean()
        flag="*유의*" if (pg>=0.975 or pg<=0.025) else "동급"
        print(f"  {u}-{v:<6} 차이 {np.median(d):+.3f}  95%CI[{np.percentile(d,2.5):+.3f}, {np.percentile(d,97.5):+.3f}]  P({u}>{v})={pg:.2f}  {flag}")
    print()
