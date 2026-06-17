# ===== 슬림 다중척도 + KAN : AR onset -> 3일+ 지속 =====
# 기초통계상 설명력 있는 변수만 (Z500/SST 제외):
#   IVT wavelet(32일,128스텝,L6)=7 + IVT onset(raw)=1
#   U250 jet wavelet(64일,L5)=6   + U250 jet onset(raw)=1   -> 15 피처
# 시간순 train/val/test(60/20/20), train으로만 표준화, val 조기종료, test AUC.
import os, copy
import numpy as np, torch, torch.nn as nn, pywt
from sklearn.metrics import roc_auc_score
from efficient_kan import KAN

HERE=os.path.dirname(os.path.abspath(__file__)); RAW=os.path.join(HERE,"data","raw")
def find(name):
    for p in [os.path.join(RAW,name), name, "/content/"+name]:
        if os.path.exists(p): return p
    raise FileNotFoundError(name)
ivt=np.load(find("ivt_sf_1980_2023.npy")).astype("float64")   # 6h
jet=np.load(find("circ_indices.npz"))["jet"].astype("float64")  # 일별 U250 제트

dmax=ivt.reshape(-1,4).max(1); ND=len(dmax); ar=dmax>250
def wlast(arr,lvl): return [c[-1] for c in pywt.swt(arr,'db2',level=lvl,trim_approx=True,norm=True)]
X=[]; y=[]; i=0
while i<ND:
    if ar[i]:
        j=i
        while j<ND and ar[j]: j+=1
        o=i; dur=j-i; end6=(o+1)*4
        if end6-128>=0 and o-63>=0:
            ivw=wlast(ivt[end6-128:end6], 6)   # 7  IVT wavelet 32일
            ivo=[dmax[o]]                       # 1  IVT onset
            ujo=[jet[o]]                        # 1  U250 jet onset
            ujw=wlast(jet[o-63:o+1], 5)         # 6  U250 jet wavelet 64일
            X.append(ivw+ivo+ujo+ujw); y.append(int(dur>=3))
        i=j
    else: i+=1
X=np.array(X); y=np.array(y); n=len(y)
i1=int(n*0.6); i2=int(n*0.8); trs=slice(0,i1); vas=slice(i1,i2); tes=slice(i2,n)
mu=X[trs].mean(0); sd=X[trs].std(0)+1e-8; X=(X-mu)/sd
dev="cuda" if torch.cuda.is_available() else "cpu"
Xt=torch.tensor(X,dtype=torch.float32).to(dev); yt=torch.tensor(y,dtype=torch.float32).to(dev)
print(f"n={n}, feat={X.shape[1]} (IVTwav7+IVTonset1+U250onset1+U250wav6), tr/va/te={i1}/{i2-i1}/{n-i2}, "
      f"pos% {y[trs].mean()*100:.0f}/{y[vas].mean()*100:.0f}/{y[tes].mean()*100:.0f}, dev={dev}")

torch.manual_seed(0)
net=KAN([X.shape[1],16,1]).to(dev)
opt=torch.optim.AdamW(net.parameters(),lr=5e-3,weight_decay=1e-3)
pw=torch.tensor((len(y[trs])-y[trs].sum())/max(y[trs].sum(),1),dtype=torch.float32,device=dev)
lossf=nn.BCEWithLogitsLoss(pos_weight=pw)
def auc(sl):
    net.eval()
    with torch.no_grad(): p=torch.sigmoid(net(Xt[sl]).squeeze(-1)).cpu().numpy()
    return roc_auc_score(y[sl],p)
best=-1; bs=None
for ep in range(300):
    net.train(); opt.zero_grad(); lossf(net(Xt[trs]).squeeze(-1),yt[trs]).backward(); opt.step()
    if ep%5==0:
        v=auc(vas)
        if v>best: best=v; bs=copy.deepcopy(net.state_dict())
net.load_state_dict(bs)
print(f"slim multiscale + KAN  val AUC={best:.3f}  test AUC={auc(tes):.3f}")
