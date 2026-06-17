# ===== Temporal Patch Encoder + KAN : AR onset -> 3일+ 지속 =====
# train/val/test (시간순 60/20/20), 표준화는 train만, val 조기종료, test AUC 보고
import os
import numpy as np, torch, torch.nn as nn, pywt, copy
from sklearn.metrics import roc_auc_score
from efficient_kan import KAN

HERE=os.path.dirname(os.path.abspath(__file__))
CANDS=[os.path.join(HERE,"data/raw/ivt_sf_1980_2023.npy"),
       "ivt_sf_1980_2023.npy","/content/ivt_sf_1980_2023.npy"]
path=next((p for p in CANDS if os.path.exists(p)),None); assert path
ivt=np.load(path).astype("float64")

SEQ=64
dmax=ivt.reshape(-1,4).max(1); ND=len(dmax); ar=dmax>250
seqs=[]; y=[]; i=0
while i<ND:                      # 이벤트는 시간순으로 수집됨
    if ar[i]:
        j=i
        while j<ND and ar[j]: j+=1
        o=i; dur=j-i; end6=(o+1)*4
        if end6-SEQ>=0:
            w=ivt[end6-SEQ:end6]
            coeffs=pywt.swt(w,'db2',level=5,trim_approx=True,norm=True)
            seqs.append(np.stack([np.log1p(w)]+list(coeffs),axis=1)); y.append(int(dur>=3))
        i=j
    else: i+=1
X=np.array(seqs); y=np.array(y); n=len(y); C=X.shape[-1]; L=X.shape[1]

# 시간순 분할
i1=int(n*0.6); i2=int(n*0.8)
trs=slice(0,i1); vas=slice(i1,i2); tes=slice(i2,n)
mu=X[trs].reshape(-1,C).mean(0); sd=X[trs].reshape(-1,C).std(0)+1e-8   # train만으로 표준화
X=(X-mu)/sd
dev="cuda" if torch.cuda.is_available() else "cpu"
Xt=torch.tensor(X,dtype=torch.float32).to(dev); yt=torch.tensor(y,dtype=torch.float32).to(dev)
print(f"n={n}, train={i1} val={i2-i1} test={n-i2}, pos% tr/va/te="
      f"{y[trs].mean()*100:.0f}/{y[vas].mean()*100:.0f}/{y[tes].mean()*100:.0f}, dev={dev}")

class PatchKAN(nn.Module):
    def __init__(s,C,L=64,patch=8,d=24):
        super().__init__()
        s.P=L//patch; s.patch=patch; s.C=C
        s.embed=nn.Linear(patch*C,d); s.pos=nn.Parameter(torch.randn(1,s.P,d)*0.02)
        enc=nn.TransformerEncoderLayer(d,2,d*2,dropout=0.2,batch_first=True)
        s.enc=nn.TransformerEncoder(enc,1); s.kan=KAN([d,16,1])
    def forward(s,x):
        m=x.shape[0]; xp=x.reshape(m,s.P,s.patch*s.C)
        h=s.embed(xp)+s.pos; h=s.enc(h).mean(1)
        return s.kan(h).squeeze(-1)

torch.manual_seed(0)
net=PatchKAN(C,L).to(dev)
opt=torch.optim.AdamW(net.parameters(),lr=3e-3,weight_decay=1e-3)
pw=torch.tensor((len(y[trs])-y[trs].sum())/max(y[trs].sum(),1),dtype=torch.float32,device=dev)
lossf=nn.BCEWithLogitsLoss(pos_weight=pw)
def auc(sl):
    net.eval()
    with torch.no_grad(): p=torch.sigmoid(net(Xt[sl])).cpu().numpy()
    return roc_auc_score(y[sl],p)
best_val=-1; best=None
for ep in range(300):
    net.train(); opt.zero_grad(); lossf(net(Xt[trs]),yt[trs]).backward(); opt.step()
    if ep%5==0:
        v=auc(vas)
        if v>best_val: best_val=v; best=copy.deepcopy(net.state_dict())
net.load_state_dict(best)
print(f"PatchEncoder+KAN  val AUC={best_val:.3f}  test AUC={auc(tes):.3f}")
print("(in-sample 0.994는 과적합이었음. 이게 정직한 hold-out 수치)")
