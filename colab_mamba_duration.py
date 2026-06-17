# ===== Mamba로 AR onset -> 3일+ 지속 판별 (Colab GPU에서 python 직접 실행 가능) =====
# 입력 = raw 6h IVT 시퀀스 + wavelet 계수 시퀀스(척도별) 를 다채널로 함께 투입
import os, sys, subprocess
subprocess.run([sys.executable,"-m","pip","install","-q",
                "mamba-ssm","causal-conv1d","pywavelets"], check=False)

import numpy as np, torch, torch.nn as nn, pywt
from mamba_ssm import Mamba
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
CANDS = [os.path.join(HERE,"data/raw/ivt_sf_1980_2023.npy"),
         "ivt_sf_1980_2023.npy","/content/ivt_sf_1980_2023.npy",
         "/content/drive/MyDrive/Colab Notebooks/ivt_sf_1980_2023.npy",
         "data/raw/ivt_sf_1980_2023.npy"]
path = next((p for p in CANDS if os.path.exists(p)), None)
assert path, "ivt_sf_1980_2023.npy 를 업로드하거나 Drive에 두세요"
ivt = np.load(path).astype("float64")

SEQ=64
dmax=ivt.reshape(-1,4).max(1); ND=len(dmax); ar=dmax>250
seqs=[]; y=[]; i=0
while i<ND:
    if ar[i]:
        j=i
        while j<ND and ar[j]: j+=1
        o=i; dur=j-i; end6=(o+1)*4
        if end6-SEQ>=0:
            w=ivt[end6-SEQ:end6]                       # raw 6h IVT (64,)
            coeffs=pywt.swt(w,'db2',level=5,trim_approx=True,norm=True)  # [cA5,cD5..cD1] 각 (64,)
            chans=[np.log1p(w)]+list(coeffs)            # 7 채널: raw + cA5,cD5,cD4,cD3,cD2,cD1
            seqs.append(np.stack(chans,axis=1))         # (64, 7)
            y.append(int(dur>=3))
        i=j
    else: i+=1
X=np.array(seqs); y=np.array(y)                          # (N, 64, 7)
mu=X.reshape(-1,X.shape[-1]).mean(0); sd=X.reshape(-1,X.shape[-1]).std(0)+1e-8
X=(X-mu)/sd
X=torch.tensor(X,dtype=torch.float32); yt_all=torch.tensor(y,dtype=torch.float32)
dev="cuda"; C=X.shape[-1]
print(f"events={len(y)}, >=3day={y.sum()} ({y.mean()*100:.0f}%), channels={C} (raw+wavelet)")

class Net(nn.Module):
    def __init__(self,cin,d=32):
        super().__init__()
        self.inp=nn.Linear(cin,d)
        self.m1=Mamba(d_model=d,d_state=16,d_conv=4,expand=2)
        self.m2=Mamba(d_model=d,d_state=16,d_conv=4,expand=2)
        self.norm=nn.LayerNorm(d); self.head=nn.Linear(d,1)
    def forward(self,x):
        h=self.inp(x); h=h+self.m1(h); h=h+self.m2(h)
        h=self.norm(h).mean(1)
        return self.head(h).squeeze(-1)

pw=torch.tensor((len(y)-y.sum())/y.sum(),dtype=torch.float32,device=dev)
skf=StratifiedKFold(5,shuffle=True,random_state=0); aucs=[]
for tr,te in skf.split(np.zeros(len(y)),y):
    net=Net(C).to(dev); opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-2)
    lossf=nn.BCEWithLogitsLoss(pos_weight=pw)
    Xt=X[tr].to(dev); yt=yt_all[tr].to(dev); Xv=X[te].to(dev)
    for ep in range(60):
        net.train(); opt.zero_grad(); lossf(net(Xt),yt).backward(); opt.step()
    net.eval()
    with torch.no_grad(): p=torch.sigmoid(net(Xv)).cpu().numpy()
    aucs.append(roc_auc_score(y[te],p))
print(f"Mamba(raw+wavelet) 5-fold AUC = {np.mean(aucs):.3f} (+/-{np.std(aucs):.3f})")
print("(참고: wavelet+로지스틱 ~0.644)")
