# ===== Temporal Patch Encoder + KAN Classifier : AR onset -> 3일+ 지속 (in-sample) =====
# 입력: 6h IVT(+wavelet) 7채널 시퀀스(64스텝) -> 패치 임베딩 -> 1-layer Transformer -> KAN head
# 의존성: efficient-kan, pywavelets (Colab 셀에서 설치)
import os
import numpy as np, torch, torch.nn as nn, pywt
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
while i<ND:
    if ar[i]:
        j=i
        while j<ND and ar[j]: j+=1
        o=i; dur=j-i; end6=(o+1)*4
        if end6-SEQ>=0:
            w=ivt[end6-SEQ:end6]
            coeffs=pywt.swt(w,'db2',level=5,trim_approx=True,norm=True)
            chans=[np.log1p(w)]+list(coeffs)         # 7채널
            seqs.append(np.stack(chans,axis=1)); y.append(int(dur>=3))
        i=j
    else: i+=1
X=np.array(seqs); y=np.array(y)
mu=X.reshape(-1,X.shape[-1]).mean(0); sd=X.reshape(-1,X.shape[-1]).std(0)+1e-8
X=(X-mu)/sd
dev="cuda" if torch.cuda.is_available() else "cpu"
Xt=torch.tensor(X,dtype=torch.float32).to(dev); yt=torch.tensor(y,dtype=torch.float32).to(dev)
N,L,C=X.shape
print(f"events={N}, >=3day={y.sum()} ({y.mean()*100:.0f}%), L={L}, C={C}, dev={dev}")

class PatchKAN(nn.Module):
    def __init__(self,C,L=64,patch=8,d=24):
        super().__init__()
        self.P=L//patch; self.patch=patch; self.C=C
        self.embed=nn.Linear(patch*C,d)
        self.pos=nn.Parameter(torch.randn(1,self.P,d)*0.02)
        enc=nn.TransformerEncoderLayer(d,nhead=2,dim_feedforward=d*2,dropout=0.2,batch_first=True)
        self.enc=nn.TransformerEncoder(enc,num_layers=1)
        self.kan=KAN([d,16,1])
    def forward(self,x):
        n=x.shape[0]
        xp=x.reshape(n,self.P,self.patch*self.C)   # patchify (연속 patch스텝 x 채널)
        h=self.embed(xp)+self.pos
        h=self.enc(h).mean(1)
        return self.kan(h).squeeze(-1)

torch.manual_seed(0)
model=PatchKAN(C,L).to(dev)
opt=torch.optim.AdamW(model.parameters(),lr=3e-3,weight_decay=1e-3)
pw=torch.tensor((len(y)-y.sum())/y.sum(),dtype=torch.float32,device=dev)
lossf=nn.BCEWithLogitsLoss(pos_weight=pw)
for ep in range(150):
    model.train(); opt.zero_grad(); lossf(model(Xt),yt).backward(); opt.step()
model.eval()
with torch.no_grad(): p=torch.sigmoid(model(Xt)).cpu().numpy()
print(f"PatchEncoder+KAN 전체1폴드 AUC = {roc_auc_score(y,p):.3f}")
print("(참고: 로지스틱 0.661, LSTM 0.686, Mamba 0.606)")
