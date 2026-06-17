# ===== LSTM로 AR onset -> 3일+ 지속 판별 (전체 1폴드 = in-sample) =====
# 입력 = raw 6h IVT 시퀀스 + wavelet 계수 시퀀스(척도별) 다채널. (mamba wheel 불필요, torch만)
import os
import numpy as np, torch, torch.nn as nn, pywt
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
CANDS = [os.path.join(HERE,"data/raw/ivt_sf_1980_2023.npy"),
         "ivt_sf_1980_2023.npy","/content/ivt_sf_1980_2023.npy",
         "/content/drive/MyDrive/Colab Notebooks/ivt_sf_1980_2023.npy",
         "data/raw/ivt_sf_1980_2023.npy"]
path = next((p for p in CANDS if os.path.exists(p)), None)
assert path, "ivt_sf_1980_2023.npy 없음"
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
            w=ivt[end6-SEQ:end6]
            coeffs=pywt.swt(w,'db2',level=5,trim_approx=True,norm=True)
            chans=[np.log1p(w)]+list(coeffs)        # 7채널
            seqs.append(np.stack(chans,axis=1))
            y.append(int(dur>=3))
        i=j
    else: i+=1
X=np.array(seqs); y=np.array(y)
mu=X.reshape(-1,X.shape[-1]).mean(0); sd=X.reshape(-1,X.shape[-1]).std(0)+1e-8
X=(X-mu)/sd
dev="cuda" if torch.cuda.is_available() else "cpu"
Xt=torch.tensor(X,dtype=torch.float32).to(dev); yt=torch.tensor(y,dtype=torch.float32).to(dev)
C=X.shape[-1]
print(f"events={len(y)}, >=3day={y.sum()} ({y.mean()*100:.0f}%), channels={C}, dev={dev}")

class Net(nn.Module):
    def __init__(self,cin,hid=48):
        super().__init__()
        self.lstm=nn.LSTM(cin,hid,batch_first=True,bidirectional=True)
        self.drop=nn.Dropout(0.3); self.head=nn.Linear(hid*2,1)
    def forward(self,x):
        out,_=self.lstm(x); h=out.mean(1)
        return self.head(self.drop(h)).squeeze(-1)

torch.manual_seed(0)
net=Net(C).to(dev)
opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-2)
pw=torch.tensor((len(y)-y.sum())/y.sum(),dtype=torch.float32,device=dev)
lossf=nn.BCEWithLogitsLoss(pos_weight=pw)
for ep in range(80):
    net.train(); opt.zero_grad(); lossf(net(Xt),yt).backward(); opt.step()
net.eval()
with torch.no_grad(): p=torch.sigmoid(net(Xt)).cpu().numpy()
print(f"LSTM(raw+wavelet) 전체1폴드 AUC = {roc_auc_score(y,p):.3f}")
print("(참고: wavelet+로지스틱 in-sample 0.661, Mamba 5-fold 0.606)")
