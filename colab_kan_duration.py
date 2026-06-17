# ===== KAN으로 AR onset -> 3일+ 지속 판별 (전체 1폴드 = in-sample) =====
# KAN은 시퀀스모델 아님 -> wavelet onset 피처 6개(로지스틱 0.661 낸 것)를 입력.
# 의존성: efficient-kan, pywavelets (Colab 셀에서 설치)
import os
import numpy as np, torch
import pywt
from sklearn.metrics import roc_auc_score
from efficient_kan import KAN

HERE = os.path.dirname(os.path.abspath(__file__))
CANDS = [os.path.join(HERE,"data/raw/ivt_sf_1980_2023.npy"),
         "ivt_sf_1980_2023.npy","/content/ivt_sf_1980_2023.npy"]
path = next((p for p in CANDS if os.path.exists(p)), None)
assert path, "ivt_sf_1980_2023.npy 없음"
ivt = np.load(path).astype("float64")

SEQ=64
dmax=ivt.reshape(-1,4).max(1); ND=len(dmax); ar=dmax>250
X=[]; y=[]; i=0
while i<ND:
    if ar[i]:
        j=i
        while j<ND and ar[j]: j+=1
        o=i; dur=j-i; end6=(o+1)*4
        if end6-SEQ>=0:
            w=ivt[end6-SEQ:end6]
            wav=[c[-1] for c in pywt.swt(w,'db2',level=5,trim_approx=True,norm=True)]  # 6개
            X.append(wav); y.append(int(dur>=3))
        i=j
    else: i+=1
X=np.array(X); y=np.array(y)
X=(X-X.mean(0))/(X.std(0)+1e-8)
dev="cuda" if torch.cuda.is_available() else "cpu"
Xt=torch.tensor(X,dtype=torch.float32).to(dev); yt=torch.tensor(y,dtype=torch.float32).to(dev)
print(f"events={len(y)}, >=3day={y.sum()} ({y.mean()*100:.0f}%), feat={X.shape[1]}, dev={dev}")

torch.manual_seed(0)
model=KAN([X.shape[1], 16, 1]).to(dev)
opt=torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-3)
pw=torch.tensor((len(y)-y.sum())/y.sum(),dtype=torch.float32,device=dev)
lossf=torch.nn.BCEWithLogitsLoss(pos_weight=pw)
for ep in range(200):
    model.train(); opt.zero_grad()
    out=model(Xt).squeeze(-1); lossf(out,yt).backward(); opt.step()
model.eval()
with torch.no_grad(): p=torch.sigmoid(model(Xt).squeeze(-1)).cpu().numpy()
print(f"KAN(wavelet onset) 전체1폴드 AUC = {roc_auc_score(y,p):.3f}")
print("(참고: 같은 피처 로지스틱 in-sample 0.661, LSTM 0.686, Mamba 5-fold 0.606)")
