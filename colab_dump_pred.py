# ===== Colab: TabPFN 표본별 예측 덤프 (심사자1 3·5번 수문 검증용) =====
# 목적: 모델이 원예보 대비 어떤 이벤트를 새로 잡아내는지 표본 단위로 특정한다.
#       기존 스크립트는 F1 만 내고 표본별 예측을 저장하지 않아 유량·강수 결합이 불가능했다.
# 구성은 colab_final_2reg.py 와 동일: 피처 B(D8, 고정리드 48~90h), 타깃 omin,
#       워크포워드 5폴드(64일 임베고), 판정 = 예측 omin >= THR.
# 준비: git pull 로 opdenom_full_{ca,chile}.npz. !pip install tabpfn -q
# 출력: pred_dump_{ca,chile}_24h.npz  {oday, y, raw, tabpfn, omin_true, omin_pred, fcv, THR}
#       -> git add -f 해서 로컬로 가져온다.
import warnings, logging, os
warnings.filterwarnings("ignore"); logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
import numpy as np, torch
from sklearn.metrics import f1_score
from tabpfn import TabPFNRegressor

DEV = "cuda" if torch.cuda.is_available() else "cpu"
REGIONS = ["ca", "chile"]


def folds(N, od, Nf=5, emb=64):
    """colab_final_2reg.py:83 과 동일."""
    f = N // (Nf + 1); out = []
    for k in range(1, Nf + 1):
        ts = k * f; te = (k + 1) * f if k < Nf else N
        out.append((np.array([j for j in range(0, ts) if od[j] <= od[ts] - emb]), np.arange(ts, te)))
    return out


for R in REGIONS:
    d = np.load(f"opdenom_full_{R}.npz")
    D8 = d["D8"]; fcv = d["fcv"]; y = d["y"]; omin = d["omin"]; oday = d["oday"]
    THR = float(d["THR"])

    FL = [(tr, te) for tr, te in folds(len(y), oday) if len(tr) >= 40 and len(np.unique(y[tr])) > 1]
    te_all = np.concatenate([te for _, te in FL])

    pred_omin = np.full(len(y), np.nan)
    for tr, te in FL:
        m = TabPFNRegressor(device=DEV)
        m.fit(np.nan_to_num(D8[tr]), omin[tr])
        pred_omin[te] = np.asarray(m.predict(np.nan_to_num(D8[te])))

    yt = y[te_all]
    raw = (fcv[te_all] >= THR).astype(int)
    tab = (pred_omin[te_all] >= THR).astype(int)

    np.savez(f"pred_dump_{R}_24h.npz",
             oday=oday[te_all], y=yt, raw=raw, tabpfn=tab,
             omin_true=omin[te_all], omin_pred=pred_omin[te_all],
             fcv=fcv[te_all], THR=THR)

    f_raw = f1_score(yt, raw, zero_division=0)
    f_tab = f1_score(yt, tab, zero_division=0)
    # 원예보가 놓친 것(FN) 중 모델이 회수한 비율 = 이번 검증의 핵심 수치
    fn = (yt == 1) & (raw == 0)
    rec = int((fn & (tab == 1)).sum())
    print(f"[{R}] n={len(yt)}  raw F1={f_raw:.3f}  TabPFN F1={f_tab:.3f}  dF1={f_tab-f_raw:+.3f}")
    print(f"      원예보 FN {int(fn.sum())}건 중 모델 회수 {rec}건 ({100*rec/max(int(fn.sum()),1):.1f}%)")
    print(f"      saved pred_dump_{R}_24h.npz")

print("\n로컬로 가져오기:  git add -f pred_dump_*.npz && git commit -m 'pred dump' && git push")
