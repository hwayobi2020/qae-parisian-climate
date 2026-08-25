# ===== Colab: 원예보 vs TabPFN 4칸 분할표 =====
# colab_dump_pred.py 가 저장한 pred_dump_{R}_24h.npz 를 읽어 판정 이동을 표로 낸다.
import numpy as np
for R in ["ca", "chile"]:
    z = np.load(f"pred_dump_{R}_24h.npz")
    y = z["y"].astype(bool); raw = z["raw"].astype(bool); tab = z["tabpfn"].astype(bool)
    P = y                      # 실제 지속
    both  = ( raw &  tab)
    gain  = (~raw &  tab)      # 모델만 지속으로 판정
    loss  = ( raw & ~tab)      # 원예보만 지속으로 판정
    none_ = (~raw & ~tab)
    print(f"=== {R}  n={len(y)}  실제 지속 {int(P.sum())}건 ===")
    print(f"  {'':22s} {'실제 지속(회수/손실)':>20s} {'실제 미지속(오경보)':>20s}")
    for nm, m in [("둘 다 지속판정", both), ("모델만 판정(gain)", gain),
                  ("원예보만 판정(loss)", loss), ("둘 다 미판정", none_)]:
        print(f"  {nm:22s} {int((m&P).sum()):>20d} {int((m&~P).sum()):>20d}")
    print(f"  -> 회수 {int((gain&P).sum())}건 / 손실 {int((loss&P).sum())}건 "
          f"= 순증 {int((gain&P).sum())-int((loss&P).sum())}건")
    print(f"  -> 오경보 원예보 {int((raw&~P).sum())}건 -> 모델 {int((tab&~P).sum())}건 "
          f"({int((tab&~P).sum())-int((raw&~P).sum()):+d})")
    print()
