"""D-2 예보 피처(풍부): 온셋 주변 리드별 IVT 궤적 [온셋,+6,+12,+18,+24h]=D-2리드[48+OH..72+OH] + min/mean/std.
transfer의 s에 정렬. 저장 d2feat_{region}.npz {s, D2, D2min}.  사용: python export_d2feat.py
"""
import numpy as np
OFFS = [0, 6, 12, 18, 24]         # 온셋 기준 (D-2 리드 = 48+OH+off)
for REGION in ["ca", "uk", "chile"]:
    z = np.load(f"gefs_ivt_{REGION}_d2c00.npz"); S2 = z["s"].astype(int); OH2 = z["onset_hour"].astype(int)
    leads = list(z["leads"]); IVT = z["ivt"]   # (온셋,1멤버,리드)
    ss = []; D2 = []; D2min = []
    for k in range(len(S2)):
        Ls = [48 + OH2[k] + off for off in OFFS]
        if not all(L in leads for L in Ls): continue
        v = [IVT[k, 0, leads.index(L)] for L in Ls]
        if any(np.isnan(x) for x in v): continue
        cont = [v[1], v[2], v[3]]                       # 연속 3점(+6,+12,+18)
        feat = v + [min(cont), float(np.mean(cont)), float(np.std(cont)), v[4] - v[0]]  # 궤적5 + min/mean/std + (온셋+24)-(온셋) 기울기
        ss.append(int(S2[k])); D2.append(feat); D2min.append(min(cont))
    np.savez(f"d2feat_{REGION}.npz", s=np.array(ss), D2=np.array(D2, float), D2min=np.array(D2min, float))
    print(f"{REGION}: D-2 예보온셋 {len(ss)} | D2 피처 {np.array(D2).shape[1]}차원 (궤적5+min/mean/std+기울기)")
