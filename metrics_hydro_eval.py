# -*- coding: utf-8 -*-
"""심사자1 3·5번 대응 산출 — (A) 수문·기상 표준 이진검증 지표, (B) 지속 AR 의 수문 신호,
(C) 판정 이동(회수/손실)의 수문 평가.

(A)(B) 는 모델 예측이 필요 없어 로컬에서 돈다.
(C) 는 pred_dump_{R}_24h.npz (colab_dump_pred.py 산출) 가 있어야 하므로 Colab 에서만 돈다.
표본은 전부 전수(달력일 창)로 통일한다 — 온셋시각 정렬판은 군별 탈락률이 달라(캘리포니아
미지속 51.4% 대 지속 74.5%) 두 군 비교에 쓸 수 없다.

출력: 표준출력
실행: python metrics_hydro_eval.py
"""
import os
import numpy as np
from scipy import stats



def P(*a):
    print(" ".join(str(x) for x in a))


GAUGE = {"11181040": ("San Lorenzo C", 32.5), "11460400": ("Lagunitas C", 37.1),
         "11162630": ("Pilarcitos C", 32.0), "11181000": ("San Lorenzo C Hayward", 39.0)}


def find(*cands):
    for c in cands:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(cands[0])


def load_precip(R):
    f = find(f"data/hydro/precip_hourly_{R}.npz",
             f"data/raw/precip_hourly_{'sf' if R == 'ca' else R}_2000_2019.npz")
    z = np.load(f, allow_pickle=True)
    t = z["times"].astype(str); p = z["precipitation"].astype("f8")
    day = np.array([s[:10] for s in t], dtype="datetime64[D]")
    ud, inv = np.unique(day, return_inverse=True)
    daily = np.zeros(len(ud)); np.add.at(daily, inv, p)
    return {d: i for i, d in enumerate(ud)}, daily


def load_q(sid):
    D = {}
    for L in open(find(f"data/hydro/q_{sid}.txt"), encoding="utf-8"):
        if L.startswith("#"):
            continue
        f = L.rstrip("\n").split("\t")
        if len(f) < 4 or f[0] != "USGS":
            continue
        try:
            D[np.datetime64(f[2])] = float(f[3])
        except ValueError:
            pass
    return D


def scores(y, pred):
    """POD/FAR/CSI/F1 — WMO 계열 이진 검증 지표."""
    TP = int((y & pred).sum()); FP = int((~y & pred).sum()); FN = int((y & ~pred).sum())
    POD = TP / max(TP + FN, 1)                 # = recall
    FAR = FP / max(TP + FP, 1)                 # = 1 - precision
    CSI = TP / max(TP + FP + FN, 1)            # threat score
    PRE = TP / max(TP + FP, 1)
    F1 = 2 * PRE * POD / max(PRE + POD, 1e-12)
    return dict(TP=TP, FP=FP, FN=FN, POD=POD, FAR=FAR, CSI=CSI, F1=F1)


def desc(x, u=""):
    if len(x) == 0:
        return "n=0"
    return (f"n={len(x):4d} 중앙 {np.median(x):7.2f} 평균 {np.mean(x):7.2f}{u} "
            f"IQR {np.percentile(x,25):6.2f}~{np.percentile(x,75):7.2f}")


# ══════════════════════════ (A) 이진 검증 지표 — 원예보 ══════════════════════════
P("=" * 100)
P("(A) 수문·기상 표준 이진 검증 지표 — 원예보 기준선 (테스트블록, 첫 블록 N//6 제외)")
P("    POD=probability of detection(=재현율)  FAR=false alarm ratio(=1-정밀도)  CSI=critical success index")
P("=" * 100)
for suf, lab in [("_18h", "18h"), ("", "24h"), ("_30h", "30h")]:
    for R in ["ca", "chile"]:
        z = np.load(f"opdenom_full_{R}{suf}.npz")
        y = z["y"].astype(bool); pred = z["fcv"] >= float(z["THR"])
        f = len(y) // 6
        s = scores(y[f:], pred[f:])
        P(f"  [{lab} {R:5s}] n={len(y)-f:4d}  TP={s['TP']:3d} FP={s['FP']:3d} FN={s['FN']:3d}  "
          f"POD={s['POD']:.3f}  FAR={s['FAR']:.3f}  CSI={s['CSI']:.3f}  F1={s['F1']:.3f}")

# ══════════════════════ (B) 지속 AR 의 수문 신호 — 전수, 모델 무관 ══════════════════════
P("")
P("=" * 100)
P("(B) 지속 AR 의 수문 신호 — 전수 표본(달력일 창). 모델과 무관.")
P("    P24=온셋일 UTC 하루,  P48=온셋일+다음날.  Q3=온셋일~+2일 첨두유량(USGS 일유량, cfs)")
P("=" * 100)
for R in ["ca", "chile"]:
    dmap, daily = load_precip(R)
    z = np.load(f"opdenom_full_{R}.npz")
    y = z["y"].astype(bool); on = z["omin"] >= 0          # 자리표시자, 아래에서 재정의
    oday = z["oday"].astype(int)
    dates = np.datetime64("1980-01-01") + oday.astype("timedelta64[D]")
    # 3군: 온셋 아님 / 온셋·미지속 / 온셋·지속  — y3 (0/1/2) 사용
    y3 = z["y3"].astype(int)
    idx = np.array([dmap.get(d, -1) for d in dates])
    ok = idx >= 0
    P24 = np.where(ok, daily[np.clip(idx, 0, len(daily) - 1)], np.nan)
    nxt = np.array([dmap.get(d + np.timedelta64(1, "D"), -1) for d in dates])
    P48 = P24 + np.where(nxt >= 0, daily[np.clip(nxt, 0, len(daily) - 1)], np.nan)
    P("")
    P(f"  ####### [{R}]  조인 {int(ok.sum())}/{len(ok)}")
    for lab, arr in [("P24", P24), ("P48", P48)]:
        P(f"    --- {lab} (mm) ---")
        g = []
        for k, nm in [(0, "온셋 아님   "), (1, "온셋·미지속"), (2, "온셋·지속  ")]:
            v = arr[(y3 == k) & ~np.isnan(arr)]; g.append(v)
            dry = 100 * (v < 0.1).mean() if len(v) else float("nan")
            P(f"      {nm} {desc(v)}  무강수 {dry:5.1f}%")
        P(f"      지속 vs 미지속  Mann-Whitney p={stats.mannwhitneyu(g[2],g[1])[1]:.3e}   "
          f"미지속 vs 온셋아님 p={stats.mannwhitneyu(g[1],g[0])[1]:.3e}")
    if R != "ca":
        continue
    P("    --- Q3 첨두유량 (cfs, 중앙값) ---")
    for sid, (nm, dist) in GAUGE.items():
        Q = load_q(sid)
        q3 = np.array([max([Q.get(d + np.timedelta64(k, "D"), np.nan) for k in (0, 1, 2)])
                       for d in dates])
        v = [q3[(y3 == k) & ~np.isnan(q3)] for k in (0, 1, 2)]
        p = stats.mannwhitneyu(v[2], v[1])[1]
        P(f"      {nm:24s}({dist:4.1f} km)  {np.median(v[0]):7.1f} / {np.median(v[1]):7.1f} / "
          f"{np.median(v[2]):7.1f}   비율 {np.median(v[2])/max(np.median(v[1]),1e-9):.2f}배  p={p:.2e}"
          f"   n={len(v[0])}/{len(v[1])}/{len(v[2])}")

# ══════════════════ (C) 판정 이동의 수문 평가 — pred_dump 필요 (Colab) ══════════════════
P("")
P("=" * 100)
P("(C) 판정 이동(회수/손실)의 수문 평가 — 테스트블록. pred_dump_{R}_24h.npz 필요")
P("=" * 100)
for R in ["ca", "chile"]:
    if not os.path.exists(f"pred_dump_{R}_24h.npz"):
        P(f"  [{R}] pred_dump 없음 -> 건너뜀 (Colab 에서 colab_dump_pred.py 먼저 실행)")
        continue
    z = np.load(f"pred_dump_{R}_24h.npz")
    y = z["y"].astype(bool); raw = z["raw"].astype(bool); tab = z["tabpfn"].astype(bool)
    dates = np.datetime64("1980-01-01") + z["oday"].astype(int).astype("timedelta64[D]")
    sr, st = scores(y, raw), scores(y, tab)
    P("")
    P(f"  ####### [{R}] n={len(y)}  실제 지속 {int(y.sum())}건")
    P(f"    원예보  POD={sr['POD']:.3f} FAR={sr['FAR']:.3f} CSI={sr['CSI']:.3f} F1={sr['F1']:.3f}"
      f"  (TP={sr['TP']} FP={sr['FP']} FN={sr['FN']})")
    P(f"    TabPFN  POD={st['POD']:.3f} FAR={st['FAR']:.3f} CSI={st['CSI']:.3f} F1={st['F1']:.3f}"
      f"  (TP={st['TP']} FP={st['FP']} FN={st['FN']})")
    P(f"    변화    dPOD={st['POD']-sr['POD']:+.3f} dFAR={st['FAR']-sr['FAR']:+.3f} "
      f"dCSI={st['CSI']-sr['CSI']:+.3f} dF1={st['F1']-sr['F1']:+.3f}")

    gain = ~raw & tab & y; loss = raw & ~tab & y; both = raw & tab & y
    P(f"    회수 {int(gain.sum())}건 / 손실 {int(loss.sum())}건 = 순증 {int(gain.sum())-int(loss.sum())}건"
      f"   오경보 {sr['FP']} -> {st['FP']} ({st['FP']-sr['FP']:+d})")

    dmap, daily = load_precip(R)
    idx = np.array([dmap.get(d, -1) for d in dates])
    nxt = np.array([dmap.get(d + np.timedelta64(1, "D"), -1) for d in dates])
    P24 = np.where(idx >= 0, daily[np.clip(idx, 0, len(daily) - 1)], np.nan)
    P48 = P24 + np.where(nxt >= 0, daily[np.clip(nxt, 0, len(daily) - 1)], np.nan)
    for lab, arr in [("P24", P24), ("P48", P48)]:
        P(f"    --- {lab} 강수 (mm) ---")
        for nm, m in [("둘 다 판정", both), ("회수", gain), ("손실", loss)]:
            P(f"      {nm:10s} {desc(arr[m & ~np.isnan(arr)])}")
        a = arr[gain & ~np.isnan(arr)]; b = arr[loss & ~np.isnan(arr)]
        if len(a) and len(b):
            P(f"      회수 vs 손실  Mann-Whitney p={stats.mannwhitneyu(a,b)[1]:.3e}")
    # 원예보 FN vs TP (심사자 1-5 직접 대응)
    fn = y & ~raw; tp = y & raw
    for lab, arr in [("P24", P24), ("P48", P48)]:
        a = arr[fn & ~np.isnan(arr)]; b = arr[tp & ~np.isnan(arr)]
        P(f"    {lab} 원예보 FN vs TP: 중앙 {np.median(a):.2f} 대 {np.median(b):.2f}  "
          f"p={stats.mannwhitneyu(a,b)[1]:.3e}   놓친 이벤트 총강수 비중 "
          f"{100*np.nansum(a)/max(np.nansum(a)+np.nansum(b),1e-9):.1f}%")
    if R != "ca":
        continue
    P("    --- Q3 첨두유량 (cfs, 중앙값) 회수 vs 손실 ---")
    med = {}
    for sid, (nm, dist) in GAUGE.items():
        Q = load_q(sid)
        q3 = np.array([max([Q.get(d + np.timedelta64(k, "D"), np.nan) for k in (0, 1, 2)])
                       for d in dates])
        v = {k: q3[m & ~np.isnan(q3)] for k, m in [("both", both), ("gain", gain), ("loss", loss)]}
        med[nm] = [np.median(v["both"]), np.median(v["gain"]), np.median(v["loss"])]
        p = stats.mannwhitneyu(v["gain"], v["loss"])[1] if len(v["gain"]) and len(v["loss"]) else np.nan
        P(f"      {nm:24s} 둘다 {med[nm][0]:7.1f} / 회수 {med[nm][1]:7.1f} / 손실 {med[nm][2]:7.1f}"
          f"   회수 vs 손실 p={p:.3f}")
    M = np.array([med[k] for k in med])                      # 4유역 x 3군
    P(f"      Friedman(3군)  chi2={stats.friedmanchisquare(*M.T)[0]:.3f}  "
      f"p={stats.friedmanchisquare(*M.T)[1]:.4f}   순위합 "
      f"{list(stats.rankdata(M,axis=1).sum(0))}")
    w = stats.wilcoxon(M[:, 1], M[:, 2])
    P(f"      회수 vs 손실 직접검정  Wilcoxon signed-rank p={w[1]:.4f}   "
      f"(유역 4개 -> 부호검정 최소 p=0.125, 유의 불가)")
    P(f"      회수가 '둘 다 판정'보다 낮은 유역 "
      f"{int((M[:,1] < M[:,0]).sum())}/4")

# ══════════════ (D) 원예보 대 모델 — 잡아낸 지속 이벤트 전체 비교 ══════════════
P("")
P("=" * 100)
P("(D) 원예보가 잡은 지속 이벤트 vs 모델이 잡은 지속 이벤트 (겹치는 38건 포함, 전체 집합)")
P("=" * 100)
for R in ["ca", "chile"]:
    if not os.path.exists(f"pred_dump_{R}_24h.npz"):
        P(f"  [{R}] pred_dump 없음 -> 건너뜀")
        continue
    z = np.load(f"pred_dump_{R}_24h.npz")
    y = z["y"].astype(bool); raw = z["raw"].astype(bool); tab = z["tabpfn"].astype(bool)
    dates = np.datetime64("1980-01-01") + z["oday"].astype(int).astype("timedelta64[D]")
    dmap, daily = load_precip(R)
    idx = np.array([dmap.get(d, -1) for d in dates])
    nxt = np.array([dmap.get(d + np.timedelta64(1, "D"), -1) for d in dates])
    P24 = np.where(idx >= 0, daily[np.clip(idx, 0, len(daily) - 1)], np.nan)
    P48 = P24 + np.where(nxt >= 0, daily[np.clip(nxt, 0, len(daily) - 1)], np.nan)
    A = y & raw          # 원예보가 잡은 지속
    B = y & tab          # 모델이 잡은 지속
    P("")
    P(f"  ####### [{R}] 실제 지속 {int(y.sum())}건 중 원예보 {int(A.sum())}건 / 모델 {int(B.sum())}건")
    for lab, arr in [("P24", P24), ("P48", P48)]:
        tot = np.nansum(arr[y])
        a, b = np.nansum(arr[A]), np.nansum(arr[B])
        P(f"    {lab} 총강수(mm)  실제지속 전체 {tot:8.1f} | 원예보 {a:8.1f} ({100*a/tot:4.1f}%) "
          f"| 모델 {b:8.1f} ({100*b/tot:4.1f}%)  회수분 {b-a:+8.1f} ({100*(b-a)/tot:+.1f}%p)")
        P(f"          중앙값  원예보 {np.nanmedian(arr[A]):6.2f} | 모델 {np.nanmedian(arr[B]):6.2f}")
    if R != "ca":
        continue
    P("    --- Q3 첨두유량 (cfs) ---")
    for sid, (nm, dist) in GAUGE.items():
        Q = load_q(sid)
        q3 = np.array([max([Q.get(d + np.timedelta64(k, "D"), np.nan) for k in (0, 1, 2)])
                       for d in dates])
        ta = np.nansum(q3[y]); qa, qb = np.nansum(q3[A]), np.nansum(q3[B])
        P(f"      {nm:24s} 합계 실제지속 {ta:9.0f} | 원예보 {qa:9.0f} ({100*qa/ta:4.1f}%) "
          f"| 모델 {qb:9.0f} ({100*qb/ta:4.1f}%)   중앙 {np.nanmedian(q3[A]):7.1f} -> "
          f"{np.nanmedian(q3[B]):7.1f}")

# ══════════ (E) 판정 이동의 유의성 + 오경보 비용 (표 8 보정) ══════════
P("")
P("=" * 100)
P("(E) McNemar 정확검정(회수 대 손실) + 오경보일에 실린 강수 + 경보예산 동일화")
P("=" * 100)
from scipy.stats import binomtest
for R in ["ca", "chile"]:
    if not os.path.exists(f"pred_dump_{R}_24h.npz"):
        P(f"  [{R}] pred_dump 없음 -> 건너뜀")
        continue
    z = np.load(f"pred_dump_{R}_24h.npz")
    y = z["y"].astype(bool); raw = z["raw"].astype(bool); tab = z["tabpfn"].astype(bool)
    dates = np.datetime64("1980-01-01") + z["oday"].astype(int).astype("timedelta64[D]")
    gain = int((~raw & tab & y).sum()); loss = int((raw & ~tab & y).sum())
    mc = binomtest(gain, gain + loss, 0.5)
    P("")
    P(f"  ####### [{R}] 회수 {gain} / 손실 {loss}  McNemar 정확검정 p={mc.pvalue:.4f}")
    sr, st = scores(y, raw), scores(y, tab)
    P(f"    판정 건수 {sr['TP']+sr['FP']} -> {st['TP']+st['FP']}   "
      f"FAR {sr['FAR']:.3f} -> {st['FAR']:.3f}")
    dmap, daily = load_precip(R)
    idx = np.array([dmap.get(d, -1) for d in dates])
    nxt = np.array([dmap.get(d + np.timedelta64(1, "D"), -1) for d in dates])
    P24 = np.where(idx >= 0, daily[np.clip(idx, 0, len(daily) - 1)], np.nan)
    P48 = P24 + np.where(nxt >= 0, daily[np.clip(nxt, 0, len(daily) - 1)], np.nan)
    fr, ft = raw & ~y, tab & ~y                      # 오경보
    for lab, arr in [("P24", P24), ("P48", P48)]:
        P(f"    {lab} 오경보일 총강수(mm)  원예보 {np.nansum(arr[fr]):7.1f} ({int(fr.sum())}건, "
          f"중앙 {np.nanmedian(arr[fr]):5.2f}) | 모델 {np.nansum(arr[ft]):7.1f} "
          f"({int(ft.sum())}건, 중앙 {np.nanmedian(arr[ft]):5.2f})")
    # 경보예산 동일화: 모델을 원예보와 같은 판정 건수로 잘랐을 때 (예측 최소IVT 상위 k)
    if "pred_min" in z.files:
        k = sr["TP"] + sr["FP"]
        thr_k = np.sort(z["pred_min"])[::-1][k - 1]
        cut = z["pred_min"] >= thr_k
        sc = scores(y, cut)
        P(f"    경보예산 동일화(k={k})  모델 POD={sc['POD']:.3f} FAR={sc['FAR']:.3f} "
          f"F1={sc['F1']:.3f}  (원예보 POD={sr['POD']:.3f} FAR={sr['FAR']:.3f} F1={sr['F1']:.3f})")
        for lab, arr in [("P24", P24), ("P48", P48)]:
            tot = np.nansum(arr[y])
            P(f"      {lab} 회수율  원예보 {100*np.nansum(arr[y&raw])/tot:4.1f}% | "
              f"동일예산 모델 {100*np.nansum(arr[y&cut])/tot:4.1f}%")
    else:
        P("    경보예산 동일화: pred_dump 에 pred_min 없음 -> colab_dump_pred.py 에 추가 필요")
