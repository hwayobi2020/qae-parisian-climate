"""[Colab] 지역 일반화 지속기간 임계값 스윕 + TabPFN.
캘리포니아 파이프라인 그대로, AR 임계값만 "그 지역 IVT 일별최대의 85번째 백분위수"로 자동 계산.
독립 모델 (지역 간 풀링 없음). 동일 20피처, 시간순 60/20 홀드아웃.

사용법:  python threshold_sweep_region.py uk     (또는 chile, ca)
  - ca   : ivt_sf_1980_2023.npy   + circ_indices.npz     (임계값 250 = 85th 검증용)
  - uk   : ivt_uk_1980_2023.npy   + circ_indices_uk.npz
  - chile: ivt_chile_1980_2023.npy+ circ_indices_chile.npz
Colab 셀:
  !pip install tabpfn -q
  import os; os.environ["TABPFN_API_KEY"]="키"
  %cd /content/drive/MyDrive/Colab Notebooks/qae-parisian-climate
  !git pull -q && python threshold_sweep_region.py uk
"""
import os, sys, numpy as np, pywt, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

REGION = sys.argv[1] if len(sys.argv) > 1 else "uk"
IVT_FILE = {"ca": "ivt_sf_1980_2023.npy"}.get(REGION, f"ivt_{REGION}_1980_2023.npy")
CIRC_FILE = {"ca": "circ_indices.npz"}.get(REGION, f"circ_indices_{REGION}.npz")
HERE = os.path.dirname(os.path.abspath(__file__)); RAW = os.path.join(HERE, "data", "raw")

def find(name):
    for p in [os.path.join(RAW, name), name, "/content/" + name,
              os.path.join("/content/qae-parisian-climate/data/raw", name)]:
        if os.path.exists(p): return p
    raise FileNotFoundError(name)

ivt = np.load(find(IVT_FILE)).astype("float64")
ci = np.load(find(CIRC_FILE)); jet = ci["jet"].astype("float64"); blk = ci["blocking"].astype("float64")
dmax = ivt.reshape(-1, 4).max(1); ND = len(dmax)
THR = np.percentile(dmax, 85)                       # 지역별 AR 임계값 = 85th percentile
ar = dmax > THR
print(f"[{REGION}] IVT mean {ivt.mean():.0f}, AR 임계값(85th)={THR:.0f}, AR-day={ar.mean()*100:.1f}%, "
      f"6h steps={len(ivt)}, days={ND}")

def wl(a, lvl): return [c[-1] for c in pywt.swt(a, 'db2', level=lvl, trim_approx=True, norm=True)]
X = []; durs = []; i = 0
while i < ND:
    if ar[i]:
        j = i
        while j < ND and ar[j]: j += 1
        o = i; dur = j - i; end6 = (o + 1) * 4
        if end6 - 64 >= 0 and o - 63 >= 0 and not (np.isnan(jet[o]) or np.isnan(blk[o])):
            X.append(wl(ivt[end6 - 64:end6], 5) + [dmax[o], jet[o]] +
                     wl(jet[o - 63:o + 1], 5) + wl(blk[o - 63:o + 1], 5)); durs.append(dur)
        i = j
    else: i += 1
X = np.array(X); durs = np.array(durs); n = len(durs); i1 = int(n * 0.6); i2 = int(n * 0.8)
u, c = np.unique(durs, return_counts=True)
print(f"이벤트수 n={n}, train={i1} test={n - i2}, 지속분포(일:개수): "
      + ", ".join(f"{int(d)}:{int(cc)}" for d, cc in zip(u, c) if d <= 8) + " ...")
try:
    from tabpfn import TabPFNClassifier; HAVE_TP = True
except Exception as e:
    HAVE_TP = False; print(f"[TabPFN 미설치: {e}]")
print(f"\n{'임계값':>7} {'양성%':>6} {'Logistic':>9} {'RF(5시드)':>15} {'TabPFN(5시드)':>17} {'test양성':>8}")
for thr in [2, 3, 4, 5]:
    y = (durs >= thr).astype(int)
    sc = StandardScaler().fit(X[:i1]); Xtr, Xte = sc.transform(X[:i1]), sc.transform(X[i2:]); ytr, yte = y[:i1], y[i2:]
    if len(np.unique(yte)) < 2 or len(np.unique(ytr)) < 2:
        print(f"{thr:>6}일 {y.mean()*100:>5.0f}%  클래스부족 스킵(test양성={int(yte.sum())})"); continue
    lr = LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced").fit(Xtr, ytr)
    al = roc_auc_score(yte, lr.predict_proba(Xte)[:, 1])
    rf = np.array([roc_auc_score(yte, RandomForestClassifier(n_estimators=400, max_depth=4, min_samples_leaf=20,
                  class_weight="balanced", random_state=sd).fit(Xtr, ytr).predict_proba(Xte)[:, 1]) for sd in range(5)])
    if HAVE_TP:
        tp = np.array([roc_auc_score(yte, TabPFNClassifier(random_state=sd).fit(Xtr, ytr).predict_proba(Xte)[:, 1]) for sd in range(5)])
        tps = f"{tp.mean():.3f}±{tp.std():.3f}"
    else: tps = "(미설치)"
    print(f"{thr:>6}일 {y.mean()*100:>5.0f}%   {al:>7.3f}   {rf.mean():.3f}±{rf.std():.3f}   {tps:>15} {int(yte.sum()):>8}")
