"""NWP(GEFSv12 재예보) forecast IVT — 프록시 없이 실제 IVT (절충 충실도).
면: 1000·850(_pres) + 700·500(_abv700mb).  선행: 0·12·24·36·48·60·72h.
효율: 메시지 단위 byte-range(딱 필요한 전지구 필드만), 점 추출, keep-alive 세션, 체크포인트·재개.
IVT = (1/g)|∫ q V dp|, V=(u,v). g=9.80665. 단위 kg/(m·s).
누수: init=onset 당일 00Z. 출력 gefs_ivt_{REGION}.npz (s,e,onset_hour,leads,levels,u,q,v,ivt,ivtu,ivtv,THR).
사용: python gefs_ivt.py [Y0 Y1] [NTEST]
"""
import numpy as np, requests, eccodes as ec, os, sys, time, warnings, threading
from concurrent.futures import ThreadPoolExecutor
warnings.filterwarnings("ignore")
ECLOCK = threading.Lock()                # eccodes 비-thread-safe → 디코드 직렬화

B = "https://noaa-gefs-retrospective.s3.amazonaws.com/"
REGION = os.environ.get("GEFS_REGION", "ca")
_PTS = {"ca": (37.77, 237.58), "uk": (50.0, 355.0), "chile": (-33.0, 288.5)}
_IVT = {"ca": "ivt_sf_1980_2023.npy", "uk": "ivt_uk_1980_2023.npy", "chile": "ivt_chile_1980_2023.npy"}
_TIM = {"ca": "times_sf_1980_2023.npy", "uk": "times_uk_1980_2023.npy", "chile": "times_chile_1980_2023.npy"}
LAT, LON = _PTS[REGION]
MEMBERS = os.environ.get("GEFS_MEMBERS", "c00,p01,p02,p03,p04").split(",")
LEADS = [int(x) for x in os.environ.get("GEFS_LEADS", "0,12,24,36,48,60,72").split(",")]
PRES_LV = [1000, 850]                 # _pres 파일에서
ABV_LV = [700, 500]                   # _abv700mb 파일에서
ALL_LV = sorted(PRES_LV + ABV_LV, reverse=True)   # [1000,850,700,500]
VARS = ["ugrd", "vgrd", "spfh"]
G = 9.80665
Y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
Y1 = int(sys.argv[2]) if len(sys.argv) > 2 else 2019
NTEST = int(sys.argv[3]) if len(sys.argv) > 3 else 0
WORKERS = int(os.environ.get("GEFS_WORKERS", "10"))


def find(name):
    for p in ["data/raw/" + name, "/content/" + name,
              "/content/qae-parisian-climate/data/raw/" + name,
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw", name)]:
        if os.path.exists(p): return p
    raise FileNotFoundError(name)


def url(var, init, m, abv):
    tag = var + ("_pres_abv700mb" if abv else "_pres")
    return B + requests.utils.quote(f"GEFSv12/reforecast/{init[:4]}/{init}/{m}/Days:1-10/{tag}_{init}_{m}.grib2")


def lead_of(stepstr):
    t = stepstr.split()
    if not t: return None
    if t[0] == "anl": return 0
    return int(t[0]) if t[0].isdigit() else None


def fetch_msgs(sess, var, init, m, abv, wantlv):
    """딱 필요한 (level in wantlv, lead in LEADS) 메시지만 개별 Range로 받아 점값 추출."""
    base = url(var, init, m, abv)
    for _ in range(3):
        try:
            idx = sess.get(base + ".idx", timeout=60).text.splitlines()
            rows = []
            for ln in idx:
                p = ln.split(":")
                if len(p) >= 6:
                    rows.append((int(p[1]), p[4].split()[0], lead_of(p[5])))
            rows.sort()
            offs = [r[0] for r in rows]
            want = []                                  # (off, endbyte, lv, ld)
            for i, (off, lvs, ld) in enumerate(rows):
                lv = int(lvs) if lvs.isdigit() else None
                if lv in wantlv and ld in LEADS:
                    end = offs[i + 1] - 1 if i + 1 < len(offs) else ""
                    want.append((off, end, lv, ld))
            out = {}
            for off, end, lv, ld in want:
                r = sess.get(base, headers={"Range": f"bytes={off}-{end}"}, timeout=120)
                if r.status_code not in (200, 206): continue
                b = r.content; q = 0; N = len(b)
                with ECLOCK:
                    while q + 16 <= N:
                        if b[q:q + 4] != b"GRIB":
                            q += 1; continue
                        ln = int.from_bytes(b[q + 8:q + 16], "big")
                        if ln <= 0 or q + ln > N: break
                        gid = ec.codes_new_from_message(b[q:q + ln]); q += ln
                        try:
                            out[(lv, ld)] = ec.codes_grib_find_nearest(gid, LAT, LON)[0].value
                        finally:
                            ec.codes_release(gid)
            return out
        except Exception:
            time.sleep(2)
    return {}


# --- onsets ---
ivt = np.load(find(_IVT[REGION])).astype("float64")
times = np.load(find(_TIM[REGION]))
THR = np.percentile(ivt.reshape(-1, 4).max(1), 85); ar = ivt > THR; T = len(ivt)
runs = []; i = 0
while i < T:
    if ar[i]:
        j = i
        while j < T and ar[j]: j += 1
        runs.append((i, j)); i = j
    else: i += 1
OFFSET = int(os.environ.get("GEFS_INIT_OFFSET_DAYS", "0"))   # 발표를 온셋 N일 전으로
ons = []
if os.environ.get("GEFS_NEARMISS"):                             # near-miss(거짓경보 후보) 날짜 리스트로 다운로드
    z = np.load(f"nearmiss_{REGION}.npz")
    for k in range(len(z["s0"])):
        ons.append((int(z["s0"][k]), int(z["s0"][k]), str(z["init_str"][k]), 0))   # (s0, s0, D-2발표, onset_hour=0)
else:
    for s, e in runs:
        ts = str(times[s])
        if Y0 <= int(ts[:4]) <= Y1:
            d0 = times[s].astype("datetime64[D]") - np.timedelta64(OFFSET, "D")
            istr = str(d0).replace("-", "") + "00"
            if int(istr[:4]) < 2000: continue
            ons.append((s, e, istr, int(ts[11:13])))
if NTEST: ons = ons[:NTEST]
n = len(ons); nlv = len(ALL_LV); nlead = len(LEADS)
LVpos = {lv: k for k, lv in enumerate(ALL_LV)}; Lpos = {L: li for li, L in enumerate(LEADS)}
print(f"[GEFS IVT] {REGION} {Y0}-{Y1}: onsets={n}, 멤버={len(MEMBERS)}, 변수={VARS}, "
      f"면={ALL_LV}, 선행={LEADS}, msg/onset·멤버≈{len(VARS)*2*2*nlead}, workers={WORKERS}", flush=True)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"gefs_ivt_{REGION}{os.environ.get('GEFS_SUFFIX','')}.npz")
U = np.full((n, len(MEMBERS), nlv, nlead), np.nan); V = np.full_like(U, np.nan); Q = np.full_like(U, np.nan)
VARARR = {"ugrd": U, "vgrd": V, "spfh": Q}
if os.path.exists(OUT):                              # --- 재개 ---
    z = np.load(OUT)
    if z["u"].shape == U.shape:
        U[:] = z["u"]; V[:] = z["v"]; Q[:] = z["q"]
        print(f"  재개: 기존 {OUT} 로드", flush=True)

tasks = [(oi, mi, vi) for oi in range(n) for mi in range(len(MEMBERS)) for vi in range(len(VARS))
         if np.isnan(VARARR[VARS[vi]][oi, mi]).all()]
print(f"  남은 작업 {len(tasks)}/{n*len(MEMBERS)*len(VARS)}", flush=True)

_tl = {}
def work(t):
    oi, mi, vi = t; var = VARS[vi]; init = ons[oi][2]; m = MEMBERS[mi]
    sess = _tl.get("s")
    if sess is None: sess = _tl["s"] = requests.Session()
    d = fetch_msgs(sess, var, init, m, False, PRES_LV)
    d.update(fetch_msgs(sess, var, init, m, True, ABV_LV))
    return (oi, mi, var, d)


def save():
    np.savez(OUT, s=[o[0] for o in ons], e=[o[1] for o in ons], onset_hour=[o[3] for o in ons],
             leads=LEADS, levels=ALL_LV, u=U, v=V, q=Q, THR=THR)


t0 = time.time(); done = 0
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    for oi, mi, var, d in ex.map(work, tasks):
        arr = VARARR[var]
        for (lv, ld), val in d.items():
            if lv in LVpos and ld in Lpos: arr[oi, mi, LVpos[lv], Lpos[ld]] = val
        done += 1
        if done % 50 == 0:
            el = time.time() - t0
            print(f"  {done}/{len(tasks)}  {el:.0f}s  eta {el/done*(len(tasks)-done):.0f}s", flush=True)
            save()
save()

# --- IVT 적분 ---
pPa = np.array(ALL_LV, float) * 100.0
ivtu = np.full((n, len(MEMBERS), nlead), np.nan); ivtv = np.full_like(ivtu, np.nan); IVT = np.full_like(ivtu, np.nan)
for oi in range(n):
    for mi in range(len(MEMBERS)):
        for li in range(nlead):
            q = Q[oi, mi, :, li]; u = U[oi, mi, :, li]; v = V[oi, mi, :, li]
            ok = ~(np.isnan(q) | np.isnan(u) | np.isnan(v))
            if ok.sum() < 3: continue
            pp = pPa[ok]; o2 = np.argsort(pp)
            iu = np.trapz((q * u)[ok][o2], pp[o2]) / G; iv = np.trapz((q * v)[ok][o2], pp[o2]) / G
            ivtu[oi, mi, li] = iu; ivtv[oi, mi, li] = iv; IVT[oi, mi, li] = np.hypot(iu, iv)
np.savez(OUT, s=[o[0] for o in ons], e=[o[1] for o in ons], onset_hour=[o[3] for o in ons],
         leads=LEADS, levels=ALL_LV, u=U, v=V, q=Q, ivt=IVT, ivtu=ivtu, ivtv=ivtv, THR=THR)
print(f"저장 {OUT}  ({time.time()-t0:.0f}s)", flush=True)

# --- sanity: onset 시점 forecast IVT vs 관측 ---
oh = np.array([o[3] for o in ons]); ss = np.array([o[0] for o in ons])
fc_on = []
for oi in range(n):
    cand = [L for L in LEADS if L >= oh[oi]]; L0 = cand[0] if cand else LEADS[-1]
    fc_on.append(np.nanmean(IVT[oi, :, Lpos[L0]]))
fc_on = np.array(fc_on); obs_on = ivt[ss]; m = ~np.isnan(fc_on)
print(f"\n[sanity] AR THR={THR:.0f} | onset forecast IVT 평균{np.nanmean(fc_on):.0f} 중앙{np.nanmedian(fc_on):.0f} "
      f"(관측 onset IVT 평균{obs_on.mean():.0f}) 유효{m.sum()}/{n}", flush=True)
if m.sum() > 3:
    from scipy.stats import spearmanr
    print(f"[sanity] forecast vs 관측 onset IVT  ρ={spearmanr(fc_on[m], obs_on[m]).correlation:+.2f}", flush=True)
