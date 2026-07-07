# ===== Colab: 피처 정렬 ablation 전체판 — peak정렬 vs 고정리드 × 6모델 (+24h: env 인코더·웨이블릿 재검증) =====
# 질문 1: 예보 peak 정렬(A)이 고정 리드(B·C)보다 나은가 — 6모델 전체에서 (기존 TabPFN·LGBM 2모델 -> 6모델 확장).
# 질문 2: 고정리드 base(C) 위에서 관측 env(인코더8 / IVT웨이블릿6 직접)가 이득 있나 — 전 horizon(18/24/30h).
#         "env는 예보에 흡수" 폐기결론이 peak정렬(D2) 기준으로 잰 것이라 고정리드 base 로 재검증.
# 피처셋: A_peak9 = D2(peak정렬 9) / B_fix8 = D8(리드48~90 고정 8) / C_fix8+요약 = D8 + D2 요약4(min/mean/std/기울기)
#         D_C+인코더8 = C + env44->MLP 8차원(3클래스 헤드, 폴드별 train만 학습) / E_C+웨이블릿6 = C + IVT 16일 웨이블릿 직접
# raw 기준선 = fcv(peak정렬) 고정. 회귀 -> omin 예측 -> THR 판정 -> F1. 3지역 × 18/24/30h + 통합(지역평균 ΔF1 부트스트랩).
# LSTM: 시퀀스 = A는 peak정렬 궤적 5점, B~E는 고정리드 8점 / 시퀀스 외 피처(요약4·인코더8·웨이블릿6)는 LSTM 마지막 출력에 결합(fc 입력).
# 누수방지: 인코더·모델 전부 매 폴드 train 에서만 학습 (walk-forward 5폴드, 64일 임베고).
# 준비: opdenom_full_{r}{,_18h,_30h}.npz (git pull). !pip install tabpfn tabicl lightgbm pytorch-tabnet -q
import warnings, logging, os
warnings.filterwarnings("ignore")                              # sklearn feature-name / numpy overflow 등 파이썬 warning 전부 차단
os.environ["PYTHONWARNINGS"] = "ignore"
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)   # HF_TOKEN 미설정 경고(로거 경유) 차단
logging.getLogger("pytorch_tabnet").setLevel(logging.ERROR)
import numpy as np, torch, torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from lightgbm import LGBMRegressor
from tabpfn import TabPFNRegressor
from tabicl import TabICLRegressor
from pytorch_tabnet.tab_model import TabNetRegressor

DEV = "cuda" if torch.cuda.is_available() else "cpu"; NB = 2000; REGIONS = ["ca", "uk", "chile"]
HORIZONS = [("_18h", "18h"), ("", "24h"), ("_30h", "30h")]
LGBM_HP = dict(num_leaves=15, learning_rate=0.03, n_estimators=200, min_child_samples=20)   # 블록0 시간순CV 튜닝값(colab_opdenom.py)
IVW = slice(0, 6)                                              # ENV 앞 6열 = IVT 16일 웨이블릿 (build_op_denom_full.py env_feats 순서)


# ---------- 모델 6종 (공통 시그니처: (Xtr, ytr, Xte, seq_n) -> 예측 omin. seq_n 은 LSTM 만 사용) ----------
def p_lr(Xtr, ytr, Xte, seq_n):
    sc = StandardScaler().fit(np.nan_to_num(Xtr))
    return LinearRegression().fit(sc.transform(np.nan_to_num(Xtr)), ytr).predict(sc.transform(np.nan_to_num(Xte)))
def p_lgbm(Xtr, ytr, Xte, seq_n):
    return LGBMRegressor(subsample=0.8, verbose=-1, **LGBM_HP).fit(np.nan_to_num(Xtr), ytr).predict(np.nan_to_num(Xte))
def p_tabpfn(Xtr, ytr, Xte, seq_n):
    m = TabPFNRegressor(device=DEV); m.fit(np.nan_to_num(Xtr), ytr); return np.asarray(m.predict(np.nan_to_num(Xte)))
def p_tabicl(Xtr, ytr, Xte, seq_n):
    m = TabICLRegressor(); m.fit(np.nan_to_num(Xtr), ytr); return np.asarray(m.predict(np.nan_to_num(Xte)))
def p_tabnet(Xtr, ytr, Xte, seq_n):
    sc = StandardScaler().fit(np.nan_to_num(Xtr)); bs = max(16, len(Xtr) // 4)
    m = TabNetRegressor(verbose=0, device_name=DEV, seed=0)
    m.fit(sc.transform(np.nan_to_num(Xtr)).astype("float32"), ytr.reshape(-1, 1).astype("float32"),
          max_epochs=150, batch_size=bs, virtual_batch_size=bs, drop_last=True)
    return m.predict(sc.transform(np.nan_to_num(Xte)).astype("float32")).ravel()


class LSTMreg(nn.Module):                                      # 시퀀스(예보 IVT 궤적) + 정적피처(요약/인코더/웨이블릿) 결합 회귀
    def __init__(s, nstat=0, h=16):
        super().__init__(); s.lstm = nn.LSTM(1, h, batch_first=True); s.fc = nn.Linear(h + nstat, 1)
    def forward(s, xs, xa):
        o, _ = s.lstm(xs); z = o[:, -1] if xa is None else torch.cat([o[:, -1], xa], 1); return s.fc(z).squeeze(-1)
def p_lstm(Xtr, ytr, Xte, seq_n):
    Xtr = np.nan_to_num(Xtr); Xte = np.nan_to_num(Xte)
    sc = StandardScaler().fit(Xtr); Ztr = sc.transform(Xtr); Zte = sc.transform(Xte)
    nstat = Xtr.shape[1] - seq_n
    xstr = torch.tensor(Ztr[:, :seq_n][:, :, None], dtype=torch.float32, device=DEV)
    xste = torch.tensor(Zte[:, :seq_n][:, :, None], dtype=torch.float32, device=DEV)
    xatr = torch.tensor(Ztr[:, seq_n:], dtype=torch.float32, device=DEV) if nstat else None
    xate = torch.tensor(Zte[:, seq_n:], dtype=torch.float32, device=DEV) if nstat else None
    ym, ys = float(ytr.mean()), float(ytr.std() + 1e-8); ytn = torch.tensor((ytr - ym) / ys, dtype=torch.float32, device=DEV)
    torch.manual_seed(0); net = LSTMreg(nstat).to(DEV); opt = torch.optim.Adam(net.parameters(), lr=1e-2); lf = nn.MSELoss()
    rng = np.random.default_rng(0); n = len(ytr)
    for ep in range(200):
        net.train(); perm = rng.permutation(n)
        for b in range(0, n, 128):
            bi = torch.as_tensor(perm[b:b + 128], device=DEV); opt.zero_grad()
            lf(net(xstr[bi], None if xatr is None else xatr[bi]), ytn[bi]).backward(); opt.step()
    net.eval()
    with torch.no_grad(): return net(xste, xate).cpu().numpy() * ys + ym


MODELS = {"LR": p_lr, "LGBM": p_lgbm, "LSTM": p_lstm, "TabPFN": p_tabpfn, "TabICL": p_tabicl, "TabNet": p_tabnet}


# ---------- env 인코더 (77a33f2 원본과 동일 구조: 44 -> 16 -> 8차원 병목 -> 3클래스 헤드) ----------
class MLPenc(nn.Module):
    def __init__(s, cin=44, hid=8, w=16, ncls=3):
        super().__init__(); s.f1 = nn.Linear(cin, w); s.dp = nn.Dropout(0.3); s.f2 = nn.Linear(w, hid); s.head = nn.Linear(hid, ncls)
    def rep(s, x): return torch.relu(s.f2(s.dp(torch.relu(s.f1(x)))))
    def forward(s, x): return s.head(s.rep(x))
def fit_enc(ENVtr, y3tr, hid=8, epochs=300):                   # 폴드 train 에서만 인코더 학습 -> 임의 행의 8차원 rep 반환
    mu = np.nanmean(ENVtr, 0); sd = np.nanstd(ENVtr, 0) + 1e-8
    Xtr = torch.tensor(np.nan_to_num((ENVtr - mu) / sd), dtype=torch.float32, device=DEV)
    yt = torch.tensor(y3tr, dtype=torch.long, device=DEV)
    cnt = np.bincount(y3tr, minlength=3).astype(float); cw = len(y3tr) / (3 * np.maximum(cnt, 1))   # 역빈도 클래스가중
    lf = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device=DEV))
    torch.manual_seed(0); net = MLPenc(ENVtr.shape[1], hid).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
    rng = np.random.default_rng(0); n = len(y3tr)
    for ep in range(epochs):
        net.train(); perm = rng.permutation(n)
        for b in range(0, n, 64):
            bi = torch.as_tensor(perm[b:b + 64], device=DEV); opt.zero_grad(); lf(net(Xtr[bi]), yt[bi]).backward(); opt.step()
    net.eval()
    def rep(E):
        with torch.no_grad(): return net.rep(torch.tensor(np.nan_to_num((E - mu) / sd), dtype=torch.float32, device=DEV)).cpu().numpy()
    return rep


# ---------- 폴드 / 부트스트랩 ----------
def folds(N, od, Nf=5, emb=64):
    f = N // (Nf + 1); out = []
    for k in range(1, Nf + 1):
        ts = k * f; te = (k + 1) * f if k < Nf else N
        out.append((np.array([j for j in range(0, ts) if od[j] <= od[ts] - emb]), np.arange(ts, te)))
    return out
def boot_p(yt, pa, pb_):                                       # P(모델 F1 > 기준 F1), 케이스 부트스트랩
    rng = np.random.default_rng(0); dd = []
    for _ in range(NB):
        ix = rng.integers(0, len(yt), len(yt))
        if len(np.unique(yt[ix])) > 1: dd.append(f1_score(yt[ix], pa[ix], zero_division=0) - f1_score(yt[ix], pb_[ix], zero_division=0))
    return np.mean(np.array(dd) > 0)


# ---------- 실행: horizon × (지역별 피처셋 폴드 캐시 -> 6모델 공유) ----------
for suf, hlab in HORIZONS:
    print(f"\n========== {hlab} : 피처 정렬 ablation (raw=peak-fcv 고정) ==========")
    DATA = {}
    for R in REGIONS:
        fn = f"opdenom_full_{R}{suf}.npz"
        if not os.path.exists(fn): print(f"[{R} {hlab}] {fn} 없음 -> skip"); continue
        d = np.load(fn)
        D2 = d["D2"]; D8 = d["D8"]; fcv = d["fcv"]; y = d["y"]; omin = d["omin"]; oday = d["oday"]; THR = float(d["THR"])
        C = np.column_stack([D8, D2[:, 5:9]])
        FL = [(tr, te) for tr, te in folds(len(y), oday) if len(tr) >= 40 and len(np.unique(y[tr])) > 1]
        FEAT = {"A_peak9": (5, [(D2[tr], D2[te]) for tr, te in FL]),
                "B_fix8": (8, [(D8[tr], D8[te]) for tr, te in FL]),
                "C_fix8+요약": (8, [(C[tr], C[te]) for tr, te in FL])}
        if "ENV" in d.files and "y3" in d.files:                # env 재검증 세트 — 전 horizon (npz 에 ENV·y3 동봉)
            ENV = d["ENV"]; y3 = d["y3"].astype(int)
            dl = []
            for tr, te in FL:                                   # 인코더는 폴드당 1회만 학습 -> 6모델이 같은 rep 공유
                rep = fit_enc(ENV[tr], y3[tr])
                dl.append((np.column_stack([C[tr], rep(ENV[tr])]), np.column_stack([C[te], rep(ENV[te])])))
            FEAT["D_C+인코더8"] = (8, dl)
            FEAT["E_C+웨이블릿6"] = (8, [(np.column_stack([C[tr], ENV[tr][:, IVW]]), np.column_stack([C[te], ENV[te][:, IVW]])) for tr, te in FL])
        yt0 = np.concatenate([y[te] for _, te in FL])
        p0 = np.concatenate([(fcv[te] >= THR).astype(int) for _, te in FL])
        DATA[R] = dict(FEAT=FEAT, FL=FL, omin=omin, THR=THR, yt0=yt0, p0=p0)
    if not DATA: continue
    SETNAMES = list(next(iter(DATA.values()))["FEAT"].keys())

    for mname, pf in MODELS.items():
        print(f"  --- {mname} ---")
        REG = {}
        for R, dd in DATA.items():
            f0 = f1_score(dd["yt0"], dd["p0"], zero_division=0); line = f"    [{R}] raw={f0:.3f}"
            REG[R] = {"_raw": dd["p0"], "_yt": dd["yt0"]}
            for sn in SETNAMES:
                seq_n, fl = dd["FEAT"][sn]; pb = []
                for (Xtr, Xte), (tr, te) in zip(fl, dd["FL"]):
                    pred = np.asarray(pf(Xtr, dd["omin"][tr], Xte, seq_n))
                    pb.extend((pred >= dd["THR"]).astype(int))
                pb = np.array(pb); ff = f1_score(dd["yt0"], pb, zero_division=0)
                line += f" | {sn}={ff:.3f}(Δ{ff - f0:+.3f},P{boot_p(dd['yt0'], pb, dd['p0']):.2f})"
                REG[R][sn] = pb
            print(line)
        for sn in SETNAMES:                                     # 통합(지역평균 ΔF1, 지역 부트스트랩)
            rng = np.random.default_rng(0); md = []
            for _ in range(NB):
                ds = []
                for R in REG:
                    yt0 = REG[R]["_yt"]; p0 = REG[R]["_raw"]; pm = REG[R][sn]; ix = rng.integers(0, len(yt0), len(yt0))
                    if len(np.unique(yt0[ix])) < 2: ds = None; break
                    ds.append(f1_score(yt0[ix], pm[ix], zero_division=0) - f1_score(yt0[ix], p0[ix], zero_division=0))
                if ds is not None: md.append(np.mean(ds))
            md = np.array(md)
            obs = np.mean([f1_score(REG[R]["_yt"], REG[R][sn], zero_division=0) - f1_score(REG[R]["_yt"], REG[R]["_raw"], zero_division=0) for R in REG])
            print(f"    [통합 {sn}] 평균Δ{obs:+.3f} CI[{np.percentile(md,2.5):+.3f},{np.percentile(md,97.5):+.3f}] P{np.mean(md>0):.3f}")

        # ===== B vs C 직접 비교 (메인 피처 결정용): ΔF1 = C − B, 같은 test 표본 paired 부트스트랩 =====
        if "B_fix8" in SETNAMES and "C_fix8+요약" in SETNAMES:
            rng = np.random.default_rng(0); md = []
            for _ in range(NB):
                ds = []
                for R in REG:
                    yt0 = REG[R]["_yt"]; pB = REG[R]["B_fix8"]; pC = REG[R]["C_fix8+요약"]; ix = rng.integers(0, len(yt0), len(yt0))
                    if len(np.unique(yt0[ix])) < 2: ds = None; break
                    ds.append(f1_score(yt0[ix], pC[ix], zero_division=0) - f1_score(yt0[ix], pB[ix], zero_division=0))
                if ds is not None: md.append(np.mean(ds))
            md = np.array(md)
            pers = " | ".join(f"{R} Δ{f1_score(REG[R]['_yt'], REG[R]['C_fix8+요약'], zero_division=0) - f1_score(REG[R]['_yt'], REG[R]['B_fix8'], zero_division=0):+.3f} P{boot_p(REG[R]['_yt'], REG[R]['C_fix8+요약'], REG[R]['B_fix8']):.2f}" for R in REG)
            obs = np.mean([f1_score(REG[R]["_yt"], REG[R]["C_fix8+요약"], zero_division=0) - f1_score(REG[R]["_yt"], REG[R]["B_fix8"], zero_division=0) for R in REG])
            print(f"    [C−B 직접] {pers} | 통합 Δ{obs:+.3f} CI[{np.percentile(md,2.5):+.3f},{np.percentile(md,97.5):+.3f}] P{np.mean(md>0):.3f}")
