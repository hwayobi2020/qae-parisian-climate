# qae-parisian-climate — 대기강(Atmospheric River) 지속기간 예측가능성

> **2026-06-17 프로젝트 전면 전환.** 이 프로젝트는 더 이상 양자컴퓨터(양자 진폭추정 기반 경로의존 옵션 가격결정) 프로젝트가 **아니다.**
> 이전 양자 단계 작업은 보존: `archive/CLAUDE_quantum_legacy.md`(당시 작업지침), `archive/`(소스), `chat/session_2026-04-*.{jsonl,md}`.
> (디렉토리명 `qae-parisian-climate`는 legacy 명칭이며 현재 연구 내용과 무관하다.)

## 현재 연구 (정체성)

**대기강(atmospheric river, AR — 수증기 수송량 IVT가 임계를 넘는 좁고 긴 대기 흐름)이 발생한 시점(onset)의 대규모 환경만으로, 그 이벤트가 며칠 지속할지 예측 가능한가.**

연구 정체성 = **(A) 예측가능성·메커니즘 과학** (예보 도구가 아님).
- 수치예보(NWP)는 시간별 IVT 궤적을 예보해 지속시간을 부산물로 얻는다 → "예보 성능"으로 NWP와 경쟁하는 건 불리.
- 우리 주장 = *"AR 지속성이 onset 시점의 대규모 환경에 의해 예측 가능한 정도와 그 물리적 원천을 규명한다."*
- gap(검증 완료): AR 예보 skill 문헌(DeFlorio 2018 전구 / Nayak 2014 / NHESS 2020 이베리아)은 발생·위치·강도만 다루고, **지속성(duration)의 예측가능성 정량화는 미답.**

## 확정 결과 (캘리포니아 SF, 단일지점 1287 events)

- 본 기준 = AR onset → **≥4일 지속** 이진분류 (sweet spot; ≥2일은 흔해 변별↓, 임계 올릴수록 AUC↑)
- AUC ~0.77 (Logistic ≈ RF ≈ TabPFN — paired bootstrap상 ≥4일 셋 다 통계적 동급)
- 신호원 = **IVT 16일 wavelet + U250 상층 제트**. **Z500 블로킹·SST는 무효**
- 공간풀링(격자 45셀)=사실상 오버샘플링 → 천장 못 올림. 앙상블(스태킹/평균)=단일모델 대비 유의 이득 없음. **둘 다 폐기**

## 설계 원칙 (절대 준수)

- 3지역(**캘리포니아 SF / 영국 콘월 50N,-5W / 칠레 발파라이소 33S,-71.5W**)은 **완전 독립 모델. 풀링 절대 금지** (지형이 달라 IVT→피해 전환이 다름).
- 지역 간 연결은 오직 "**같은 방법론이 각 지역에서 독립적으로 재현되는가**"라는 비교뿐 — 재현되면 그게 일반화 입증.
- AR 임계값 = **지역별 IVT 일별최대의 85번째 백분위수** (캘리포니아 250 = 정확히 85th; 영국은 배경 IVT 높아 절대 250이면 AR-day 47% → 반드시 85th로 조정).

## 데이터 파이프라인

- **ERA5 IVT**: `reanalysis-era5-single-levels` 격자 area 박스 netcdf로 받아 nearest 셀 추출 (timeseries API는 viwve/viwvn 미지원 — MultiAdaptorNoDataError). 2년 청크 이어받기. IVT = sqrt(viwve² + viwvn²).
- **순환지수**: NCEP/NCAR OPeNDAP에서 영역 Z500/U250 → blocking = 영역평균 Z500의 day-of-year anomaly, jet = 영역 U250 공간최대. times[::4]로 일별 정렬.
- **활성 스크립트**:
  - `scripts/download_era5_ivt_{uk,chile}.py` — 지역 IVT 다운로드 (좌표만 다름)
  - `scripts/build_circ_uk.py` — 영국 북대서양 순환지수 (칠레는 남반구 영역으로 복제 예정)
  - `threshold_sweep_region.py [ca|uk|chile]` — 85th 임계값 자동 + ≥2/3/4/5일 스윕 + Logistic/RF/TabPFN
- 데이터 파일(.nc/.npy/.csv)은 `.gitignore`. `ivt_sf_1980_2023.npy`·`circ_indices.npz`만 force-add 되어 있음.
- **TabPFN은 Colab 전용** (로컬 torch c10.dll 깨짐). Colab = private repo git pull + python 워크플로.

## 출판 목표

**Q2 SCI** (Journal of Hydrometeorology / NHESS·Natural Hazards / International Journal of Climatology).
전제: ① 3지역 독립 재현 ② persistence(지속성) baseline 대비 유의 skill ③ 메커니즘(합성장 등) 분석.

## 현재 진행 (2026-06-17)

- 영국·칠레 IVT 다운로드 중 (청크당 ~8.5분 × 22청크, 이어받기).
- 다음 순서: 다운로드 완료 → `build_circ_uk.py` → Colab `threshold_sweep_region.py uk`(영국 재현 검증) → 칠레 순환지수(남반구 남동태평양 영역) → 칠레 분석 → **persistence baseline 비교**(미착수).

## 작업 규칙

- 한국어. 수치 검증 없이 확정(bluffing) 금지. 실패·한계 숨기지 말 것.
- **지역 모델 절대 섞지 말 것**(풀링 금지).
- 컨펌 없이 코드 수정·대규모 git 작업 시작 금지.
- 메모리: `[[project-parisian-climate-ar-duration]]`.
