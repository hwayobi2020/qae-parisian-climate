# -*- coding: utf-8 -*-
"""심사자 2 지적 2-7 대응 — §3.8 소프트웨어 표를 채울 값을 Colab 환경에서 수집한다.
패키지명·버전·라이선스·배포처와 파이썬/CUDA 를 한 번에 출력한다.
Colab 셀: !python colab_env_versions.py   (또는 이 파일 내용을 셀에 붙여넣기)
"""
import importlib.metadata as md
import platform
import sys

# 논문 모델 -> 패키지
PKGS = [
    ("TabPFN", "tabpfn"),
    ("TabICL", "tabicl"),
    ("TabNet", "pytorch-tabnet"),
    ("GBDT (LightGBM)", "lightgbm"),
    ("Linear regression", "scikit-learn"),
    ("LSTM / 1d-CNN", "torch"),
    ("(numeric base)", "numpy"),
]


def field(meta, *names):
    for n in names:
        v = meta.get(n)
        if v and v.strip() and v.strip().lower() != "unknown":
            return v.strip()
    return ""


def license_of(meta):
    lic = field(meta, "License-Expression", "License")
    if lic and len(lic) < 60:
        return lic
    for c in meta.get_all("Classifier") or []:
        if c.startswith("License ::"):
            return c.split("::")[-1].strip()
    return lic[:60] if lic else "(메타데이터 없음)"


def url_of(meta):
    for k in (meta.get_all("Project-URL") or []):
        low = k.lower()
        if low.startswith(("homepage", "source", "repository")):
            return k.split(",", 1)[-1].strip()
    return field(meta, "Home-page") or "(없음)"


print(f"Python {sys.version.split()[0]}   platform {platform.platform()}")
try:
    import torch
    print(f"torch CUDA {torch.version.cuda}   GPU {torch.cuda.get_device_name(0) if torch.cuda.is_available() else '없음'}")
except Exception as e:
    print("torch 확인 실패:", e)
print()
print(f"{'모델':20s} {'패키지':18s} {'버전':12s} {'라이선스':22s} 배포처")
for label, pkg in PKGS:
    try:
        meta = md.metadata(pkg)
        print(f"{label:20s} {pkg:18s} {md.version(pkg):12s} {license_of(meta):22s} "
              f"https://pypi.org/project/{pkg}/  |  {url_of(meta)}")
    except md.PackageNotFoundError:
        print(f"{label:20s} {pkg:18s} {'미설치':12s}")
print()
print("주의: Colab 은 세션마다 최신 버전을 설치하므로, 이 출력은 '이 셀을 돌린 시점'의 값이다.")
print("      논문에 싣기 전에 결과 산출에 쓴 런과 같은 환경인지 확인할 것.")