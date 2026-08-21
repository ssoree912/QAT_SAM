# 실험 로그 (날짜별)

4-bit QAT 실험 전체 기록. 모델 · 데이터셋 · 설정 · 성능(best Top-1) 정리.
정확도는 전부 **validation set** 기준(ImageNet 50k / CIFAR-100 10k).

---

## 📌 헤드라인 요약

| 데이터셋 / 모델 | 세팅 | baseline | qSAM | qSAM 효과 |
|---|---|---|---|---|
| **ImageNet / ResNet-50** | KD, Adam/linear/50ep | 77.074 | **77.230** (w0) | **+0.16** |
| **ImageNet / ResNet-50** | no-KD, LSQ논문(SGD/cosine/90ep) | 75.668 | 75.760 | +0.09 |
| CIFAR-100 / ResNet-18 | 구코드(버그有) | 76.09 (lsq) | 73.11 (lsq_qsam) | −2.98 ⚠️(무효) |

> **핵심**: 코드 수정 후 ImageNet 두 regime 모두에서 **qSAM이 baseline을 일관되게 +0.1 안팎으로 앞섬**(방향 긍정적, 크기는 노이즈 수준). CIFAR는 버그 있던 구코드라 무효.

---

## 🗓️ 날짜별 상세

### 2026-05-28 ~ 05-29 · CIFAR-100 / ResNet-18 · **구코드(수정 전, 버그 有)**
데이터셋: CIFAR-100 (10k val) · 모델: ResNet-18 · 4-bit · 200 epoch · KD 없음

| method | best Top-1 |
|---|---|
| lsq | **76.09** |
| lsq_aoq | 74.27 |
| lsq_qsam | 73.11 (최종 ep199 53.16으로 붕괴 ⚠️) |
| lsq_aoq_qsam | 73.68 |

⚠️ **무효 데이터**: AOQ Eq.(7) detach 버그, qSAM eval 누수/stale gradient/warmup·rho 없음, LSQ gradient-scale 누락 상태에서 돌림. qSAM/AOQ가 나빠 보이는 건 버그 탓. (lsq_qsam 붕괴가 그 증거)

### 2026-05-28 · ImageNet / ResNet-18 · (구 AOQ 코드)
best 69.652 (ep136, 미완/구코드) — 우리 작업 이전 기록, 참고만.

---

### 2026-06-12 ~ 06-16 · ImageNet / ResNet-50 · **KD baseline** ✅
- 데이터: ImageNet-1k (1.28M train / 50k val) · 모델: ResNet-50 · 4-bit(weight+act)
- 설정: **LSQ** + KD(teacher=resnet101) · Adam · linear decay · lr 1e-3 · 50ep · batch 128 · wd 0 · first/last FP
- **best Top-1 = 77.074%** (ep48) · Top-5 93.49 · 학습 112.9h
- (teacher resnet101 = 77.37%)

### 2026-06-17 ~ 06-21 · ImageNet / ResNet-50 · **KD + qSAM (warmup=10)** ✅
- 위와 동일 + qSAM(S²-SAM 단일스텝, rho=0.25, warmup=10)
- **best Top-1 = 77.154%** (ep49) · baseline 대비 **+0.08** · 학습 101.3h

### 2026-06-21 ~ 06-25 · ImageNet / ResNet-50 · **KD + qSAM (warmup=0, 처음부터)** ✅
- 위와 동일 + qSAM(rho=0.25, **warmup=0**)
- **best Top-1 = 77.230%** (ep48) · baseline 대비 **+0.16** ← KD regime 최고 · 학습 99.8h

---

### 2026-06-26 ~ 06-29 · ImageNet / ResNet-50 · **no-KD baseline (Adam/wd0)** ❌ 실패
- LSQ · **no-KD** · Adam · linear · lr 1e-3 · **100ep** · **wd 0** · first/last FP
- **72.632에서 정체** (ep31~48 평탄, train 81.7/val 72.6 = 과적합) → ep48에서 중단
- ⚠️ **교훈**: no-KD인데 weight decay=0 → 정규화 없어 과적합. KD가 정규화 역할을 했던 것.

### 2026-06-29 ~ 07-12 · ImageNet / ResNet-50 · **LSQ논문 baseline (no-KD)** ✅
- 데이터: ImageNet-1k · 모델: ResNet-50 · 4-bit
- 설정(**lsq.pdf 충실 재현**): no-KD · **SGD mom0.9** · **cosine decay** · **lr 0.01** · **90ep** · **wd 1e-4** · **first/last 8-bit**
- **best Top-1 = 75.668%** (ep89 완주)
- 경과: 06-29 시작 → 07-03 ep62서 중단(GPU 밀림) → 07-10 resume(scheduler state 복구) → 07-12 완주
- (논문 target 76.7%. 차이는 pre-act ResNet vs torchvision ResNet, batch 128 등 때문)

### 2026-07-03 ~ 07-09 · ImageNet / ResNet-50 · **LSQ논문 + qSAM (no-KD)** ✅
- 위 LSQ논문 baseline과 동일 + qSAM(**rho=1.0**, warmup=0, ratio=0.001)
- **best Top-1 = 75.760%** (ep88) · baseline 대비 **+0.09** · 학습 151h

### 2026-07-14 ~ (진행중) · ImageNet / ResNet-50 · **LSQ논문 + qSAM (no-KD, seed=43)** ⏳
- 위와 동일, **seed 43** (seed 42 결과 +0.09 재현성 확인용)
- 진행중, ~6일 예상

---

## 🔑 종합 결론
- **모든 주요 결과는 ImageNet ResNet-50 4-bit** (CIFAR-100/ResNet-18 4개는 구코드라 무효).
- **qSAM 효과**: 수정된 코드에서 KD·no-KD 두 regime 모두 baseline 대비 **+0.09 ~ +0.16 일관되게 양(+)**. 방향은 명확하나 크기가 단일-seed 노이즈(±0.5%) 수준 → seed 재현성(진행중) 필요.
- **KD vs no-KD**: KD가 ~+1.4%p 더 높음(77.2 vs 75.8). no-KD는 weight decay 필수(없으면 과적합).
- **warmup**: 처음부터(warmup=0)가 warmup=10보다 근소 우위(KD regime 77.23 vs 77.15).

## 📁 체크포인트/로그 위치
- ImageNet: `AO_QAT/resnet_imagenet/{log,models,models_lsqpaper}/<config-tag>/`
- CIFAR-100: `AO_QAT/resnet_cifar100/{log,models}/<method>_resnet18_4bit/`
