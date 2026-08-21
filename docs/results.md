# ImageNet 4-bit ResNet-50 — LSQ Baseline vs qSAM 결과

## 요약 (ImageNet val, Top-1 / Top-5)

| Run | Quantizer | qSAM | Best Top-1 | Best Top-5 | 상태 |
|---|---|---|---|---|---|
| **Baseline** | LSQ | off | **77.074 %** | 93.488 % | 완료 |
| **qSAM (warmup=10)** | LSQ | on (ep10~) | **77.154 %** | 93.422 % | 완료 |
| qSAM (warmup=0) | LSQ | on (ep0~) | — | — | 진행 중 |
| _참고: Teacher (resnet101, FP)_ | — | — | 77.372 % | 93.560 % | — |

> **qSAM(w10) 77.154 % vs Baseline 77.074 % → +0.08 %p.** 학습 중에는 qSAM이 더 크게 앞섰으나(아래 추세) 수렴 구간에서 격차가 좁혀짐. **단일 seed 기준 +0.08은 노이즈 바닥(±0.5%) 안**이라, "통계적으로 우세"라고 단정할 수는 없음(아래 해석 참조).

---

## 공통 설정 (Baseline = qSAM, 동일 하네스)

| 항목 | 값 |
|---|---|
| Student / Bit-width | ResNet-50 / 4-bit (weight & activation) |
| Quantizer | **LSQ** (learned step size + gradient-scale `g=1/√(N·Qp)`) |
| Knowledge Distillation | on, teacher = full-precision **ResNet-101** (frozen) |
| Epochs / Batch | 50 / 128 |
| Optimizer / LR | Adam (0.9, 0.999), lr 1e-3, linear decay → 0 @ep50 |
| Weight decay / Seed | 0.0 / 42 |
| First conv / FC | full-precision (LSQ-style, 양자화·qSAM 제외) |
| GPU | 1× RTX A6000 (48 GB) |
| Data | ImageNet-1k (1,281,167 train / 50,000 val) |

**유일한 차이**: `use_qsam` (baseline=False, qSAM=True). qSAM 세부: single-step **S²-SAM**(이전 step gradient로 perturbation, zero extra cost) + **RA-qSAM p=2** lattice (`δ = ρ·Δ·sign(g_prev)`, top-K, ratio=0.001, rho=0.25), warmup 후 활성 + 매 validate 전 **BN recalibration**.

---

## Baseline (LSQ) 상세

| | Top-1 |
|---|---|
| Best (epoch 48) | **77.074 %** (Top-5 93.488) |
| Final (epoch 49) | 77.016 % (Top-5 93.472) |
| Teacher와 격차 | −0.30 %p |
| 총 학습 시간 | 112.91 h |

수렴: ep0 56.4 → ep10 72.9 → ep20 74.6 → ep30 75.7 → ep40 76.7 → **ep48 77.07**(막판 LR 감쇠 상승).

---

## qSAM (warmup=10) 상세 + Baseline 대비

| | Top-1 |
|---|---|
| Best (epoch 49) | **77.154 %** (Top-5 93.422) |
| Baseline 대비 (best) | **+0.080 %p** |
| 총 학습 시간 | 101.30 h (※ wall-clock 차이는 공유 GPU 혼잡 탓, 연산량은 baseline과 동일) |

### qSAM − Baseline 격차 추세 (epoch별 Top-1 차, qSAM ON 구간)
| 구간 | 평균 diff |
|---|---|
| ep 10–19 | **+0.372** |
| ep 20–29 | +0.239 |
| ep 30–39 | +0.270 |
| ep 40–49 | +0.111 |

- **qSAM 활성 직후(ep10)부터 40 epoch 내내 일관되게 baseline을 앞섬**(매 epoch diff 양수).
- warmup 구간(ep0–9, 둘 다 순수 LSQ)에서는 qSAM run이 오히려 평균 −0.32 뒤처져 있었음(run간 비결정성 오프셋) → qSAM이 켜지며 부호가 +로 뒤집힘.
- 다만 수렴할수록 격차 축소(+0.37 → +0.11), 최종 +0.08.

---

## 해석 (정직한 평가)

- **긍정 신호**: qSAM 활성 시점과 정확히 맞물려 ~40 epoch 연속 baseline 우위 + warmup 오프셋(−0.32)을 뒤집은 점은 단순 노이즈로 보기 어려움. 학습 동역학상 qSAM이 더 나은 방향을 잡았다는 정황.
- **한계**: 최종 격차 +0.08 %p는 우리가 측정한 **단일 seed 노이즈 바닥(±0.5%) 이내**. 따라서 ImageNet 단일 run 한 쌍만으로 "qSAM이 더 좋다"를 통계적으로 확정할 수 없음.
- **연구적으로 필요한 것**: multi-seed 또는 더 빠른 세팅(CIFAR-100)에서의 ablation(warmup·ratio·rho·p=2/∞)으로 유의성 확인. ImageNet은 run당 ~4–5일이라 multi-seed가 비현실적.
- **warmup=0 (원조 SAM처럼 처음부터 qSAM) run 진행 중** → warmup이 실제로 도움 되는지 비교 예정.

---

## 재현 (Reproduction)

```bash
cd AO_QAT/resnet_imagenet
# Baseline (LSQ)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  USE_KD=True USE_QSAM=False bash run.sh
# qSAM (warmup=10)
USE_KD=True USE_QSAM=True QSAM_WARMUP=10 bash run.sh      # + 위 CUDA_* env
# qSAM (warmup=0, 처음부터)
USE_KD=True USE_QSAM=True QSAM_WARMUP=0  bash run.sh
```

로그·체크포인트 (warmup이 경로에 포함됨):
- Baseline: `log|models/resnet50_4bit_qsam_False_kd_True/`
- qSAM w10: `log|models/resnet50_4bit_qsam_True_kd_True_w10/`
- qSAM w0 : `log|models/resnet50_4bit_qsam_True_kd_True_w0/`

---

## 비고

- batch 256→**128** 조정(QAT 활성화 텐서 오버헤드로 256은 48GB OOM). 세 run 모두 동일 적용.
- 공유 서버 GPU 혼잡으로 epoch당 wall-clock 2.0~4.5h 변동(연산 자체 ~2.2h/epoch). run간 총시간 차이는 방법이 아니라 혼잡도 차이.
- 학습 기간: Baseline 2026-06-12~16, qSAM(w10) 2026-06-17~21.
