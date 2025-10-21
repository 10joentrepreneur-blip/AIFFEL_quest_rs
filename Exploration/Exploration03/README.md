# Playground_ResNet18


# 📊 CIFAR-10 Baseline vs ResNet18 Training Report

## 1️⃣ 실험 개요
본 실험은 PyTorch 기반 이미지 분류 학습에서 **Baseline(MLP)** 과 **ResNet18(CNN)** 을 CIFAR-10 데이터셋으로 비교하기 위한 것이다.
ResNet18은 skip connection을 사용하여 학습 안정성을 높이고, 데이터 증강 및 러닝레이트 스케줄러를 적용하였다.

---
## 2️⃣ 데이터셋
- **데이터셋:** CIFAR-10 (32×32 RGB, 클래스 10개)
- **Train/Test:** 50,000 / 10,000
- **전처리:** RandomCrop(32, padding=4), RandomHorizontalFlip(), RandAugment, RandomErasing
- **정규화:** CIFAR10 mean/std 적용

---
## 3️⃣ 모델 및 학습 설정
| 항목 | 설정값 |
|------|--------|
| 모델 | ResNet18 (CIFAR 버전) |
| Optimizer | SGD(lr=0.1, momentum=0.9, weight_decay=5e-4) |
| Loss | CrossEntropyLoss(label_smoothing=0.1) |
| Scheduler | CosineAnnealingLR |
| Epoch | 30 |
| Device | CUDA GPU |

---
## 4️⃣ 결과 요약
- 초기(1~5 epoch): 약 **50% 정확도**
- 30 epoch 기준 **77% 정확도**
- Validation Loss: **1.47 → 0.69**
- 안정적 수렴, 증강/스케줄러 효과 확인

![Training Curves](resnet18_training_curves.png)

---
## 5️⃣ 결론
- **Baseline:** 약 45~55% 수준에서 정체
- **ResNet18:** 77% 이상 도달
- **향후 개선:** Epoch 100+, Warmup, MixUp/CutMix 추가 시 85~90% 기대

---
**요약:**
CIFAR-10 환경에서 ResNet18은 Baseline 대비 30%p 이상 높은 성능을 보였으며,
적절한 증강과 스케줄링으로 더 높은 정확도를 달성할 수 있다.