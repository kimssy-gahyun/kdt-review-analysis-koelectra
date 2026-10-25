# KDT Review Analysis with KoELECTRA

고용24 NCS/KDT 수강평 데이터를 활용하여 **만족도와 리뷰 텍스트 간의 불일치를 탐지**하고,  
부정 리뷰의 주요 원인을 **Multi-label Classification**으로 분석한 프로젝트입니다.

석사학위논문 「배달 앱 리뷰 신뢰도 향상을 위한 별점–텍스트 불일치 탐지 및 부정 유형 분류」의 연구 방법을 참고하여, 분석 도메인을 배달 앱 리뷰에서 **NCS/KDT 교육과정 수강평**으로 확장하였습니다.

---

## Project Overview

본 프로젝트는 두 단계로 구성됩니다.

### Study 1. 만족도–텍스트 감성 불일치 탐지

수강생이 입력한 정량적 만족도와 실제 리뷰 텍스트의 감성이 일치하는지 분석합니다.

```text
Work24 Review Data
        ↓
Data Preprocessing
        ↓
Gold Label Dataset
        ↓
KoELECTRA Fine-tuning
        ↓
Sentiment Prediction
        ↓
Satisfaction–Text Comparison
        ↓
Hidden Negative Detection
```

### Study 2. 부정 리뷰 유형 분류

Study 1에서 구축한 부정 리뷰를 대상으로 교육과정에서 발생하는 부정 경험을 5개 유형으로 분류합니다.

- 강사/강의
- 교육과정/커리큘럼
- 운영/행정
- 시설/환경
- 취업지원

KoELECTRA 기반 **Multi-label Classification**을 적용하고, 5-Fold Cross Validation과 OOF Prediction으로 성능을 평가했습니다.

---

## Dataset

고용24에서 서울 소재 NCS/KDT 교육과정 수강평을 수집했습니다.

| Dataset | Count |
|---|---:|
| Raw Reviews | 3,852 |
| Analysis Dataset | 2,450 |
| Gold Label | 700 |
| Study 2 Negative Reviews | 113 |

Study 1의 Gold Label 데이터는 다음과 같이 분할했습니다.

| Split | Count |
|---|---:|
| Train | 490 |
| Validation | 105 |
| Test | 105 |

---

## Model

Pre-trained model:

```text
monologg/koelectra-base-v3-discriminator
```

### Study 1

- Binary Text Classification
- Weighted Cross Entropy Loss
- AdamW Optimizer
- Early Stopping
- Validation Macro-F1 기반 모델 선정

Learning Rate 후보:

```text
2e-5 / 3e-5 / 5e-5
```

최종 선택:

```text
Learning Rate : 5e-5
Best Epoch    : 2
```

### Study 2

- Multi-label Text Classification
- 5 output logits
- Sigmoid
- BCEWithLogitsLoss
- Positive Class Weight
- 5-Fold Cross Validation
- OOF Prediction

---

## Results

### Study 1 — Sentiment Classification

| Metric | Score |
|---|---:|
| Accuracy | 0.9524 |
| Precision | 0.8750 |
| Recall | 0.8235 |
| F1 | 0.8485 |
| Macro-F1 | **0.9101** |

전체 분석 데이터 2,450건에 대해 추론한 결과:

- Text Non-negative: 2,195건 (89.59%)
- Text Negative: 255건 (10.41%)

만족도 기준 긍정 리뷰 2,220건 중 텍스트에서는 부정으로 예측된 **Hidden Negative가 150건(6.76%)** 발견되었습니다.

### Study 2 — Multi-label Classification

Overall OOF Performance:

| Metric | Score |
|---|---:|
| Macro Precision | 0.2956 |
| Macro Recall | 0.4081 |
| Macro-F1 | **0.3404** |
| Micro-F1 | **0.5064** |

Per-label F1:

| Category | F1 |
|---|---:|
| 교육과정/커리큘럼 | **0.6179** |
| 강사/강의 | 0.5625 |
| 운영/행정 | 0.4262 |
| 취업지원 | 0.0952 |
| 시설/환경 | 0.0000 |

데이터가 상대적으로 많은 교육과정/커리큘럼, 강사/강의 유형에서는 비교적 높은 성능을 보였으나, 학습 사례가 적은 시설/환경과 취업지원 유형에서는 성능 확보에 한계가 있었습니다.

---

## Project Structure

```text
kdt-review-analysis-koelectra/
│
├── data/
│   ├── train.csv
│   ├── validation.csv
│   ├── test.csv
│   ├── work24_reviews_analysis.csv
│   └── work24_negative_113_multilabel.csv
│
├── study1/
│   ├── train_koelectra.py
│   ├── train_koelectra_experiment.py
│   ├── inference_run.py
│   ├── fine_tuning_results.csv
│   └── work24_reviews_with_prediction.csv
│
├── study2/
│   ├── train_koelectra_multilabel.py
│   ├── train_koelectra_multilabel_experiment.py
│   ├── multilabel_fold_results.csv
│   └── multilabel_oof_predictions.csv
│
├── models/
├── test_koelectra.py
└── README.md
```

---

## Environment

- Python 3.12.10
- PyTorch 2.11.0 + CUDA 12.8
- Transformers
- scikit-learn
- pandas
- NumPy
- matplotlib
- iterative-stratification

GPU:

```text
NVIDIA GeForce RTX 5060 Laptop GPU
```

---

## Limitations

- Gold Label 700건을 단일 연구자가 라벨링하여 라벨링 주관성이 존재할 수 있습니다.
- Study 2의 부정 리뷰가 113건으로 적고 유형별 데이터 불균형이 존재합니다.
- 고용24 수강평에서 개별 리뷰의 작성 시점 및 과정 회차를 확인할 수 없어 시기별 분석에는 한계가 있습니다.
- Study 2의 일부 초기 멀티라벨 구축 과정에서 생성형 AI를 활용한 후 직접 검토 및 수정했습니다.

---

## Conclusion

본 프로젝트에서는 KoELECTRA 기반 텍스트 분류를 통해 **정량적 만족도만으로는 확인하기 어려운 부정 경험을 탐지**하고, 부정 리뷰를 교육과정의 주요 개선 영역으로 세분화했습니다.

이를 통해 정량적 만족도와 리뷰 텍스트 분석을 결합하여 NCS/KDT 수강평을 보다 입체적으로 분석할 수 있음을 확인했습니다.