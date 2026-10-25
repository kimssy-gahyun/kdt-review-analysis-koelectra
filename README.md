# KDT Review Analysis with KoELECTRA

고용24 NCS/KDT 수강평 데이터를 활용하여 **만족도와 리뷰 텍스트 간의 불일치를 탐지**하고,  
부정 리뷰의 주요 원인을 **Multi-label Classification**으로 분석한 프로젝트입니다.

석사학위논문 「배달 앱 리뷰 신뢰도 향상을 위한 별점–텍스트 불일치 탐지 및 부정 유형 분류」의 연구 방법을 참고하여, 분석 도메인을 배달 앱 리뷰에서 **NCS/KDT 교육과정 수강평**으로 확장하였습니다.

---

## 1. Project Overview

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

Study 1에서 구축한 부정 리뷰를 대상으로 부정 경험을 다음 5개 유형으로 분류합니다.

- 강사/강의
- 교육과정/커리큘럼
- 운영/행정
- 시설/환경
- 취업지원

KoELECTRA 기반 Multi-label Classification을 적용하고, 5-Fold Cross Validation과 OOF Prediction으로 성능을 평가했습니다.

---

## 2. Dataset

고용24에서 서울 소재 NCS/KDT 교육과정 수강평을 수집하여 분석에 활용했습니다.

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

## 3. Model

사용한 사전학습 모델은 다음과 같습니다.

```text
monologg/koelectra-base-v3-discriminator
```

### Study 1

- Binary Text Classification
- Weighted Cross Entropy Loss
- AdamW Optimizer
- Early Stopping
- Validation Macro-F1 기반 최적 모델 선정

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
- 5 Output Logits
- Sigmoid
- BCEWithLogitsLoss
- Positive Class Weight
- 5-Fold Cross Validation
- OOF Prediction

---

## 4. Requirements

### Python

본 프로젝트는 다음 환경에서 구현 및 실행했습니다.

```text
Python 3.12.10
```

### Required Libraries

실행 전 다음 라이브러리를 설치해야 합니다.

```bash
pip install torch transformers accelerate pandas numpy scikit-learn matplotlib iterative-stratification
```

주요 라이브러리:

- PyTorch
- Transformers
- Accelerate
- pandas
- NumPy
- scikit-learn
- matplotlib
- iterative-stratification

### PyTorch / CUDA

본 실험에서는 다음 PyTorch 환경을 사용했습니다.

```text
PyTorch 2.11.0 + CUDA 12.8
```

NVIDIA GPU 및 CUDA 12.8 환경에서 동일한 PyTorch 버전을 구성하는 경우 다음 명령어를 사용할 수 있습니다.

```bash
pip install --force-reinstall torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

※ GPU 및 CUDA 환경은 실험 수행에 사용한 환경이며, 시스템 환경에 따라 PyTorch 설치 방법이 달라질 수 있습니다.

---

## 5. Experimental Environment

본 프로젝트의 실제 실험 수행 환경은 다음과 같습니다.

| Item | Environment |
|---|---|
| OS | Windows |
| Python | 3.12.10 |
| PyTorch | 2.11.0 + cu128 |
| CUDA | 12.8 |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| Pre-trained Model | monologg/koelectra-base-v3-discriminator |

모델 Fine-tuning 과정에서 GPU를 사용했습니다.

---

## 6. Project Structure

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
│   ├── fine_tuning_results.csv
│   └── work24_reviews_with_prediction.csv
│
├── study2/
│   ├── train_koelectra_multilabel.py
│   ├── train_koelectra_multilabel_experiment.py
│   ├── multilabel_fold_results.csv
│   └── multilabel_oof_predictions.csv
│
├── test_koelectra.py
└── README.md
```

---

## 7. Execution

### 7.1 KoELECTRA 실행 환경 확인

프로젝트 루트에서 다음 파일을 실행하여 KoELECTRA 모델 로딩 및 실행 환경을 확인할 수 있습니다.

```bash
python test_koelectra.py
```

최초 실행 시 Hugging Face에서 다음 사전학습 모델을 다운로드하므로 인터넷 연결이 필요합니다.

```text
monologg/koelectra-base-v3-discriminator
```

### 7.2 Study 1 실행

`study1` 디렉터리로 이동합니다.

```bash
cd study1
```

기본 KoELECTRA Fine-tuning 및 평가:

```bash
python train_koelectra.py
```

Learning Rate 비교 및 최적 모델 선정 실험:

```bash
python train_koelectra_experiment.py
```

각 파일의 주요 목적은 다음과 같습니다.

- `train_koelectra.py` : KoELECTRA Fine-tuning 및 기본 학습·평가
- `train_koelectra_experiment.py` : Learning Rate별 학습, Validation 평가, Early Stopping 및 최적 모델 선정

### 7.3 Study 2 실행

프로젝트 루트의 `study2` 디렉터리로 이동합니다.

```bash
cd study2
```

기본 Multi-label Classification:

```bash
python train_koelectra_multilabel.py
```

5-Fold Cross Validation 실험:

```bash
python train_koelectra_multilabel_experiment.py
```

각 파일의 주요 목적은 다음과 같습니다.

- `train_koelectra_multilabel.py` : KoELECTRA 기반 부정 유형 Multi-label Classification
- `train_koelectra_multilabel_experiment.py` : 5-Fold Cross Validation 및 OOF Prediction 기반 성능 평가

※ 각 Python 파일은 `data/` 디렉터리의 데이터 파일을 사용합니다.  
※ 모델 학습에는 시간이 소요될 수 있으며 GPU 환경 사용을 권장합니다.

---

## 8. Results

### Study 1 — Sentiment Classification

최종 Test Dataset 평가 결과입니다.

| Metric | Score |
|---|---:|
| Accuracy | 0.9524 |
| Precision | 0.8750 |
| Recall | 0.8235 |
| F1 | 0.8485 |
| Macro-F1 | **0.9101** |

전체 분석 데이터 2,450건에 대한 추론 결과:

- Text Non-negative : 2,195건 (89.59%)
- Text Negative : 255건 (10.41%)

만족도 기준 긍정 리뷰 2,220건 중 텍스트에서는 부정으로 예측된 **Hidden Negative 150건(6.76%)**을 확인했습니다.

### Study 2 — Multi-label Classification

전체 OOF 성능:

| Metric | Score |
|---|---:|
| Macro Precision | 0.2956 |
| Macro Recall | 0.4081 |
| Macro-F1 | **0.3404** |
| Micro-F1 | **0.5064** |

유형별 OOF F1:

| Category | F1 |
|---|---:|
| 교육과정/커리큘럼 | **0.6179** |
| 강사/강의 | 0.5625 |
| 운영/행정 | 0.4262 |
| 취업지원 | 0.0952 |
| 시설/환경 | 0.0000 |

데이터가 상대적으로 많은 교육과정/커리큘럼, 강사/강의 유형에서는 비교적 높은 성능을 보였으나, 학습 사례가 적은 시설/환경과 취업지원 유형에서는 성능 확보에 한계가 있었습니다.

---

## 9. Reproducibility

실험의 Random Seed는 다음과 같이 고정했습니다.

```text
Random Seed : 42
```

동일한 데이터 분할 및 실험 조건을 사용하도록 구성했으나, GPU, CUDA, PyTorch 및 기타 실행 환경 차이에 따라 학습 결과에는 일부 차이가 발생할 수 있습니다.

제출 용량 제한으로 학습 완료된 모델 Weight는 포함하지 않았으며, 실험 결과 CSV를 함께 제공합니다.

---

## 10. Limitations

- Gold Label 700건을 단일 연구자가 라벨링하여 라벨링 주관성이 존재할 수 있습니다.
- Study 2의 부정 리뷰가 113건으로 적고 유형별 데이터 불균형이 존재합니다.
- 고용24 수강평에서 개별 리뷰의 작성 시점 및 과정 회차를 확인할 수 없어 시기별 분석에는 한계가 있습니다.
- Study 2의 초기 멀티라벨 구축 과정에서 생성형 AI를 활용한 후 전체 데이터를 직접 검토 및 수정했습니다.

---

## 11. Conclusion

본 프로젝트에서는 KoELECTRA 기반 텍스트 분류를 통해 **정량적 만족도만으로는 확인하기 어려운 부정 경험을 탐지**하고, 부정 리뷰를 교육과정의 주요 개선 영역으로 세분화했습니다.

이를 통해 정량적 만족도와 리뷰 텍스트 분석을 결합하여 NCS/KDT 수강평을 보다 입체적으로 분석할 수 있음을 확인했습니다.