# 부정 수강평 데이터를 불러와서, KoELECTRA 멀티라벨 분류하는 모델을 Fine-tuning

# 데이터 처리를 위한 라이브러리
import pandas as pd
import numpy as np
import random

# PyTorch 라이브러리
import torch
from torch.utils.data import Dataset, DataLoader

# Hugging Face Transformers에서 KoELECTRA 모델과 Tokenizer를 불러오기 위한 클래스
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 멀티라벨 데이터의 비율을 유지하며 Fold를 나누기 위한 클래스
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# 모델의 멀티라벨 분류 성능을 평가하기 위한 라이브러리
from sklearn.metrics import precision_score, recall_score, f1_score

# 1. KoELECTRA 사전학습 모델 설정
# 한국어 말뭉치로 사전학습된 KoELECTRA Base v3 모델 사용
MODEL_NAME = 'monologg/koelectra-base-v3-discriminator'

# 2. Random Seed 고정
# 동일한 조건에서 실험을 반복할 수 있도록 Random Seed 고정
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# GPU 환경에서도 동일한 Random Seed를 사용하겠다.
if torch.cuda.is_available() :

    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

# 3. 데이터 불러오기
# Study 1에서 부정으로 분류된 Gold Label 113건의 멀티라벨 데이터를 불러옴
review_df = pd.read_csv('../data/work24_negative_113_multilabel.csv')

# 4. 멀티라벨 유형 설정
# KDT 부정 수강평을 5개의 부정 유형으로 분류하겠다.
LABEL_COLUMNS = [
    'instructor',
    'curriculum',
    'operation',
    'facility',
    'employment'
]

# 5. 데이터셋 기본 정보 확인
print('데이터셋 크기')
print('전체 데이터 : ', len(review_df))

print('\n')
print('----- Label별 부정 유형 데이터 수 -----')

for label in LABEL_COLUMNS :

    print(label, ': ', int(review_df[label].sum()))

# 6. KoELECTRA Tokenizer 불러오기
# 수강평 텍스트를 KoELECTRA가 처리할 수 있는 토큰 ID 형태로 변환하기 위한 Tokenizer를 불러오기
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print('\n')
print('KoELECTRA Tokenizer 로딩 성공!')

# 7. KoELECTRA 멀티라벨 학습용 Dataset 클래스 생성
# 수강평 텍스트와 5개의 Label을 KoELECTRA 학습에 사용할 수 있는 형태로 변환하겠다.
class ReviewMultiLabelDataset(Dataset) :

    # Dataset 생성 시 수강평 텍스트, 정답 Label, Tokenizer를 저장하겠다.
    def __init__(self, dataframe, tokenizer, max_length=128) :

        self.texts = dataframe['review_text'].tolist()
        self.labels = dataframe[LABEL_COLUMNS].values
        self.tokenizer = tokenizer
        self.max_length = max_length

    # Dataset에 포함된 전체 데이터 개수를 반환하겠다.
    def __len__(self) :

        return len(self.texts)

    # 특정 위치의 수강평 1건을 KoELECTRA 입력 형태로 변환하여 반환하겠다.
    def __getitem__(self, idx) :

        text = self.texts[idx]
        labels = self.labels[idx]

        # 수강평 텍스트를 KoELECTRA 입력 형식으로 Tokenization 하겠다.
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Tokenization 결과와 5개의 정답 Label을 반환하겠다.
        return {
            'input_ids' : encoded['input_ids'].squeeze(0),
            'attention_mask' : encoded['attention_mask'].squeeze(0),
            'labels' : torch.tensor(labels, dtype=torch.float)
        }

# 8. GPU 사용 설정
# CUDA 사용이 가능하면 GPU를 사용하고, 사용할 수 없으면 CPU를 사용하겠다.
device = torch.device(
    'cuda'
    if torch.cuda.is_available()
    else 'cpu'
)

print('\n')
print('----- 학습 장치 -----')
print('사용 장치 : ', device)

if torch.cuda.is_available() :

    print('GPU : ', torch.cuda.get_device_name(0))

# 9. Fine-tuning 조건 설정
MAX_LENGTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
MAX_EPOCHS = 10

# Validation Macro-F1이 연속 2 Epoch 동안 개선되지 않으면 학습을 종료하겠다.
PATIENCE = 2

# Validation Macro-F1이 0.0005 이상 증가한 경우 성능이 개선된 것으로 판단하겠다.
MIN_DELTA = 0.0005

# Sigmoid 결과가 0.5 이상이면 해당 부정 유형을 1로 예측하겠다.
THRESHOLD = 0.5

print('\n')
print('----- Fine-tuning 조건 -----')
print('Max Length : ', MAX_LENGTH)
print('Batch Size : ', BATCH_SIZE)
print('Learning Rate : ', LEARNING_RATE)
print('Max Epoch : ', MAX_EPOCHS)
print('Early Stopping Patience : ', PATIENCE)
print('Min Delta : ', MIN_DELTA)
print('Threshold : ', THRESHOLD)

# 10. 5-Fold 데이터 분할
# 5개 부정 유형의 분포를 고려하여 데이터를 5개의 Fold로 분할하겠다.
kfold = MultilabelStratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED
)

# 멀티라벨 분할에 사용할 5개 Label을 가져오기
multilabel_values = review_df[LABEL_COLUMNS].values

# 11. 전체 Fold의 결과를 저장할 변수 생성
# 각 Fold의 성능을 저장하겠다.
fold_results = []

# 전체 113건의 Out-of-Fold 실제 Label을 저장하겠다.
oof_labels = np.zeros(
    (len(review_df), len(LABEL_COLUMNS)),
    dtype=int
)

# 전체 113건의 Out-of-Fold 예측 Label을 저장하겠다.
oof_predictions = np.zeros(
    (len(review_df), len(LABEL_COLUMNS)),
    dtype=int
)

# 전체 113건의 Out-of-Fold 예측 확률을 저장하겠다.
oof_probabilities = np.zeros(
    (len(review_df), len(LABEL_COLUMNS)),
    dtype=float
)

# 12. 5-Fold Cross Validation
# 전체 데이터를 5개의 Fold로 나누어 순서대로 Fine-tuning을 수행하겠다.
for fold, (train_index, validation_index) in enumerate(
    kfold.split(review_df, multilabel_values),
    start=1
) :

    print('\n')
    print('=' * 60)
    print('Fold : ', fold)
    print('=' * 60)

    # 13. 현재 Fold의 Train / Validation 데이터 생성
    train_df = review_df.iloc[train_index].reset_index(drop=True)
    validation_df = review_df.iloc[validation_index].reset_index(drop=True)

    print('Train : ', len(train_df))
    print('Validation : ', len(validation_df))

    print('\n')
    print('Train Label 분포')

    for label in LABEL_COLUMNS :

        print(label, ': ', int(train_df[label].sum()))

    print('\n')
    print('Validation Label 분포')

    for label in LABEL_COLUMNS :

        print(label, ': ', int(validation_df[label].sum()))

    # 14. Train / Validation Dataset 생성
    train_dataset = ReviewMultiLabelDataset(
        train_df,
        tokenizer,
        MAX_LENGTH
    )

    validation_dataset = ReviewMultiLabelDataset(
        validation_df,
        tokenizer,
        MAX_LENGTH
    )

    # 15. DataLoader 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # 16. KoELECTRA 멀티라벨 분류 모델 생성
    # 각 Fold마다 동일한 사전학습 KoELECTRA에서 새롭게 Fine-tuning을 시작하겠다.
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=5
    )

    model = model.to(device)

    # 17. Positive Weight 계산
    # 현재 Fold의 Train 데이터를 기준으로 각 부정 유형의 Positive Weight를 계산하겠다.
    positive_counts = train_df[LABEL_COLUMNS].sum().values
    negative_counts = len(train_df) - positive_counts

    positive_weights = torch.tensor(
        negative_counts / positive_counts,
        dtype=torch.float
    ).to(device)

    print('\n')
    print('----- Positive Weight -----')

    for index, label in enumerate(LABEL_COLUMNS) :

        print(
            label,
            ': ',
            round(positive_weights[index].item(), 4)
        )

    # 18. Loss Function 설정
    # 멀티라벨 분류와 클래스 불균형을 반영하기 위해 BCEWithLogitsLoss를 사용하겠다.
    loss_function = torch.nn.BCEWithLogitsLoss(
        pos_weight=positive_weights
    )

    # 19. Optimizer 설정
    # KoELECTRA Fine-tuning을 위해 AdamW Optimizer를 사용하겠다.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # 20. Early Stopping 변수 설정
    # 현재 Fold의 최고 Validation Macro-F1을 저장하겠다.
    best_macro_f1 = -1

    # 최고 Validation Macro-F1을 기록한 Epoch를 저장하겠다.
    best_epoch = 0

    # Early Stopping을 위한 Count를 저장하겠다.
    patience_count = 0

    # 최고 성능 모델의 가중치를 저장하겠다.
    best_model_state = None

    # 21. KoELECTRA Fine-tuning
    for epoch in range(MAX_EPOCHS) :

        model.train()

        total_loss = 0

        print('\n')
        print(
            '----- Fold : ', fold,
            ' / Epoch : ', epoch + 1, ' -----'
        )

        # Train DataLoader에서 Batch 단위로 데이터를 가져와 학습
        for batch_index, batch in enumerate(train_loader) :

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 이전 Batch에서 계산된 Gradient를 초기화하겠다.
            optimizer.zero_grad()

            # KoELECTRA에 Train 데이터를 입력하여 예측 결과를 구하겠다.
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            logits = outputs.logits

            # 예측 결과와 실제 멀티라벨을 비교하여 Loss를 계산하겠다.
            loss = loss_function(logits,labels)

            # Loss를 기준으로 역전파를 수행
            loss.backward()

            # 계산된 Gradient를 이용하여 모델의 가중치를 업데이트
            optimizer.step()
            total_loss += loss.item()

        # 한 Epoch의 평균 Loss를 계산
        average_loss = (
            total_loss
            / len(train_loader)
        )

        print(
            'Train Loss : ',
            round(average_loss, 4)
        )

        # 22. 매 Epoch Validation 성능 평가
        model.eval()

        validation_labels = []
        validation_predictions = []

        with torch.no_grad() :

            for batch in validation_loader :

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                logits = outputs.logits

                # 5개의 Logit에 Sigmoid를 적용하여 각각의 확률을 계산
                probabilities = torch.sigmoid(logits)

                # 확률이 0.5 이상인 부정 유형을 1로 예측하겠다.
                predictions = (
                    probabilities >= THRESHOLD
                ).int()

                validation_labels.extend(
                    labels.cpu().int().tolist()
                )

                validation_predictions.extend(
                    predictions.cpu().tolist()
                )

        # 23. Validation 평가 지표 계산
        macro_f1 = f1_score(
            validation_labels,
            validation_predictions,
            average='macro',
            zero_division=0
        )

        micro_f1 = f1_score(
            validation_labels,
            validation_predictions,
            average='micro',
            zero_division=0
        )

        print(
            'Validation Macro-F1 : ',
            round(macro_f1, 4)
        )

        print(
            'Validation Micro-F1 : ',
            round(micro_f1, 4)
        )

        # 24. Best Model 선정 및 Early Stopping
        # Validation Macro-F1이 기존 최고 성능보다 개선되었는지 확인하겠다.
        if macro_f1 > best_macro_f1 + MIN_DELTA :

            best_macro_f1 = macro_f1
            best_epoch = epoch + 1
            patience_count = 0

            # 현재 모델의 가중치를 최고 성능 모델로 저장하겠다.
            best_model_state = {
                key : value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            print('Best Model 갱신')

        else :

            patience_count += 1

            print(
                'Early Stopping Count : ',
                patience_count,
                '/',
                PATIENCE
            )

        # Validation Macro-F1이 연속 2 Epoch 동안 개선되지 않으면 학습을 종료하겠다.
        if patience_count >= PATIENCE :

            print('Early Stopping')

            break

    # 25. 현재 Fold의 Best Model 복원
    model.load_state_dict(best_model_state)
    model = model.to(device)
    model.eval()

    # 26. Best Model의 Validation 데이터 최종 예측
    fold_labels = []
    fold_predictions = []
    fold_probabilities = []

    with torch.no_grad() :

        for batch in validation_loader :

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids,attention_mask=attention_mask)

            logits = outputs.logits

            probabilities = torch.sigmoid(logits)

            predictions = (
                probabilities >= THRESHOLD
            ).int()

            fold_labels.extend(
                labels.cpu().int().tolist()
            )

            fold_predictions.extend(
                predictions.cpu().tolist()
            )

            fold_probabilities.extend(
                probabilities.cpu().tolist()
            )

    fold_labels = np.array(fold_labels)
    fold_predictions = np.array(fold_predictions)
    fold_probabilities = np.array(fold_probabilities)

    # 27. 현재 Fold의 최종 성능 계산
    fold_macro_precision = precision_score(
        fold_labels,
        fold_predictions,
        average='macro',
        zero_division=0
    )

    fold_macro_recall = recall_score(
        fold_labels,
        fold_predictions,
        average='macro',
        zero_division=0
    )

    fold_macro_f1 = f1_score(
        fold_labels,
        fold_predictions,
        average='macro',
        zero_division=0
    )

    fold_micro_f1 = f1_score(
        fold_labels,
        fold_predictions,
        average='micro',
        zero_division=0
    )

    print('\n')
    print('----- Fold 최종 결과 -----')
    print('Fold : ', fold) # 5
    print('Best Epoch : ', best_epoch) # 7
    print('Macro Precision : ', round(fold_macro_precision, 4)) # 0.2909
    print('Macro Recall : ', round(fold_macro_recall, 4)) # 0.2667
    print('Macro-F1 : ', round(fold_macro_f1, 4)) # 0.225
    print('Micro-F1 : ', round(fold_micro_f1, 4)) # 0.5098

    # 28. Fold별 결과 저장
    fold_results.append({
        'fold' : fold,
        'train_size' : len(train_df),
        'validation_size' : len(validation_df),
        'best_epoch' : best_epoch,
        'macro_precision' : fold_macro_precision,
        'macro_recall' : fold_macro_recall,
        'macro_f1' : fold_macro_f1,
        'micro_f1' : fold_micro_f1
    })

    # 29. OOF 예측 결과 저장
    # 현재 Validation 데이터는 해당 Fold에서 학습에 사용되지 않은 데이터이므로 OOF 결과로 저장하겠다.
    oof_labels[validation_index] = fold_labels
    oof_predictions[validation_index] = fold_predictions
    oof_probabilities[validation_index] = fold_probabilities

    # 현재 Fold의 GPU 메모리를 정리하겠다.
    del model
    del optimizer
    del loss_function

    if torch.cuda.is_available() :

        torch.cuda.empty_cache()

# 30. 5-Fold 학습 결과 확인
fold_results_df = pd.DataFrame(
    fold_results
)

print('\n')
print('=' * 60)
print('5-Fold Cross Validation 결과')
print('=' * 60)

print(
    fold_results_df.to_string(
        index=False
    )
)

# 31. 전체 OOF 성능 평가
# 전체 113건은 각각 한 번씩 Validation 데이터로 사용되었으므로 OOF 결과를 이용해 최종 성능을 평가하겠다.
oof_macro_precision = precision_score(
    oof_labels,
    oof_predictions,
    average='macro',
    zero_division=0
)

oof_macro_recall = recall_score(
    oof_labels,
    oof_predictions,
    average='macro',
    zero_division=0
)

oof_macro_f1 = f1_score(
    oof_labels,
    oof_predictions,
    average='macro',
    zero_division=0
)

oof_micro_f1 = f1_score(
    oof_labels,
    oof_predictions,
    average='micro',
    zero_division=0
)

print('\n')
print('=' * 60)
print('전체 OOF 최종 평가 결과')
print('=' * 60)

print(
    'Macro Precision : ',
    round(oof_macro_precision, 4)
)

print(
    'Macro Recall : ',
    round(oof_macro_recall, 4)
)

print(
    'Macro-F1 : ',
    round(oof_macro_f1, 4)
)

print(
    'Micro-F1 : ',
    round(oof_micro_f1, 4)
)

# 32. Label별 OOF 성능 평가
label_precision = precision_score(
    oof_labels,
    oof_predictions,
    average=None,
    zero_division=0
)

label_recall = recall_score(
    oof_labels,
    oof_predictions,
    average=None,
    zero_division=0
)

label_f1 = f1_score(
    oof_labels,
    oof_predictions,
    average=None,
    zero_division=0
)

print('\n')
print('----- Label별 OOF 평가 결과 -----')

for index, label in enumerate(LABEL_COLUMNS) :

    print(
        label,
        '| Precision : ',
        round(label_precision[index], 4),
        '| Recall : ',
        round(label_recall[index], 4),
        '| F1 : ',
        round(label_f1[index], 4)
    )

# 33. Fold별 결과 CSV 저장
fold_results_df.to_csv(
    'multilabel_fold_results.csv',
    index=False,
    encoding='utf-8-sig'
)

# 34. OOF 예측 결과 CSV 생성
oof_result_df = review_df.copy()

for index, label in enumerate(LABEL_COLUMNS) :

    # 각 Label의 OOF 예측 확률을 저장하겠다.
    oof_result_df[
        label + '_probability'
    ] = oof_probabilities[:, index]

    # 각 Label의 OOF 예측 결과를 저장하겠다.
    oof_result_df[
        label + '_prediction'
    ] = oof_predictions[:, index]

oof_result_df.to_csv(
    'multilabel_oof_predictions.csv',
    index=False,
    encoding='utf-8-sig'
)

print('\n')
print('결과 파일 저장 완료!')
print('multilabel_fold_results.csv')
print('multilabel_oof_predictions.csv')