# 부정 수강평 데이터를 불러와서, KoELECTRA 멀티라벨 분류가 정상적으로 동작하는지 확인

# 데이터 처리를 위한 라이브러리
import pandas as pd
import numpy as np
import random

# PyTorch 라이브러리
import torch

# Hugging Face Transformers에서 KoELECTRA의 토크나이저를 불러오기 위한 클래스
from transformers import AutoTokenizer

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
print('전체 데이터 : ', len(review_df))  # 113

print('\n')
print('데이터 컬럼')
print(review_df.columns.tolist())

print('\n')
print('----- Label별 부정 유형 데이터 수 -----')

for label in LABEL_COLUMNS :
    print(label, ': ', int(review_df[label].sum()))

# 6. KoELECTRA Tokenizer 불러오기
# 수강평 텍스트를 KoELECTRA가 처리할 수 있는 토큰 ID 형태로 변환하기 위한 Tokenizer를 불러오기
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print('\n')
print('KoELECTRA Tokenizer 로딩 성공!')

# 7. 실제 수강평 1건 확인
# 첫 번째 수강평 텍스트와 5개의 정답 Label을 가져와서 출력
sample_text = review_df.iloc[0]['review_text']
sample_labels = review_df.iloc[0][LABEL_COLUMNS].tolist()

print('\n')
print('----- 원본 수강평 -----')
print(sample_text)

print('\n')
print('정답 Label : ', sample_labels)

# 8. 수강평 Tokenization
# 실제 수강평을 KoELECTRA 입력 형식으로 변환하기
encoded = tokenizer(
    sample_text,
    truncation=True,  # 최대 길이를 초과하는 텍스트는 잘라내겠다.
    padding='max_length',  # 모든 입력 길이를 max_length에 맞추겠다.
    max_length=128,  # 최대 토큰 길이를 128로 설정하겠다.
    return_tensors='pt'  # 결과를 PyTorch Tensor 형태로 반환하겠다.
)

# 9. Tokenization 결과 확인
print('\n')
print('----- Tokenization 결과 -----')

# 입력 Tensor의 크기를 확인
print('input_ids shape : ', encoded['input_ids'].shape)  
# torch.Size([1, 128])

# 사람이 확인할 수 있도록 실제 토큰 형태도 출력
tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])

print('\n')
print('변환된 Token : ')
print(tokens)

# ---

# PyTorch에서 Dataset을 만들기 위한 클래스
from torch.utils.data import Dataset

# 10. KoELECTRA 멀티라벨 학습용 Dataset 클래스 생성
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

# ---

# 멀티라벨 데이터의 비율을 유지하며 Fold를 나누기 위한 클래스
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# 11. 5-Fold 데이터 분할
# 5개 부정 유형의 분포를 고려하여 데이터를 5개의 Fold로 분할하겠다.
kfold = MultilabelStratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED
)

# 멀티라벨 분할에 사용할 5개 Label을 가져오기
multilabel_values = review_df[LABEL_COLUMNS].values

# 먼저 Fold 1만 사용하여 1 Epoch 학습이 정상적으로 동작하는지 확인하겠다.
train_index, validation_index = next(
    kfold.split(
        review_df,
        multilabel_values
    )
)

train_df = review_df.iloc[train_index].reset_index(drop=True)
validation_df = review_df.iloc[validation_index].reset_index(drop=True)

# 12. Fold 1 데이터 확인
print('\n')
print('----- Fold 1 데이터 분할 결과 -----')
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

# 13. Train / Validation Dataset 생성
train_dataset = ReviewMultiLabelDataset(
    train_df,
    tokenizer
)

validation_dataset = ReviewMultiLabelDataset(
    validation_df,
    tokenizer
)

print('\n')
print('----- Dataset 생성 결과 -----')
print('Train Dataset : ', len(train_dataset))
print('Validation Dataset : ', len(validation_dataset))

# ---

# PyTorch에서 Data를 불러올 때 사용하는 클래스
from torch.utils.data import DataLoader

# 14. DataLoader 생성
# Dataset의 데이터를 Batch 단위로 나누어 모델 학습에 사용할 수 있도록 구성
BATCH_SIZE = 16

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

print('\n')
print('----- DataLoader 생성 결과 -----')
print('Train Batch : ', len(train_loader))
print('Validation Batch : ', len(validation_loader))

# ---

# Hugging Face Transformers에서 KoELECTRA 분류 모델을 불러오기 위한 클래스
from transformers import AutoModelForSequenceClassification

# 15. KoELECTRA 멀티라벨 분류 모델 생성
# 사전학습된 KoELECTRA에 5개의 부정 유형을 분류하기 위한 분류층을 추가
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=5
)

# 16. GPU 사용 설정
# CUDA 사용이 가능하면 GPU를 사용하고, 사용할 수 없으면 CPU를 사용하겠다.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# KoELECTRA 모델을 설정한 장치로 이동
model = model.to(device)

print('\n')
print('----- KoELECTRA 모델 생성 결과 -----')
print('사용 장치 : ', device)

if torch.cuda.is_available() :
    print('GPU : ', torch.cuda.get_device_name(0))

# ---

# 17. Positive Weight 계산
# Train 데이터의 Label 개수를 기준으로 각 부정 유형의 Positive Weight를 계산하겠다.
positive_counts = train_df[LABEL_COLUMNS].sum().values
negative_counts = len(train_df) - positive_counts

# 데이터가 적은 부정 유형의 Loss가 더 크게 반영되도록 Positive Weight를 계산하겠다.
positive_weights = torch.tensor(
    negative_counts / positive_counts,
    dtype=torch.float
)

# 계산한 Positive Weight를 모델과 동일한 장치로 이동
positive_weights = positive_weights.to(device)

print('\n')
print('----- Positive Weight 계산 결과 -----')

for index, label in enumerate(LABEL_COLUMNS) :
    print(
        label,
        ': ',
        round(positive_weights[index].item(), 4)
    )
# instructor :  1.5714
# curriculum :  1.25
# operation :  3.5
# facility :  21.5
# employment :  17.0

# 18. Loss Function 설정
# 멀티라벨 분류와 클래스 불균형을 반영하기 위해 BCEWithLogitsLoss를 사용하겠다.
loss_function = torch.nn.BCEWithLogitsLoss(
    pos_weight=positive_weights
)

print('\n')
print('Loss Function : ', loss_function)

# ---

# 19. Optimizer 설정
# KoELECTRA Fine-tuning을 위해 AdamW Optimizer를 사용
LEARNING_RATE = 5e-5

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

print('Learning Rate : ', LEARNING_RATE)
print('Optimizer : ', optimizer.__class__.__name__)

# ---

# 20. KoELECTRA Fine-tuning
# 먼저 Fold 1에서 1 Epoch 학습을 진행하여 학습 과정이 정상적으로 동작하는지 확인하겠다.
EPOCHS = 1

for epoch in range(EPOCHS) :

    # 모델을 학습 모드로 설정하겠다.
    model.train()

    # Epoch 전체 Loss를 저장하기 위한 변수를 초기화
    total_loss = 0

    print('\n')
    print('----- Epoch : ', epoch + 1, '-----')

    # Train DataLoader에서 Batch 단위로 데이터를 가져와 학습하겠다.
    for batch_index, batch in enumerate(train_loader) :

        # Batch 데이터를 GPU로 이동시키겠다.
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # 이전 Batch에서 계산된 Gradient를 초기화하겠다.
        optimizer.zero_grad()

        # KoELECTRA에 수강평 데이터를 입력하여 예측 결과를 구하겠다.
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # 모델이 출력한 5개 부정 유형의 Logits를 가져오기
        logits = outputs.logits

        # 예측 결과와 실제 멀티라벨을 비교하여 Loss를 계산하겠다.
        loss = loss_function(logits, labels)

        # Loss를 기준으로 역전파를 수행하여 Gradient를 계산하겠다.
        loss.backward()

        # 계산된 Gradient를 이용하여 KoELECTRA의 가중치를 업데이트하겠다.
        optimizer.step()

        # 현재 Batch의 Loss를 Epoch 전체 Loss에 누적
        total_loss += loss.item()

        # 2 Batch마다 현재 학습 진행 상황과 Loss를 출력
        if (batch_index + 1) % 2 == 0 :

            print(
                'Batch : ',
                batch_index + 1,
                '/',
                len(train_loader),
                ' | Loss : ',
                round(loss.item(), 4)
            )

    # 한 Epoch의 평균 Loss를 계산하겠다.
    average_loss = (
        total_loss
        / len(train_loader)
    )

    print('\n')
    print('Epoch 평균 Loss : ', round(average_loss, 4))

# ---

# 모델의 멀티라벨 분류 성능을 평가하기 위한 라이브러리
from sklearn.metrics import precision_score, recall_score, f1_score

# 21. Validation 데이터 성능 평가
# 학습된 KoELECTRA 모델을 Validation 데이터로 평가
# 모델을 평가 모드로 설정하겠다.
model.eval()

# 실제 Label과 모델의 예측 Label을 저장하기 위한 List를 생성
validation_labels = []
validation_predictions = []

# 평가 과정에서는 Gradient를 계산하지 않음
with torch.no_grad() :

    # Validation DataLoader에서 Batch 단위로 데이터를 가져오겠다.
    for batch in validation_loader :

        # Batch 데이터를 GPU로 이동
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # KoELECTRA에 Validation 데이터를 입력하여 예측 결과 구하기
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # 모델이 출력한 Logits를 가져오기
        logits = outputs.logits

        # 각 부정 유형의 Logits를 0~1 사이의 확률로 변환하겠다.
        probabilities = torch.sigmoid(logits)

        # 확률이 0.5 이상인 부정 유형을 1로 예측하겠다.
        predictions = (
            probabilities >= 0.5
        ).int()

        # 성능 평가를 위해 실제 Label과 예측 Label을 CPU로 이동하여 List에 저장
        validation_labels.extend(
            labels.cpu().int().tolist()
        )

        validation_predictions.extend(
            predictions.cpu().tolist()
        )

# 22. Validation 평가 지표 계산
# 5개 부정 유형의 F1 Score를 동일한 비중으로 평가하겠다.
macro_f1 = f1_score(
    validation_labels,
    validation_predictions,
    average='macro',
    zero_division=0
)

# 전체 Label의 예측 결과를 하나로 합쳐서 F1 Score를 계산하겠다.
micro_f1 = f1_score(
    validation_labels,
    validation_predictions,
    average='micro',
    zero_division=0
)

macro_precision = precision_score(
    validation_labels,
    validation_predictions,
    average='macro',
    zero_division=0
)

macro_recall = recall_score(
    validation_labels,
    validation_predictions,
    average='macro',
    zero_division=0
)

# 23. Validation 평가 결과 출력
print('\n')
print('----- Validation 평가 결과 -----')
print('Macro Precision : ', round(macro_precision, 4)) # 0.2693 
print('Macro Recall : ', round(macro_recall, 4)) # 0.5467
print('Macro-F1 : ', round(macro_f1, 4)) # 0.3534
print('Micro-F1 : ', round(micro_f1, 4)) # 0.5412

# 24. Label별 Validation F1 Score 확인
# 각 부정 유형별 F1 Score를 계산하겠다.
label_f1_scores = f1_score(
    validation_labels,
    validation_predictions,
    average=None,
    zero_division=0
)

print('\n')
print('----- Label별 F1 Score -----')

for index, label in enumerate(LABEL_COLUMNS) :
    print(
        label,
        ': ',
        round(label_f1_scores[index], 4)
    )
# instructor :  0.5625
# curriculum :  0.75
# operation :  0.4545
# facility :  0.0
# employment :  0.0