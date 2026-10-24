# KoELECTRA를 활용하여 수강평 텍스트의 부정 여부를 분류하는 모델을 Fine-tuning

# 데이터 처리를 위한 라이브러리
import pandas as pd
import numpy as np
import random
import os

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
# 직접 구축한 Gold Label 데이터를 Train / Validation / Test 데이터로 각각 불러옴
train_df = pd.read_csv('../data/train.csv')
validation_df = pd.read_csv('../data/validation.csv')
test_df = pd.read_csv('../data/test.csv')

# 4. 데이터셋 기본 정보 확인
print('데이터셋 크기')
print('Train : ', len(train_df)) # 490
print('Validation : ', len(validation_df)) # 105
print('Test : ', len(test_df)) # 105

print('\n')
print('Train 데이터 컬럼')
print(train_df.columns.tolist()) 
# ['sample_id', 'course_id', 'rating', 'score_100', 'review_text', 'gold_label']

print('\n')
print('Train Label 분포')
print(train_df['gold_label'].value_counts().sort_index()) 
# gold_label
# 0    411
# 1     79
# Name: count, dtype: int64 

# 5. KoELECTRA Tokenizer 불러오기
# 수강평 텍스트를 KoELECTRA가 처리할 수 있는 토큰 ID 형태로 변환하기 위한 Tokenizer를 불러오기
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print('\n')
print('KoELECTRA Tokenizer 로딩 성공!')

# 6. 실제 수강평 1건 확인
# Train 데이터의 첫 번째 수강평 텍스트와 정답 라벨을 가져와서 출력
sample_text = train_df.iloc[0]['review_text']
sample_label = train_df.iloc[0]['gold_label']

print('\n')
print('----- 원본 수강평 -----')
print(sample_text)
# 제게 미흡했던 부분을 보강할수 있었고 팀 리더 역할을 경험해볼수 있어서 좋았습니다.

print('\n')
print('정답 Label : ', sample_label) # 0 (긍정)

# 7. 수강평 Tokenization
# 실제 수강평을 KoELECTRA 입력 형식으로 변환하기
encoded = tokenizer(
    sample_text,
    truncation=True, # 최대 길이를 초과하는 텍스트는 잘라내겠다.
    padding='max_length', # 모든 입력 길이를 max_length에 맞추겠다.
    max_length=128, # 최대 토큰 길이를 128로 설정하겠다.
    return_tensors='pt' # 결과를 PyTorch Tensor 형태로 반환하겠다.
)

# 8. Tokenization 결과 확인
print('\n')
print('----- Tokenization 결과 -----')

# input_ids : 각 토큰을 KoELECTRA Vocabulary의 정수 ID로 변환한 결과
print('input_ids : ')
print(encoded['input_ids'])
# tensor([[    2, 18526, 13031,  4398,  4820,  6486,  4292, 12573,  4519,  4141,
#           3249,  4480,  4219,  3675,  7909,  6639,  4292,  6745,  4151,  4899,
#           4141,  3249,  4025,  4129,  3311,  4494,  4576,  6216,    18,     3,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0]])

# attention_mask : 실제 토큰은 1, Padding 영역은 0으로 표시
print('\n')
print('attention_mask : ')
print(encoded['attention_mask'])
# tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#          0, 0, 0, 0, 0, 0, 0, 0]])

# 입력 Tensor의 크기를 확인
print('\n')
print('input_ids shape : ', encoded['input_ids'].shape) # torch.Size([1, 128])

# 사람이 확인할 수 있도록 실제 토큰 형태도 출력
tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])

print('\n')
print('변환된 Token : ')
print(tokens)
# ['[CLS]', '제게', '미흡', '##했', '##던', '부분', '##을', '보강', '##할', '##수', '있', '##었', '##고', 
# '팀', '리더', '역할', '##을', '경험', '##해', '##볼', '##수', '있', '##어', '##서', '좋', '##았', '##습', 
# '##니다', '.', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', 
# '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', 
# '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', 
# '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', 
# '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', 
# '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', 
# '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', 
# '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', 
# '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', 
# '[PAD]', '[PAD]']

# ---

# PyTorch에서 Dataset을 만들기 위한 클래스
from torch.utils.data import Dataset

# 9. KoELECTRA 학습용 Dataset 클래스 생성
# 수강평 텍스트와 Gold Label을 KoELECTRA 학습에 사용할 수 있는 형태로 변환하겠다.
class ReviewDataset(Dataset) :

    # Dataset 생성 시 수강평 텍스트, 정답 Label, Tokenizer를 저장하겠다.
    def __init__(self, dataframe, tokenizer, max_length=128) :
        self.texts = dataframe['review_text'].tolist()
        self.labels = dataframe['gold_label'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    # Dataset에 포함된 전체 데이터 개수를 반환하겠다.
    def __len__(self) :
        return len(self.texts)

    # 특정 위치의 수강평 1건을 KoELECTRA 입력 형태로 변환하여 반환하겠다.
    def __getitem__(self, idx) :

        text = self.texts[idx]
        label = self.labels[idx]

        # 수강평 텍스트를 KoELECTRA 입력 형식으로 Tokenization 하겠다.
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Tokenization 결과와 정답 Label을 반환하겠다.
        return {
            'input_ids' : encoded['input_ids'].squeeze(0),
            'attention_mask' : encoded['attention_mask'].squeeze(0),
            'labels' : label
        }

# 10. Train / Validation / Test Dataset 생성
train_dataset = ReviewDataset(train_df, tokenizer)
validation_dataset = ReviewDataset(validation_df, tokenizer)
test_dataset = ReviewDataset(test_df, tokenizer)

# 생성된 Dataset의 데이터 개수를 확인
print('\n')
print('----- Dataset 생성 결과 -----')
print('Train Dataset : ', len(train_dataset)) # 490
print('Validation Dataset : ', len(validation_dataset)) # 105
print('Test Dataset : ', len(test_dataset)) # 105

# ---

# PyTorch에서 Data를 불러올 때 사용하는 클래스
from torch.utils.data import DataLoader

# 11. DataLoader 생성
# Dataset의 데이터를 Batch 단위로 나누어 모델 학습에 사용할 수 있도록 구성
BATCH_SIZE = 16

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 생성된 DataLoader의 Batch 개수를 확인하겠다.
print('\n')
print('----- DataLoader 생성 결과 -----')
print('Train Batch : ', len(train_loader)) # 31
print('Validation Batch : ', len(validation_loader)) # 7
print('Test Batch : ', len(test_loader)) # 7

# ---

# Hugging Face Transformers에서 KoELECTRA 분류 모델을 불러오기 위한 클래스
from transformers import AutoModelForSequenceClassification

# 12. KoELECTRA 분류 모델 생성
# 사전학습된 KoELECTRA에 비부정(0) / 부정(1)을 분류하기 위한 분류층을 추가
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

# 13. GPU 사용 설정
# CUDA 사용이 가능하면 GPU를 사용하고, 사용할 수 없으면 CPU를 사용하겠다.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# KoELECTRA 모델을 설정한 장치로 이동
model = model.to(device)

# 모델이 사용할 장치를 확인
print('\n')
print('----- KoELECTRA 모델 생성 결과 -----')
print('사용 장치 : ', device) # cuda

if torch.cuda.is_available() :
    print('GPU : ', torch.cuda.get_device_name(0)) # NVIDIA GeForce RTX 5060 Laptop GPU

# ---

# 14. Class Weight 계산
# Train 데이터의 Label 개수를 기준으로 클래스별 가중치를 계산
class_counts = train_df['gold_label'].value_counts().sort_index()

negative_count = class_counts[0]
positive_count = class_counts[1]
total_count = len(train_df)

# 데이터가 적은 부정(1) 클래스의 Loss가 더 크게 반영되도록 가중치를 계산
class_weights = torch.tensor(
    [
        total_count / (2 * negative_count),
        total_count / (2 * positive_count)
    ],
    dtype=torch.float
)

# 계산한 Class Weight를 모델과 동일한 장치로 이동
class_weights = class_weights.to(device)

print('\n')
print('----- Class Weight 계산 결과 -----')
print('비부정(0) 데이터 수 : ', negative_count) # 411
print('부정(1) 데이터 수 : ', positive_count) # 79
print('Class Weight : ', class_weights) # tensor([0.5961, 3.1013], device='cuda:0')


# 15. Loss Function 설정
# 클래스 불균형을 반영하기 위해 Class Weight가 적용된 CrossEntropyLoss를 사용
loss_function = torch.nn.CrossEntropyLoss(
    weight=class_weights
)

print('\n')
print('Loss Function : ', loss_function) # CrossEntropyLoss()

# ---

# 16. Fine-tuning 실험 설정
# KoELECTRA Fine-tuning에서 비교할 Learning Rate를 설정
LEARNING_RATES = [
    2e-5,
    3e-5,
    5e-5
]

# 최대 Epoch를 10으로 설정
MAX_EPOCHS = 10

# Early Stopping 기준을 설정
PATIENCE = 2
MIN_DELTA = 0.0005

print('\n')
print('----- Fine-tuning 실험 설정 -----')
print('Learning Rate : ', LEARNING_RATES)
print('Max Epoch : ', MAX_EPOCHS)
print('Early Stopping Patience : ', PATIENCE)
print('Early Stopping Min Delta : ', MIN_DELTA)

# ---

# 모델의 분류 성능을 평가하기 위한 라이브러리
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 17. 모델 성능 평가 함수
# Validation 또는 Test 데이터의 분류 성능을 평가하겠다.
def evaluate_model(model, data_loader) :

    # 모델을 평가 모드로 설정하겠다.
    model.eval()

    # 실제 Label과 모델의 예측 Label을 저장하기 위한 List를 생성
    actual_labels = []
    predicted_labels = []

    # 평가 과정에서는 Gradient를 계산하지 않음
    with torch.no_grad() :

        # DataLoader에서 Batch 단위로 데이터를 가져오겠다.
        for batch in data_loader :

            # Batch 데이터를 GPU로 이동
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # KoELECTRA에 데이터를 입력하여 예측 결과를 구하기
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # 모델이 출력한 Logits를 가져오기
            logits = outputs.logits

            # Logits에서 가장 높은 값을 가진 Class를 최종 예측 Label로 선택
            predictions = torch.argmax(
                logits,
                dim=1
            )

            # 성능 평가를 위해 실제 Label과 예측 Label을 CPU로 이동하여 List에 저장
            actual_labels.extend(
                labels.cpu().tolist()
            )

            predicted_labels.extend(
                predictions.cpu().tolist()
            )

    # 실제 Label과 예측 Label을 비교하여 분류 성능을 계산
    accuracy = accuracy_score(
        actual_labels,
        predicted_labels
    )

    precision = precision_score(
        actual_labels,
        predicted_labels,
        zero_division=0
    )

    recall = recall_score(
        actual_labels,
        predicted_labels,
        zero_division=0
    )

    f1 = f1_score(
        actual_labels,
        predicted_labels,
        zero_division=0
    )

    # 비부정(0)과 부정(1)의 F1 Score를 동일한 비중으로 평가
    macro_f1 = f1_score(
        actual_labels,
        predicted_labels,
        average='macro',
        zero_division=0
    )

    return {
        'accuracy' : accuracy,
        'precision' : precision,
        'recall' : recall,
        'f1' : f1,
        'macro_f1' : macro_f1
    }

# ---

# 18. KoELECTRA Fine-tuning 함수
# Learning Rate별로 KoELECTRA Fine-tuning을 수행하겠다.
def train_experiment(learning_rate) :

    # 각 Learning Rate를 동일한 조건에서 비교하기 위해 Random Seed를 다시 고정
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available() :

        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

    # 각 실험에서 Train 데이터의 Shuffle 순서를 동일하게 설정
    generator = torch.Generator()
    generator.manual_seed(SEED)

    # Train Dataset을 Batch 단위로 불러오기 위한 DataLoader를 생성
    experiment_train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator
    )

    print('\n')
    print('-----')
    print('Learning Rate : ', learning_rate)
    print('-----')

    # 각 Learning Rate마다 동일한 사전학습 KoELECTRA 모델에서 학습을 시작
    experiment_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )

    # KoELECTRA 모델을 설정한 장치로 이동
    experiment_model = experiment_model.to(device)

    # KoELECTRA Fine-tuning을 위해 AdamW Optimizer를 사용
    optimizer = torch.optim.AdamW(
        experiment_model.parameters(),
        lr=learning_rate
    )

    print('Optimizer : ', optimizer.__class__.__name__)

    # 가장 높은 Validation Macro-F1을 저장하기 위한 변수를 초기화
    best_macro_f1 = -1.0

    # 가장 높은 Validation Macro-F1이 나온 Epoch를 저장
    best_epoch = 0

    # Early Stopping을 위한 Count를 초기화
    patience_count = 0

    # Learning Rate별 모델을 저장하기 위한 폴더를 생성
    os.makedirs('../models', exist_ok=True)

    # Learning Rate별 Best Model 저장 경로를 설정
    model_name = str(learning_rate).replace('.', '_')

    save_path = (
        '../models/koelectra_lr_'
        + model_name
        + '.pt'
    )

    # Epoch별 학습 결과를 저장하기 위한 List를 생성
    history = []

    # 최대 10 Epoch까지 Fine-tuning을 수행하겠다.
    for epoch in range(MAX_EPOCHS) :

        # 모델을 학습 모드로 설정하겠다.
        experiment_model.train()

        # Epoch 전체 Loss를 저장하기 위한 변수를 초기화
        total_loss = 0

        print('\n')
        print('----- Epoch : ', epoch + 1, '-----')

        # Train DataLoader에서 Batch 단위로 데이터를 가져와 학습하겠다.
        for batch_index, batch in enumerate(experiment_train_loader) :

            # Batch 데이터를 GPU로 이동시키겠다.
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 이전 Batch에서 계산된 Gradient를 초기화하겠다.
            optimizer.zero_grad()

            # KoELECTRA에 수강평 데이터를 입력하여 예측 결과를 구하겠다.
            outputs = experiment_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # 모델이 출력한 비부정(0) / 부정(1)의 Logits를 가져오겠다.
            logits = outputs.logits

            # 예측 결과와 실제 Gold Label을 비교하여 Loss를 계산하겠다.
            loss = loss_function(
                logits,
                labels
            )

            # Loss를 기준으로 역전파를 수행하여 Gradient를 계산하겠다.
            loss.backward()

            # 계산된 Gradient를 이용하여 KoELECTRA의 가중치를 업데이트하겠다.
            optimizer.step()

            # 현재 Batch의 Loss를 Epoch 전체 Loss에 누적하겠다.
            total_loss += loss.item()

            # 5 Batch마다 현재 학습 진행 상황과 Loss를 출력하겠다.
            if (batch_index + 1) % 5 == 0 :

                print(
                    'Batch : ',
                    batch_index + 1,
                    '/',
                    len(experiment_train_loader),
                    ' | Loss : ',
                    round(loss.item(), 4)
                )

        # 한 Epoch의 평균 Loss를 계산하겠다.
        average_loss = (
            total_loss
            / len(experiment_train_loader)
        )

        # 현재 Epoch의 학습 결과를 Validation 데이터로 평가하겠다.
        validation_metrics = evaluate_model(
            experiment_model,
            validation_loader
        )

        print('\n')
        print('Epoch 평균 Loss : ', round(average_loss, 4))

        print(
            'Validation Accuracy : ',
            round(validation_metrics['accuracy'], 4)
        )

        print(
            'Validation Precision : ',
            round(validation_metrics['precision'], 4)
        )

        print(
            'Validation Recall : ',
            round(validation_metrics['recall'], 4)
        )

        print(
            'Validation F1 Score : ',
            round(validation_metrics['f1'], 4)
        )

        print(
            'Validation Macro-F1 : ',
            round(validation_metrics['macro_f1'], 4)
        )

        # 현재 Epoch의 학습 및 Validation 결과를 저장
        history.append({
            'learning_rate' : learning_rate,
            'epoch' : epoch + 1,
            'train_loss' : average_loss,
            'accuracy' : validation_metrics['accuracy'],
            'precision' : validation_metrics['precision'],
            'recall' : validation_metrics['recall'],
            'f1' : validation_metrics['f1'],
            'macro_f1' : validation_metrics['macro_f1']
        })

        # 현재 Epoch의 Validation Macro-F1을 가져오기
        current_macro_f1 = validation_metrics['macro_f1']

        # Validation Macro-F1이 기존보다 개선되었는지 확인
        if current_macro_f1 > best_macro_f1 + MIN_DELTA :

            # 현재 Macro-F1을 Best Score로 저장
            best_macro_f1 = current_macro_f1

            # 현재 Epoch를 Best Epoch로 저장
            best_epoch = epoch + 1

            # Early Stopping Count를 초기화
            patience_count = 0

            # 현재 모델을 해당 Learning Rate의 Best Model로 저장
            torch.save(
                experiment_model.state_dict(),
                save_path
            )

            print('Best Model 저장')

        else :

            # Validation Macro-F1이 개선되지 않은 횟수를 증가
            patience_count += 1

            print(
                'Early Stopping Count : ',
                patience_count,
                '/',
                PATIENCE
            )

        # Validation Macro-F1이 2 Epoch 연속 개선되지 않으면 학습을 종료하겠다.
        if patience_count >= PATIENCE :

            print('Early Stopping 적용')
            break

    # 해당 Learning Rate의 최적 학습 결과를 반환
    return {
        'learning_rate' : learning_rate,
        'best_epoch' : best_epoch,
        'best_macro_f1' : best_macro_f1,
        'model_path' : save_path,
        'history' : history
    }

# ---

# 19. Learning Rate별 Fine-tuning
# 설정한 3개의 Learning Rate에 대해 Fine-tuning을 수행하겠다.
experiment_results = []

for learning_rate in LEARNING_RATES :

    result = train_experiment(
        learning_rate
    )

    experiment_results.append(
        result
    )

# ---

# 20. Learning Rate별 Validation 결과 비교
print('\n')
print('-----')
print('Fine-tuning 실험 결과')
print('-----')

summary_results = []

for result in experiment_results :

    print(
        'Learning Rate : ',
        result['learning_rate'],
        '| Best Epoch : ',
        result['best_epoch'],
        '| Best Macro-F1 : ',
        round(result['best_macro_f1'], 4)
    )

    # Learning Rate별 최적 Validation 결과를 저장
    summary_results.append({
        'learning_rate' : result['learning_rate'],
        'best_epoch' : result['best_epoch'],
        'best_macro_f1' : result['best_macro_f1']
    })

# Learning Rate별 실험 결과를 DataFrame으로 변환
summary_df = pd.DataFrame(
    summary_results
)

# Learning Rate별 실험 결과를 CSV 파일로 저장
summary_df.to_csv(
    'fine_tuning_results.csv',
    index=False,
    encoding='utf-8-sig'
)

# ---

# 21. 최적 모델 선정
# Validation Macro-F1이 가장 높은 모델을 최종 모델로 선정하겠다.
best_result = max(
    experiment_results,
    key=lambda result : result['best_macro_f1']
)

print('\n')
print('-----')
print('최적 모델 선정')
print('-----')

print(
    'Learning Rate : ',
    best_result['learning_rate']
)

print(
    'Best Epoch : ',
    best_result['best_epoch']
)

print(
    'Validation Macro-F1 : ',
    round(best_result['best_macro_f1'], 4)
)

# ---

# 22. 최적 모델 불러오기
# 선정된 Learning Rate의 Best Model을 불러오겠다.
best_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

# 저장된 Best Model의 가중치를 불러오기
best_model.load_state_dict(
    torch.load(
        best_result['model_path'],
        map_location=device
    )
)

# Best Model을 설정한 장치로 이동
best_model = best_model.to(device)

# ---

# 23. Test 데이터 성능 평가
# 모델 선정에 사용하지 않은 Test 데이터로 최종 모델의 성능을 평가하겠다.
test_metrics = evaluate_model(
    best_model,
    test_loader
)

print('\n')
print('-----')
print('최종 Test 평가')
print('-----')

print(
    'Accuracy : ',
    round(test_metrics['accuracy'], 4)
)

print(
    'Precision : ',
    round(test_metrics['precision'], 4)
)

print(
    'Recall : ',
    round(test_metrics['recall'], 4)
)

print(
    'F1 Score : ',
    round(test_metrics['f1'], 4)
)

print(
    'Macro-F1 : ',
    round(test_metrics['macro_f1'], 4)
)