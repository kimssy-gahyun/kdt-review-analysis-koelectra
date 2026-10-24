# Hugging Face Transformers 라이브러리에서 사전학습 모델의 토크나이저와 문장 분류 모델을 불러오기
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 사용할 KoELECTRA 사전학습 모델 지정
MODEL_NAME = 'monologg/koelectra-base-v3-discriminator'
# 한국어 말뭉치로 사전학습된 KoELECTRA Base v3 모델을 사용하겠다.

# KoELECTRA 전용 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# 입력된 수강평 텍스트를 KoELECTRA가 처리할 수 있는 토큰 ID로 변환하는 역할

# 사전학습된 KoELECTRA 모델 로드
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)
# num_labels=2로 설정하여 비부정(0) / 부정(1)을 분류하는 이진 분류 모델로 구성

# 모델이 정상적으로 로드되었는지 확인
print('KoELECTRA 로딩 성공!')