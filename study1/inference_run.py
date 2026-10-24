# KoELECTRA inference only (no training, no fine-tuning).
# 저장된 koelectra_lr_5e-05.pt 가중치를 그대로 로드하여 model.eval() + torch.no_grad() 상태로 예측만 수행한다.
import os

import pandas as pd
import torch
from transformers import AutoTokenizer, ElectraForSequenceClassification

MODEL_NAME = 'monologg/koelectra-base-v3-discriminator'
PT_PATH = '../models/koelectra_lr_5e-05.pt'
CSV_PATH = '../data/work24_reviews_analysis.csv'
OUT_PATH = 'work24_reviews_with_prediction.csv'

MAX_LENGTH = 128
BATCH_SIZE = 16
DROPOUT = 0.1

POS_THRESHOLD = 3.6   # rating >= 3.6 -> 만족도 긍정
NEUTRAL_RATING = 2.9  # 중립 (불일치 분석 제외)
NEG_THRESHOLD = 2.1   # rating <= 2.1 -> 만족도 부정

def load_model(device):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = ElectraForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        hidden_dropout_prob=DROPOUT,
    )
    state_dict = torch.load(PT_PATH, map_location='cpu')  # 읽기만 하며 파일은 수정하지 않음
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return tokenizer, model

def predict(texts, tokenizer, model, device):
    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            enc = tokenizer(
                batch,
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
                return_tensors='pt',
            ).to(device)
            logits = model(**enc).logits
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
    return preds

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    df = pd.read_csv(CSV_PATH, encoding='utf-8')
    print(f'loaded rows: {len(df):,}')

    tokenizer, model = load_model(device)
    texts = df['review_text'].fillna('').astype(str).tolist()
    df['prediction'] = predict(texts, tokenizer, model, device)

    df.to_csv(OUT_PATH, index=False, encoding='utf-8-sig')
    print(f'saved: {OUT_PATH}')
    dist = df['prediction'].value_counts().sort_index().to_dict()
    print(f'prediction 분포: {dist}')

    pos = df[df['rating'] >= POS_THRESHOLD]
    neg = df[df['rating'] <= NEG_THRESHOLD]
    neu = df[df['rating'] == NEUTRAL_RATING]

    pos_nonneg = int((pos['prediction'] == 0).sum())
    pos_neg = int((pos['prediction'] == 1).sum())
    neg_nonneg = int((neg['prediction'] == 0).sum())
    neg_neg = int((neg['prediction'] == 1).sum())

    hidden_negative = pos_neg
    hidden_ratio = hidden_negative / len(pos) * 100

    print()
    print('[만족도 - 텍스트 불일치 분석]')
    print()
    print('| 만족도 | 텍스트 비부정 | 텍스트 부정 |')
    print('|---|---:|---:|')
    print(f'| 긍정 | {pos_nonneg:,}건 | {pos_neg:,}건 |')
    print(f'| 부정 | {neg_nonneg:,}건 | {neg_neg:,}건 |')
    print()
    print(f'Hidden Negative : {hidden_negative:,}건')
    print(f'Hidden Negative 비율 : {hidden_ratio:.2f}%')
    print()

    pos_sum = pos_nonneg + pos_neg
    neg_sum = neg_nonneg + neg_neg
    target_sum = pos_sum + neg_sum
    total_sum = target_sum + len(neu)

    def mark(ok):
        return 'OK' if ok else 'MISMATCH'

    print('검산:')
    print(f'- 긍정 합계: {pos_sum:,}건 (기대 2,220) {mark(pos_sum == 2220)}')
    print(f'- 부정 합계: {neg_sum:,}건 (기대 88) {mark(neg_sum == 88)}')
    print(f'- 분석 대상 합계: {target_sum:,}건 (기대 2,308) {mark(target_sum == 2308)}')
    print(f'- 중립 포함 전체: {total_sum:,}건 (기대 2,450) {mark(total_sum == 2450)}')
    print()
    print('[Hidden Negative 만족도 5.0 사례]')
    print(
        df[
            (df['rating'] == 5.0) &
            (df['prediction'] == 1)
        ][['rating', 'review_text', 'prediction']]
        .head(20)
        .to_string(index=False)
    )

if __name__ == '__main__':
    main()
