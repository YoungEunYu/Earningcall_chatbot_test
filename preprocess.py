from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import time
import json
from datetime import datetime
import os
import re
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def ensure_data_directory():
    """데이터 디렉토리와 필요한 파일들이 있는지 확인"""
    # 데이터 디렉토리 생성
    if not os.path.exists('data'):
        os.makedirs('data')
        print("'data' 디렉토리 생성됨")

def preprocess_text(text):
    """텍스트 전처리 함수"""
    # 1. 기본 클리닝
    text = re.sub(r'\s+', ' ', text)
    
    # 2. 불필요한 내용 제거
    patterns_to_remove = [
        r'operator:.*?(?=\n)',
        r'\[.*?\]',
        r'good morning|good afternoon',
        r'thank you|thanks',
        r'question and answer|q&a',
        r'\d{1,2}:\d{2}',
        r'conference call',
        r'next question',
        r'please go ahead',
    ]
    
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # 3. 의미 있는 문장 단위로 분리
    sentences = []
    for paragraph in text.split('\n'):
        # 쉼표, 세미콜론 등으로 분리하여 phrase 단위 추출
        phrases = re.split(r'[.;,]', paragraph)
        sentences.extend([p.strip() for p in phrases if len(p.strip().split()) > 3])
    
    return sentences

def preprocess_and_analyze():
    ensure_data_directory()
    start_time = time.time()
    
    print("데이터 로딩 ��...")
    try:
        with open('data/JPMorganChase_Q3_2024.txt', 'r', encoding='utf-8') as file:
            text = file.read()
    except FileNotFoundError:
        text = "This is a sample text for JPMorgan Chase earnings call."
        with open('data/JPMorganChase_Q3_2024.txt', 'w', encoding='utf-8') as file:
            file.write(text)
        print("샘플 텍스트 파일 생성됨")
    
    print("텍스트 전처리 중...")
    processed_text = preprocess_text(text)
    
    print("토픽 모델링 중...")
    # n-gram 범위를 설정하여 phrase 추출
    vectorizer_model = CountVectorizer(ngram_range=(1, 3), 
                                     stop_words="english",
                                     min_df=2)
    
    # BERTopic 모델 설정
    topic_model = BERTopic(
        language="english",
        min_topic_size=2,
        vectorizer_model=vectorizer_model,
        verbose=True
    )
    
    topics, probs = topic_model.fit_transform(processed_text)
    
    # 토픽 정보를 더 자세하게 가공
    topic_info = topic_model.get_topic_info()
    topic_details = []
    
    for topic in topic_info['Topic']:
        if topic != -1:  # -1은 아웃라이어 토픽
            topic_words = topic_model.get_topic(topic)
            # (단어, 가중치) 튜플에서 phrase만 추출
            phrases = [word for word, _ in topic_words[:5]]
            topic_details.append({
                'Topic_Num': topic,
                'Size': topic_info.loc[topic_info['Topic'] == topic, 'Count'].iloc[0],
                'Top_Phrases': ' | '.join(phrases)
            })
    
    # 상세 토픽 정보를 DataFrame으로 변환
    topic_details_df = pd.DataFrame(topic_details)
    
    print("결과 저장 중...")
    # 처리된 텍스트 저장
    with open('data/processed_JPMorganChase_Q3_2024.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(processed_text))
    
    # 상세 토픽 정보 저장
    topic_details_df.to_csv('data/topic_info.csv', index=False)
    
    end_time = time.time()
    processing_info = {
        'processing_time': end_time - start_time,
        'processed_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('data/processing_info.json', 'w') as f:
        json.dump(processing_info, f)
    
    print(f"처리 완료! 소요 시간: {end_time - start_time:.2f}초")

if __name__ == "__main__":
    preprocess_and_analyze()
