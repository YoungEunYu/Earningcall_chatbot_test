import pandas as pd
import numpy as np
from bertopic import BERTopic
import nltk
from nltk.tokenize import sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter
import networkx as nx
import re
import os
import ssl

# SSL 인증서 설정 (기존 코드)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# NLTK 데이터 다운로드
nltk.download('punkt')
nltk.download('vader_lexicon')

def preprocess_text(text):
    """텍스트 전처리 함수"""
    # 특수문자 제거 및 정리
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def analyze_sentiment_and_keywords(text):
    """감성 분석 및 주요 키워드 추출"""
    sia = SentimentIntensityAnalyzer()
    sentences = sent_tokenize(text)
    
    sentiment_data = {
        'positive_keywords': Counter(),
        'negative_keywords': Counter(),
        'neutral_keywords': Counter(),
        'sentence_sentiments': [],
        'sentence_texts': []
    }
    
    # 금융 관련 키워드 필터
    financial_terms = {'revenue', 'profit', 'growth', 'margin', 'earnings', 
                      'investment', 'market', 'strategy', 'risk', 'capital',
                      'assets', 'dividend', 'cost', 'performance', 'outlook'}
    
    for sentence in sentences:
        # 감성 점수 계산
        scores = sia.polarity_scores(sentence)
        compound_score = scores['compound']
        
        # 문장과 감성 점수 저장
        sentiment_data['sentence_sentiments'].append(compound_score)
        sentiment_data['sentence_texts'].append(sentence)
        
        # 키워드 추출 및 분류
        words = set(word.lower() for word in re.findall(r'\b\w+\b', sentence))
        relevant_words = words.intersection(financial_terms)
        
        if compound_score > 0.2:
            sentiment_data['positive_keywords'].update(relevant_words)
        elif compound_score < -0.2:
            sentiment_data['negative_keywords'].update(relevant_words)
        else:
            sentiment_data['neutral_keywords'].update(relevant_words)
    
    return sentiment_data

def create_keyword_network(text):
    """향상된 키워드 네트워크 생성"""
    sentences = sent_tokenize(text)
    keyword_pairs = []
    
    # 금융 관련 키워드 필터
    financial_terms = {'revenue', 'profit', 'growth', 'margin', 'earnings', 
                      'investment', 'market', 'strategy', 'risk', 'capital',
                      'assets', 'dividend', 'cost', 'performance', 'outlook'}
    
    for sentence in sentences:
        words = [word.lower() for word in re.findall(r'\b\w+\b', sentence)]
        relevant_words = [w for w in words if w in financial_terms]
        
        # 한 문장 내의 키워드 페어 생성
        for i in range(len(relevant_words)):
            for j in range(i+1, len(relevant_words)):
                keyword_pairs.append((relevant_words[i], relevant_words[j]))
    
    # 키워드 간 연관성 계산
    edge_weights = Counter(keyword_pairs)
    
    # 네트워크 데이터 생성
    network_data = {
        'nodes': list(financial_terms),
        'edges': [(pair[0], pair[1], weight) 
                 for pair, weight in edge_weights.items()]
    }
    
    return network_data

def create_time_series_data():
    # 데이터 폴더 생성
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # 예시 데이터 생성 (4분기 동안의 데이터)
    data = {
        'date': ['2023-10-01', '2023-10-02', '2023-10-03', '2023-10-04'],  # 실제 날짜로 변경
        'revenue': [100, 150, 200, 250],  # 실제 수치로 변경
        'profit': [30, 50, 70, 90],  # 실제 수치로 변경
        'expenses': [70, 100, 130, 160]  # 실제 수치로 변경
    }
    
    time_series_df = pd.DataFrame(data)
    
    # CSV 파일로 저장
    time_series_df.to_csv('data/time_series_data.csv', index=False)
    print("Time series data has been created and saved.")

def main():
    create_time_series_data()  # 시계열 데이터 생성 및 저장

if __name__ == "__main__":
    main() 