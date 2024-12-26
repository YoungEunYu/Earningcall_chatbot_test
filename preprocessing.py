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

def main():
    # 데이터 폴더 생성
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # 원본 텍스트 파일 읽기
    with open('data/JPMorganChase_Q3_2024.txt', 'r', encoding='utf-8') as file:
        text = file.read()
    
    # 텍스트 전처리 (기존 코드)
    processed_text = preprocess_text(text)
    
    # 기존의 토픽 모델링 (유지)
    sentences = sent_tokenize(processed_text)
    topic_model = BERTopic(language="english", 
                          min_topic_size=5, 
                          nr_topics=8,
                          verbose=True)
    
    topics, _ = topic_model.fit_transform(sentences)
    
    # 토픽 정보 추출 및 저장
    topic_info = topic_model.get_topic_info()
    topic_info = topic_info.rename(columns={
        'Count': 'Size',
        'Topic': 'Topic_Num'
    })
    
    # 각 토픽의 상위 문구 추출
    topic_phrases = []
    for topic in topic_info['Topic_Num']:
        if topic != -1:  # -1은 아웃라이어 토픽
            phrases = [phrase for phrase, _ in topic_model.get_topic(topic)][:3]
            topic_phrases.append(' | '.join(phrases))
        else:
            topic_phrases.append('')
    
    # 토픽 정보에 문구 추가
    topic_info['Top_Phrases'] = topic_phrases
    
    # 새로운 분석 추가
    sentiment_data = analyze_sentiment_and_keywords(processed_text)
    network_data = create_keyword_network(processed_text)
    
    # 결과 저장
    topic_info.to_csv('data/topic_info.csv', index=False)
    
    # 감성 분석 결과 저장
    sentiment_df = pd.DataFrame({
        'text': sentiment_data['sentence_texts'],
        'sentiment': sentiment_data['sentence_sentiments']
    })
    sentiment_df.to_csv('data/sentiment_analysis.csv', index=False)
    
    # 키워드 데이터 저장 방식 수정
    keyword_data = {
        'keyword': [],
        'sentiment': [],
        'count': []
    }
    
    # 긍정 키워드
    for word, count in sentiment_data['positive_keywords'].items():
        keyword_data['keyword'].append(word)
        keyword_data['sentiment'].append('positive')
        keyword_data['count'].append(count)
    
    # 부정 키워드
    for word, count in sentiment_data['negative_keywords'].items():
        keyword_data['keyword'].append(word)
        keyword_data['sentiment'].append('negative')
        keyword_data['count'].append(count)
    
    # 중립 키워드
    for word, count in sentiment_data['neutral_keywords'].items():
        keyword_data['keyword'].append(word)
        keyword_data['sentiment'].append('neutral')
        keyword_data['count'].append(count)
    
    # DataFrame으로 변환하여 저장
    keyword_df = pd.DataFrame(keyword_data)
    keyword_df.to_csv('data/keyword_analysis.csv', index=False)
    
    # 네트워크 데이터 저장
    network_df = pd.DataFrame(network_data['edges'], 
                            columns=['source', 'target', 'weight'])
    network_df.to_csv('data/network_data.csv', index=False)
    
    print("전처리 및 분석 완료!")

if __name__ == "__main__":
    main() 