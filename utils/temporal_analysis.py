from utils.insight_extraction import extract_complex_insights
import pandas as pd
from textblob import TextBlob
import re

def extract_temporal_data(transcript_data):
    """어닝콜 텍스트에서 시간순 데이터 추출"""
    temporal_data = []
    
    # 문장 단위로 분리 (발화자 구분 유지)
    lines = transcript_data.split('\n')
    current_speaker = "Unknown"
    buffer_text = ""
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        # 발화자 확인
        if "Jeremy Barnum" in line:
            current_speaker = "Jeremy Barnum"
            # 다음 줄이 "Chief Financial Officer"인 경우 스킵
            if i+1 < len(lines) and "Chief Financial Officer" in lines[i+1]:
                continue
        elif "Jamie Dimon" in line and "CEO" in line:
            current_speaker = "Jamie Dimon"
            continue
        elif "Good morning" in line and "Welcome to" in line:
            current_speaker = "Operator"
            
        # 발화자 이름만 있는 줄은 건너뛰기
        if line == current_speaker:
            continue
            
        # 직책만 있는 줄은 건너뛰기
        if "Chief Financial Officer" in line or "Chairman and CEO" in line:
            continue
            
        # 문장 분리 (숫자가 포함된 경우 주의)
        sentences = []
        for part in line.split('.'):
            if part.strip():
                # 숫자가 포함된 경우 이전 문장과 합치기
                if sentences and any(c.isdigit() for c in part):
                    sentences[-1] = sentences[-1] + '.' + part
                else:
                    sentences.append(part)
        
        # 데이터 추가
        for sentence in sentences:
            if sentence.strip():
                sentiment = TextBlob(sentence).sentiment.polarity
                temporal_data.append({
                    'index': len(temporal_data),
                    'text': sentence.strip(),
                    'speaker': current_speaker,
                    'sentiment_score': sentiment
                })
    
    # DataFrame 생성
    temporal_df = pd.DataFrame(temporal_data)
    temporal_df.set_index('index', inplace=True)
    
    return temporal_df

def identify_key_events(temporal_data):
    """주요 이벤트 식별"""
    key_events = []
    
    # 감성 점수의 급격한 변화 탐지
    sentiment_changes = temporal_data['sentiment_score'].diff()
    threshold = sentiment_changes.std() * 2
    
    for idx in temporal_data.index:
        if abs(sentiment_changes.get(idx, 0)) > threshold:
            key_events.append({
                'time': idx,
                'text': temporal_data.loc[idx, 'text'],
                'speaker': temporal_data.loc[idx, 'speaker'],
                'sentiment_change': sentiment_changes[idx]
            })
    
    return key_events