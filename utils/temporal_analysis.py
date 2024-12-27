import pandas as pd
import numpy as np
from textblob import TextBlob
import re
from datetime import datetime, timedelta

def clean_speaker_name(speaker):
    """발언자 이름 정제"""
    # 불필요한 단어 제거
    remove_words = ['Next', 'Operator', 'Unknown']
    if speaker in remove_words:
        return None
    
    # 직함 정리
    titles = ['CEO', 'CFO', 'Analyst', 'Director', 'Manager']
    for title in titles:
        if title.lower() in speaker.lower():
            return f"{speaker.split('-')[0].strip()} ({title})"
    
    return speaker.strip()

def extract_temporal_data(transcript_text):
    """어닝콜 텍스트에서 시간 정보와 발언을 추출"""
    
    if not isinstance(transcript_text, str):
        print("Error: transcript_text is not a string")
        return pd.DataFrame()

    # 디버깅: 입력 텍스트 확인
    print("\nDEBUG: Text type:", type(transcript_text))
    print("DEBUG: Text length:", len(transcript_text))

    segments = []
    current_speaker = None
    current_title = None
    
    # 라인별 처리
    lines = transcript_text.split('\n')
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        # 이름 라인 확인 (대문자로 시작하는 이름)
        if re.match(r'^[A-Z][A-Za-z\s\.]+$', line):
            current_speaker = line
            # 다음 줄이 직함인지 확인
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line.startswith('Analyst') or 'at' in next_line:
                    current_title = next_line
                    current_speaker = f"{current_speaker} ({current_title})"
            continue

        # 발언 내용 확인
        if current_speaker and not line.startswith('Analyst') and 'at' not in line:
            text = line.strip()
            
            # 다음 줄의 텍스트 추가 (새로운 발언자가 나올 때까지)
            next_line_idx = i + 1
            while next_line_idx < len(lines):
                next_line = lines[next_line_idx].strip()
                if not next_line or re.match(r'^[A-Z][A-Za-z\s\.]+$', next_line):
                    break
                if not next_line.startswith('Analyst') and 'at' not in next_line:
                    text += " " + next_line
                next_line_idx += 1
            
            if text:
                segments.append({
                    'timestamp': str(i),
                    'speaker': current_speaker,
                    'text': text,
                    'sentiment_score': TextBlob(text).sentiment.polarity
                })
                print(f"Added segment for {current_speaker}: {text[:100]}...")

    df = pd.DataFrame(segments)
    
    if not df.empty:
        print(f"\nExtracted {len(df)} segments")
        print("Speakers found:", df['speaker'].unique())
    else:
        print("\nNo speakers were extracted from the text")
    
    return df

def classify_topic(text):
    """텍스트의 주제 분류"""
    topics = {
        'Revenue': ['revenue', 'sales', 'income', 'earnings'],
        'Risk': ['risk', 'challenge', 'uncertainty', 'concern'],
        'Future': ['outlook', 'guidance', 'forecast', 'expect'],
        'Market': ['market', 'competition', 'industry', 'sector']
    }
    
    text_lower = text.lower()
    for topic, keywords in topics.items():
        if any(keyword in text_lower for keyword in keywords):
            return topic
    return 'Other'

def identify_key_events(df):
    """주요 이벤트 식별"""
    # 디버깅을 위한 출력 추가
    print("Identifying key events...")
    print("Input DataFrame columns:", df.columns.tolist())
    
    if 'sentiment_score' not in df.columns:
        print("Warning: sentiment_score column not found!")
        print("Available columns:", df.columns.tolist())
        return pd.DataFrame()  # 빈 DataFrame 반환
    
    events = []
    
    # 감성 점수의 급격�� 변화 탐지
    df['sentiment_change'] = df['sentiment_score'].diff()
    significant_changes = df[abs(df['sentiment_change']) > df['sentiment_change'].std()]
    
    for idx, row in significant_changes.iterrows():
        events.append({
            'timestamp': row['timestamp'],
            'event_type': 'significant_change',
            'description': f"Significant sentiment change ({row['sentiment_change']:.2f})"
        })
    
    return pd.DataFrame(events) 