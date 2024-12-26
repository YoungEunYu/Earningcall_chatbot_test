import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
from gensim.models.phrases import Phrases, Phraser
import re
import os
import pandas as pd

# NLTK 데이터 다운로드
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

# 금융 도메인 사전 (더 구체적인 키워드 추가)
FINANCIAL_PHRASES = {
    'Revenue & Growth': [
        'revenue', 'growth', 'sales', 'income', 'earnings', 'profit', 'margin',
        'net revenue', 'revenue growth', 'sales growth', 'organic growth',
        'market share', 'top line', 'bottom line', 'profitability'
    ],
    'Credit & Risk': [
        'credit', 'risk', 'loan', 'lending', 'portfolio', 'default', 'exposure',
        'credit quality', 'risk management', 'credit risk', 'loan loss',
        'provision', 'reserve', 'allowance', 'delinquency'
    ],
    'Capital & Investment': [
        'capital', 'investment', 'asset', 'equity', 'tier', 'ratio',
        'capital ratio', 'return on equity', 'roe', 'roa', 'investment banking',
        'wealth management', 'assets under management', 'aum'
    ],
    'Market & Trading': [
        'market', 'trading', 'volatility', 'spread', 'fee', 'commission',
        'fixed income', 'equity trading', 'market making', 'securities',
        'derivatives', 'fx', 'foreign exchange', 'treasury'
    ],
    'Digital & Technology': [
        'digital', 'technology', 'platform', 'mobile', 'online', 'payment',
        'digital banking', 'mobile app', 'digital platform', 'innovation',
        'fintech', 'cyber', 'cloud', 'automation', 'ai'
    ],
    'Client & Service': [
        'client', 'customer', 'service', 'relationship', 'satisfaction',
        'client relationship', 'customer service', 'retail banking',
        'commercial banking', 'corporate banking', 'private banking'
    ],
    'Cost & Efficiency': [
        'cost', 'expense', 'efficiency', 'saving', 'productivity', 'overhead',
        'operating expense', 'cost reduction', 'cost management',
        'efficiency ratio', 'operating leverage', 'optimization'
    ],
    'Strategy & Outlook': [
        'strategy', 'outlook', 'guidance', 'forecast', 'target', 'plan',
        'strategic', 'initiative', 'priority', 'opportunity', 'challenge',
        'competitive', 'market position', 'leadership'
    ]
}

def load_earnings_calls(data_dir='data/raw'):
    """1년치 어닝콜 데이터 로드"""
    texts = []
    dates = []
    
    print(f"Looking for files in: {os.path.abspath(data_dir)}")  # 디버깅
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            print(f"Processing file: {filename}")  # 디버깅
            with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                print(f"Text length: {len(text)}")  # 디버깅
                if len(text.strip()) > 0:  # 빈 파일 체크
                    texts.append(text)
                    date = filename.split('_')[1] + '_' + filename.split('_')[2].replace('.txt', '')
                    dates.append(date)
                else:
                    print(f"Warning: Empty file found: {filename}")
    
    if not texts:
        raise ValueError(f"No valid text files found in {data_dir}")
    
    return pd.DataFrame({'date': dates, 'text': texts})

def extract_financial_phrases(text, phrases_dict=FINANCIAL_PHRASES):
    """금융 관련 구절 추출"""
    found_phrases = []
    text_lower = text.lower()
    
    for category, phrases in phrases_dict.items():
        for phrase in phrases:
            if phrase in text_lower:
                found_phrases.append((phrase, category))
    
    return found_phrases

def preprocess_text(text):
    """텍스트 전처리"""
    print(f"Input text length: {len(text)}")  # 디버깅
    
    # 1. 기본 클리닝
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    
    # 2. 문장 토큰화
    sentences = sent_tokenize(text)
    print(f"Number of sentences: {len(sentences)}")  # 디버깅
    
    # 3. 불용어 설정 (더 줄임)
    stop_words = set(['the', 'and', 'to', 'of', 'a', 'in'])
    
    # 4. 토큰화 및 전처리
    processed_sentences = []
    
    for sentence in sentences:
        # 금융 관련 구절 보존
        found_phrases = extract_financial_phrases(sentence)
        for phrase, _ in found_phrases:
            sentence = sentence.replace(phrase, phrase.replace(' ', '_'))
        
        # 토큰화
        tokens = word_tokenize(sentence)
        
        # 최소한의 전처리만 수행
        tokens = [token.lower() for token in tokens 
                 if token.isalnum()  # 알파벳과 숫자만 허용
                 and token not in stop_words
                 and len(token) > 1]
        
        if tokens:
            processed_sentences.append(tokens)
    
    print(f"Processed sentences: {len(processed_sentences)}")  # 디버깅
    return processed_sentences

def extract_topics(texts, num_topics=8):
    """토픽 모델링"""
    print(f"Number of input texts: {len(texts)}")  # 디버깅
    
    # 1. 전처리
    all_sentences = []
    for text in texts:
        sentences = preprocess_text(text)
        if sentences:
            all_sentences.extend(sentences)
    
    print(f"Total processed sentences: {len(all_sentences)}")  # 디버깅
    
    if not all_sentences:
        raise ValueError("No valid text found after preprocessing")
    
    # 2. Bigram 모델 학습 (더 낮은 임계값)
    bigram = Phrases(all_sentences, min_count=2, threshold=10)
    bigram_mod = Phraser(bigram)
    
    # 3. Bigram 적용
    texts_bigrams = [bigram_mod[doc] for doc in all_sentences]
    
    # 4. 사전 및 코퍼스 생성
    dictionary = corpora.Dictionary(texts_bigrams)
    corpus = [dictionary.doc2bow(text) for text in texts_bigrams]
    
    print(f"Dictionary size: {len(dictionary)}")  # 디버깅
    print(f"Corpus size: {len(corpus)}")  # 디버깅
    
    # 5. LDA 모델 학습
    lda_model = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        update_every=1,
        chunksize=100,
        passes=10,
        alpha='auto',
        per_word_topics=True
    )
    
    # 6. 토픽 정보 추출 및 레이블링
    topics = []
    for idx in range(num_topics):
        topic_terms = lda_model.show_topic(idx, topn=15)  # 더 많은 단어 확인
        
        # 토픽 레이블링 개선
        topic_words = [term for term, _ in topic_terms]
        topic_label = "Unknown"
        max_score = 0
        
        for category, phrases in FINANCIAL_PHRASES.items():
            # 각 카테고리와의 매칭 점수 계산
            score = 0
            matched_terms = set()
            
            for word in topic_words:
                for phrase in phrases:
                    # 완전 일치 또는 부분 일치 확인
                    if phrase in word or word in phrase:
                        score += 1
                        matched_terms.add(word)
            
            # 매칭된 고유 단어 수로 정규화
            score = score / len(phrases) if len(phrases) > 0 else 0
            
            if score > max_score and score > 0.1:  # 최소 임계값 설정
                max_score = score
                topic_label = category
                
        # 토픽 정보 저장
        topics.append({
            'label': topic_label,
            'terms': topic_terms,
            'coherence': lda_model.get_topic_terms(idx, topn=1)[0][1],
            'matched_score': max_score
        })
    
    return topics

def analyze_topic_trends(texts_df):
    """시간에 따른 토픽 트렌드 분석"""
    topics_over_time = []
    
    for _, row in texts_df.iterrows():
        text_topics = extract_topics([row['text']])
        for topic in text_topics:
            topics_over_time.append({
                'date': row['date'],
                'topic': topic['label'],
                'coherence': topic['coherence'],
                'terms': topic['terms']
            })
    
    return pd.DataFrame(topics_over_time) 