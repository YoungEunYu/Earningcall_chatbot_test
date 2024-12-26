import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from bertopic import BERTopic
import spacy
import re
from collections import Counter
import os
import ssl
from os import environ
from openai import OpenAI
from nltk.stem import WordNetLemmatizer

# SSL 인증서 문제 해결
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# NLTK 데이터 다운로드
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

# 금융 도메인 특화 불용어
FINANCE_STOPWORDS = set([
    'quarter', 'year', 'thank', 'please', 'company', 'business',
    'million', 'billion', 'call', 'earnings', 'conference',
    'forward-looking', 'statements', 'operator', 'good', 'morning',
    'thanks', 'question'
])

# 금융 도메인 사전 정의
FINANCIAL_TERMS = {
    # 재무제표 관련
    'income': ['revenue', 'sales', 'earnings', 'profit', 'margin', 'ebitda', 'eps'],
    'balance': ['assets', 'liabilities', 'equity', 'debt', 'capital', 'loans', 'deposits'],
    'cash_flow': ['cash', 'operating', 'investing', 'financing', 'dividend', 'capex'],
    
    # 비즈니스 성과
    'performance': ['growth', 'increase', 'decrease', 'improvement', 'decline', 'expansion'],
    'metrics': ['roi', 'roa', 'roe', 'cost', 'efficiency', 'ratio'],
    
    # 시장/산업
    'market': ['market', 'industry', 'sector', 'competition', 'demand', 'supply', 'trend'],
    'risk': ['risk', 'volatility', 'exposure', 'compliance', 'regulatory', 'credit'],
    
    # 전략
    'strategy': ['strategy', 'initiative', 'investment', 'acquisition', 'merger', 'partnership'],
    'innovation': ['digital', 'technology', 'innovation', 'platform', 'transformation']
}

def clean_text(text):
    """기본 텍스트 클리닝"""
    # 소문자 변환
    text = text.lower()
    
    # 특수문자 제거 (달러 기호 $ 유지)
    text = re.sub(r'[^a-zA-Z0-9\s$.]', '', text)
    
    # 불필요한 공백 제거
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_financial_phrases(text):
    """금융 관련 주요 구절 추출"""
    # 문장 토큰화
    sentences = sent_tokenize(text)
    
    # 금융 관련 핵심 구절 패턴
    financial_patterns = [
        r'(net income|revenue growth|operating margin)',
        r'(interest rate|credit quality|loan growth)',
        r'(capital ratio|asset quality|market share)',
        r'(cost reduction|expense management|efficiency ratio)',
        r'(digital banking|payment volume|customer acquisition)',
        r'(investment banking|trading revenue|wealth management)',
        r'(credit card|deposit growth|commercial banking)'
    ]
    
    # 구절 추출
    phrases = []
    for sentence in sentences:
        for pattern in financial_patterns:
            matches = re.finditer(pattern, sentence.lower())
            for match in matches:
                phrase = match.group()
                # 문맥 추가 (앞뒤 단어 포함)
                start = max(0, match.start() - 20)
                end = min(len(sentence), match.end() + 20)
                context = sentence[start:end].strip()
                phrases.append({
                    'phrase': phrase,
                    'context': context
                })
    
    return phrases

def analyze_earnings_structure(text):
    """어닝콜 구조 분석"""
    # Q&A 섹션 구분
    qa_pattern = r"Question-and-Answer Session"
    sections = re.split(qa_pattern, text)
    
    # 발화자별 텍스트 구분
    speaker_pattern = r'\b[A-Z][a-zA-Z\s]+:'
    speakers = re.findall(speaker_pattern, text)
    
    structure = {
        'presentation': sections[0] if len(sections) > 0 else "",
        'qa': sections[1] if len(sections) > 1 else "",
        'speakers': list(set(speakers))
    }
    
    return structure

def analyze_sentiment(text):
    """감성 분석"""
    from nltk.sentiment import SentimentIntensityAnalyzer
    nltk.download('vader_lexicon')
    
    sia = SentimentIntensityAnalyzer()
    sentences = sent_tokenize(text)
    
    sentiments = []
    for sentence in sentences:
        score = sia.polarity_scores(sentence)
        sentiments.append({
            'text': sentence,
            'sentiment': score['compound'],
            'positive': score['pos'],
            'negative': score['neg'],
            'neutral': score['neu']
        })
    
    return pd.DataFrame(sentiments)

def summarize_context(context, client):
    """GPT를 사용하여 문맥 요약"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a financial analyst. Summarize the given text in one concise sentence, focusing on key financial insights."},
                {"role": "user", "content": f"Summarize this earnings call context: {context}"}
            ],
            max_tokens=100,
            temperature=0.5
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"GPT 요약 중 오류 발생: {e}")
        return context[:100] + "..."  # 오류 시 원문 앞부분만 반환

def create_topic_model(text, openai_client):
    """개선된 토픽 모델링"""
    print("토픽 모델링 시작...")
    sentences = sent_tokenize(text)
    
    # 금융 관련 키워드 정의
    financial_keywords = {
        'revenue', 'profit', 'margin', 'growth', 'cost',
        'investment', 'capital', 'asset', 'liability', 'equity',
        'market', 'risk', 'strategy', 'performance', 'outlook',
        'credit', 'loan', 'deposit', 'interest', 'rate',
        'banking', 'trading', 'fee', 'income', 'expense',
        'dividend', 'share', 'earnings', 'quarter', 'guidance',
        'debt', 'cash', 'flow', 'balance', 'sheet',
        'mortgage', 'wealth', 'management', 'treasury', 'securities',
        'volatility', 'portfolio', 'client', 'transaction', 'volume'
    }
    
    # BERTopic 모델 설정
    topic_model = BERTopic(
        language="english",
        min_topic_size=5,
        nr_topics=8,
        verbose=True,
        n_gram_range=(1, 3)
    )
    
    topics, probs = topic_model.fit_transform(sentences)
    
    # 토픽 정보 가져오기
    topic_info = topic_model.get_topic_info()
    print("\n토픽 정보 컬럼:", topic_info.columns.tolist())  # 디버깅용
    
    # 컬럼명 수정
    topic_info = topic_info.rename(columns={
        'Topic': 'Topic_Num',  # 'Topic'이 실제 컬럼명
        'Count': 'Size'
    })
    
    topic_labels = []
    topic_phrases = []
    topic_contexts = []
    
    # 토픽 순회 (Topic_Num 대신 Topic 사용)
    for topic in topic_info['Topic_Num']:  # 이제 rename 했으므로 Topic_Num 사용 가능
        if topic != -1:
            topic_sentences = [sent for t, sent in zip(topics, sentences) if t == topic]
            if topic_sentences:
                context = max(topic_sentences, key=len)
                summary = summarize_context(context, openai_client)
                
                # 토픽 관련 단어/구절 추출 개선
                topic_words = topic_model.get_topic(topic)
                financial_terms = []
                
                # 1. 먼저 복합어 검사
                for word, score in topic_words:
                    words = word.split()
                    if len(words) > 1 and any(w.lower() in financial_keywords for w in words):
                        financial_terms.append(word)
                
                # 2. 단일 단어도 포함
                if len(financial_terms) < 3:
                    for word, score in topic_words:
                        if word.lower() in financial_keywords and word not in financial_terms:
                            financial_terms.append(word)
                        if len(financial_terms) >= 3:
                            break
                
                if financial_terms:
                    topic_label = " & ".join(financial_terms[:2])
                    topic_phrases.append(' | '.join(financial_terms))  # 모든 용어 포함
                else:
                    # GPT 레이블 생성
                    label_prompt = f"Based on this context, give me 3 key financial terms or phrases that best describe the topic (separated by ' | '): {summary}"
                    response = openai_client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a financial analyst. Extract key financial terms from the given context."},
                            {"role": "user", "content": label_prompt}
                        ],
                        max_tokens=50,
                        temperature=0.3
                    )
                    generated_terms = response.choices[0].message.content
                    topic_label = " & ".join(generated_terms.split(' | ')[:2])
                    topic_phrases.append(generated_terms)
                
                topic_labels.append(topic_label)
                topic_contexts.append(summary)
            else:
                topic_labels.append(f'Topic {topic}')
                topic_phrases.append('')
                topic_contexts.append('')
        else:
            topic_labels.append('Other')
            topic_phrases.append('')
            topic_contexts.append('')
    
    topic_info['Topic_Label'] = topic_labels
    topic_info['Top_Phrases'] = topic_phrases
    topic_info['Context'] = topic_contexts
    
    print("\n최종 토픽 정보:")
    print(topic_info)
    
    return topic_info

def weight_financial_terms(tokens, financial_terms=FINANCIAL_TERMS):
    """금융 용어에 가중치 부여"""
    weighted_tokens = []
    for token in tokens:
        # 금융 용어 사전에서 용어 확인
        for category, terms in financial_terms.items():
            if token in terms:
                # 중요 용어는 반복해서 추가하여 가중치 부여
                weight = 2  # 가중치 조정 가능
                weighted_tokens.extend([token] * weight)
                break
        else:
            weighted_tokens.append(token)
    return weighted_tokens

def preprocess_text(text, additional_stopwords=['JPMorgan', 'Chase', 'jamie', 'dimon', 'chief', 
                                              'executive', 'officer', 'chairman']):
    # 1. 기본 클리닝
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  # 여러 공백을 하나로
    text = re.sub(r'[^\w\s]', ' ', text)  # 특수문자 제거
    
    # 2. 불용어 설정
    stop_words = set(stopwords.words('english'))
    stop_words.update(additional_stopwords)
    
    # 3. 토큰화 및 품사 태깅
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    tagged = pos_tag(tokens)
    
    # 4. 레마타이제이션
    lemmatizer = WordNetLemmatizer()
    
    # 5. 금융 관련 중요 품사와 단어만 선택적으로 유지
    processed_tokens = []
    for token, tag in tagged:
        # 숫자와 한 글자 단어 제외
        if len(token) < 2 or token.isdigit():
            continue
            
        # 불용어 제외
        if token in stop_words:
            continue
            
        # 명사, 동사, 형용사만 선택적으로 포함
        if tag.startswith(('NN', 'VB', 'JJ')):  # 명사, 동사, 형용사
            # 동사는 'increase', 'decrease', 'grow' 등 중요한 것만 포함
            if tag.startswith('VB'):
                important_verbs = {'increase', 'decrease', 'grow', 'decline', 'rise', 'fall',
                                 'expand', 'reduce', 'improve', 'strengthen', 'invest'}
                lemma = lemmatizer.lemmatize(token, 'v')
                if lemma in important_verbs:
                    processed_tokens.append(lemma)
            else:
                # 명사와 형용사는 레마타이제이션 후 포함
                lemma = lemmatizer.lemmatize(token)
                processed_tokens.append(lemma)
    
    # 6. 금융 용어 가중치 부여
    weighted_tokens = weight_financial_terms(processed_tokens)
    
    return ' '.join(weighted_tokens)

def process_financial_text(text):
    """금융 특화 전처리"""
    
    # 1. 금액 표현 처리 ($55.55 million 등)
    def convert_amount(match):
        amount = float(match.group(1).replace(',', ''))
        unit = match.group(2).lower()
        multiplier = {
            'million': 1e6,
            'billion': 1e9,
            'trillion': 1e12
        }
        # 원래 표현 유지하면서 표준화된 값 추가
        return f"{match.group(0)} [${amount * multiplier[unit]:.0f}]"

    # 금액 패턴 (소수점과 콤마 고려)
    text = re.sub(r'\$?([\d,]+\.?\d*)\s*(million|billion|trillion)', 
                 convert_amount,
                 text,
                 flags=re.IGNORECASE)
    
    # 2. 퍼센트 표현 처리 (15.5% 등)
    def convert_percent(match):
        percent = float(match.group(1))
        # 원래 표현 유지
        return f"{match.group(0)}"

    text = re.sub(r'([\d,]+\.?\d*)\s*(%|percent)', 
                 convert_percent,
                 text,
                 flags=re.IGNORECASE)
    
    # 3. 분기 표현 표준화
    quarter_map = {'first': '1', 'second': '2', 'third': '3', 'fourth': '4'}
    text = re.sub(r'(first|second|third|fourth)\s+quarter', 
                 lambda m: f"Q{quarter_map[m.group(1).lower()]}", 
                 text,
                 flags=re.IGNORECASE)
    
    return text

def extract_financial_metrics(text):
    """금융 지표 추출 - 의미있는 금액만 추출"""
    metrics = {
        'revenue': [],
        'profit': [],
        'growth': [],
        'margin': [],
        'assets': [],
        'deposits': [],
        'loans': []
    }
    
    # 지표별 정규표현식 패턴 (컨텍스트 고려)
    patterns = {
        'revenue': r'(revenue|sales|income).{0,30}?\$?([\d,]+\.?\d*)\s*(million|billion|trillion)',
        'profit': r'(net income|profit|earnings).{0,30}?\$?([\d,]+\.?\d*)\s*(million|billion|trillion)',
        'growth': r'(growth|increase|decrease).{0,30}?([\d,]+\.?\d*)\s*(%|percent)',
        'margin': r'(margin).{0,30}?([\d,]+\.?\d*)\s*(%|percent)',
        'assets': r'(assets|AUM).{0,30}?\$?([\d,]+\.?\d*)\s*(million|billion|trillion)',
        'deposits': r'(deposits).{0,30}?\$?([\d,]+\.?\d*)\s*(million|billion|trillion)',
        'loans': r'(loans|lending).{0,30}?\$?([\d,]+\.?\d*)\s*(million|billion|trillion)'
    }
    
    for metric, pattern in patterns.items():
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            # 컨텍스트와 함께 값을 저장
            context = match.group(0)
            amount = float(match.group(2).replace(',', ''))
            unit = match.group(3).lower()
            multiplier = {'million': 1e6, 'billion': 1e9, 'trillion': 1e12}
            standardized_amount = amount * multiplier[unit]
            
            metrics[metric].append({
                'context': context,
                'original_value': f"${amount} {unit}",
                'standardized_value': standardized_amount
            })
    
    return metrics

def preprocess_for_wordcloud(text):
    """워드클라우드 전용 전처리"""
    # 워드클라우드 특화 불용어
    wordcloud_stopwords = {
        # 회사명 관련
        'jpmorgan', 'chase', 'jp', 'morgan', 'jpm',
        
        # 일반적인 비즈니스 용어
        'think', 'market', 'going', 'look', 'see', 'way',
        'know', 'mean', 'well', 'really', 'actually', 'obviously',
        'kind', 'sort', 'bit', 'thing', 'looking', 'said', 'saying',
        'thats', 'lets', 'youre', 'weve', 'theyre', 'dont',
        
        # 어닝콜 관련
        'quarter', 'year', 'thank', 'thanks', 'please', 'question',
        'operator', 'conference', 'call', 'good', 'morning', 'afternoon',
        
        # 숫자 관련
        'million', 'billion', 'percent', 'percentage', 'number',
        'first', 'second', 'third', 'fourth', 'next'
    }
    
    # 1. 문자 변환
    text = text.lower()
    
    # 2. 토큰화
    words = word_tokenize(text)
    
    # 3. 불용어 및 특수문자 제거
    words = [word for word in words 
            if word.isalnum() and  # 알파벳과 숫자만
            word not in wordcloud_stopwords and  # 커스텀 불용어 제거
            word not in stopwords.words('english') and  # 일반 불용어 제거
            len(word) > 2]  # 너무 짧은 단어 제거
    
    # 4. 단어 빈도수 계산
    word_freq = Counter(words)
    
    # 5. 의미있는 단어만 선택 (빈도수 기준)
    min_freq = 3  # 최소 등장 횟수
    filtered_words = [word for word, freq in word_freq.items() if freq >= min_freq]
    
    return ' '.join(filtered_words)

def extract_topics(texts, num_topics=5):
    """금융 도메인 특화 토픽 모델링"""
    from gensim import corpora, models
    
    # 텍스트 전처리
    processed_texts = [preprocess_text(text) for text in texts]
    
    # 문서-단어 행렬 생성
    texts_tokens = [text.split() for text in processed_texts]
    dictionary = corpora.Dictionary(texts_tokens)
    corpus = [dictionary.doc2bow(text) for text in texts_tokens]
    
    # 금융 도���인 사전 기반 시드 토픽 초기화
    seed_topics = {}
    for category, terms in FINANCIAL_TERMS.items():
        seed_terms = [term for term in terms if term in dictionary.token2id]
        if seed_terms:
            topic_ids = [dictionary.token2id[term] for term in seed_terms]
            seed_topics[category] = topic_ids
    
    # LDA 모델 학습 (시드 토픽 활용)
    lda_model = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=20,
        alpha='auto',
        eta='auto'
    )
    
    # 토픽 추출 및 금융 도메인 레이블링
    topics = []
    for topic_id in range(num_topics):
        topic_terms = lda_model.show_topic(topic_id, topn=10)
        
        # 토픽 레이블링
        topic_label = "Unknown"
        for category, terms in FINANCIAL_TERMS.items():
            category_terms = set(terms)
            topic_terms_set = set([term for term, _ in topic_terms])
            if len(category_terms & topic_terms_set) >= 2:  # 최소 2개 이상 일치
                topic_label = category.replace('_', ' ').title()
                break
        
        topics.append({
            'label': topic_label,
            'terms': topic_terms
        })
    
    return topics

def main():
    # 데이터 폴더 생성
    for dir_path in ['data/raw', 'data/processed']:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    # 원본 텍스트 파일 읽기
    with open('data/raw/JPM_2024_Q3.txt', 'r', encoding='utf-8') as file:
        text = file.read()
    
    # 텍스트 전처리 (GPT 클라이언트 전달)
    processed_data = preprocess_text(text)
    
    # 처리된 텍스트 저장
    with open('data/processed/JPM_2024_Q3_processed.txt', 'w', encoding='utf-8') as file:
        file.write(processed_data['cleaned_text'])
    
    # 금융 구절 분석 결과 저장 (수정된 부분)
    phrases_df = pd.DataFrame(processed_data['financial_phrases'])
    phrases_df['frequency'] = 1
    phrases_df = phrases_df.groupby('phrase')['frequency'].sum().reset_index()
    phrases_df.to_csv('data/financial_phrases.csv', index=False)
    
    # 감성 분석 결과 저장
    processed_data['sentiment'].to_csv('data/sentiment_analysis.csv', index=False)
    
    # 토픽 정보 저장
    processed_data['topics'].to_csv('data/topic_info.csv', index=False)
    
    # 데이터 저장 시 확인
    print("\n저장된 파일 확인:")
    topic_info = processed_data['topics']
    print("topic_info.csv 저장됨")
    print(topic_info.head())
    
    phrases_df = pd.DataFrame(processed_data['financial_phrases'])
    if not phrases_df.empty:
        phrases_df['frequency'] = 1
        phrases_df = phrases_df.groupby('phrase')['frequency'].sum().reset_index()
        phrases_df.to_csv('data/financial_phrases.csv', index=False)
        print("\nfinancial_phrases.csv 저장됨")
        print(phrases_df.head())
    else:
        print("\n경고: financial_phrases가 비어있습니다!")
    
    print("전처리 완료!")
    
    # 워드클라우드용 전처리 및 저장
    wordcloud_text = preprocess_for_wordcloud(text)
    with open('data/processed/JPM_2024_Q3_wordcloud.txt', 'w', encoding='utf-8') as file:
        file.write(wordcloud_text)

if __name__ == "__main__":
    main() 