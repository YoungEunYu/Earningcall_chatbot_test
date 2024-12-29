from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import pandas as pd
import json
from pathlib import Path
import numpy as np
from utils.topic_preprocessing import preprocess_text

class FinancialTopicAnalyzer:
    def __init__(self, num_topics=5):
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        # 금융 도메인 특화 불용어 추가
        self.stopwords = [
            'the', 'and', 'of', 'that', 'to', 'in', 'it', 'you', 're', 'we',
            'thanks', 'thank', 'please', 'operator', 'proceed', 'line', 'open',
            'question', 'morning', 'yeah', 'good', 'great', 'okay', 'right',
            'chase', 'jpmorgan', 'barnum', 'dimon', 'chairman', 'officer'
        ]
        
        # BERTopic 기본 설정
        self.topic_model = BERTopic(
            embedding_model=self.sentence_model,
            nr_topics=num_topics,
            min_topic_size=5,
            verbose=True,
            top_n_words=10
        )
        self.num_topics = num_topics
    
    def get_seed_topics(self):
        """금융 도메인 토픽 시드"""
        return [
            ['revenue', 'growth', 'earnings', 'profit', 'income'],
            ['credit', 'loan', 'deposit', 'banking', 'asset'],
            ['market', 'trading', 'investment', 'capital', 'risk'],
            ['cost', 'expense', 'efficiency', 'margin', 'savings'],
            ['guidance', 'outlook', 'forecast', 'target', 'strategy']
        ]
    
    def process_earnings_call(self, text, quarter):
        """분기별 어닝콜 텍스트 처리"""
        # 문장 분리 및 전처리
        sentences = []
        for paragraph in text.split('\n'):
            # 발화자 정보 제거
            if any(skip in paragraph for skip in ['JPMorgan', 'Officer', 'Operator']):
                continue
            
            # 문장 분리 및 전처리
            for sentence in paragraph.split('.'):
                sentence = sentence.strip()
                if len(sentence) > 20:  # 너무 짧은 문장 제외
                    # 불용어 제거
                    words = [word for word in sentence.split() if word.lower() not in self.stopwords]
                    if words:  # 의미있는 단어가 남아있는 경우만
                        sentences.append(' '.join(words))
        
        # 토픽 추출
        topics, probs = self.topic_model.fit_transform(sentences)
        
        # 토픽 정보 추출
        topic_info = self.topic_model.get_topic_info()
        
        # Top 5 토픽 추출
        top_topics = []
        for idx, row in topic_info.iterrows():
            if idx < self.num_topics:  # -1은 아웃라이어 토픽
                topic_words = self.topic_model.get_topic(row.Topic)
                top_topics.append({
                    'id': row.Topic,
                    'name': f"Topic {row.Topic}",
                    'count': row.Count,
                    'words': [word for word, score in topic_words],
                    'coherence': np.mean([score for word, score in topic_words]),
                    'representative_docs': self.topic_model.get_representative_docs(row.Topic)
                })
        
        return {
            'quarter': quarter,
            'topics': top_topics,
            'topic_distribution': self.get_topic_distribution(topics)
        }
    
    def get_topic_distribution(self, topics):
        """토픽 분포 계산"""
        topic_counts = pd.Series(topics).value_counts()
        total = len(topics)
        return {str(topic): count/total for topic, count in topic_counts.items()}

def analyze_all_quarters():
    """모든 분기 데이터 분석"""
    analyzer = FinancialTopicAnalyzer()
    # 파일명 형식 수정
    quarters = ['2024_Q2', '2024_Q3']  # 현재 가진 데이터만 처리
    results = {}
    
    for quarter in quarters:
        try:
            with open(f'data/raw/JPM_{quarter}.txt', 'r', encoding='utf-8') as f:
                text = f.read()
            results[quarter] = analyzer.process_earnings_call(text, quarter)
        except Exception as e:
            print(f"Error processing {quarter}: {str(e)}")
    
    # 결과 저장
    output_path = Path('data/processed/topic_analysis.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return results

if __name__ == "__main__":
    analyze_all_quarters() 