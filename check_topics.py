import pandas as pd
from utils.topic_preprocessing import load_earnings_calls, analyze_topic_trends, extract_topics
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
import pickle
import os


def load_or_create_cache(cache_path, create_func, *args):
    """캐시된 데이터 로드 또는 생성"""
    if os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    print(f"Creating new data and caching to {cache_path}")
    data = create_func(*args)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)
    return data

def check_topic_relevance(topic_label, quotes, model):
    """토픽과 인용구 간의 관련성 점수 계산"""
    topic_desc = f"{topic_label.lower()} related content"
    
    # 임베딩 생성 (배치 처리)
    topic_embedding = model.encode([topic_desc])[0]
    
    # 배치 크기 조정
    batch_size = 8  # 더 작은 배치 크기
    all_similarities = []
    
    for i in range(0, len(quotes), batch_size):
        batch = quotes[i:min(i+batch_size, len(quotes))]
        quote_embeddings = model.encode(batch, show_progress_bar=False)
        similarities = cosine_similarity([topic_embedding], quote_embeddings)[0]
        all_similarities.extend(similarities)
    
    return list(zip(quotes, all_similarities))

def main():
    # 1. 데이터 로드 (캐시 활용)
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    print("Loading and processing data...")
    texts_df = load_or_create_cache(
        f"{cache_dir}/texts_df.pkl",
        load_earnings_calls,
        'data/raw'
    )
    
    # 2. 문장 임베딩 모델 로드
    print("Loading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 3. 토픽 트렌드 분석 (캐시 활용)
    print("Analyzing topic trends...")
    topic_trends = load_or_create_cache(
        f"{cache_dir}/topic_trends.pkl",
        analyze_topic_trends,
        texts_df
    )
    
    # 4. 토픽별 관련 구절 분석
    print("\nAnalyzing topic relevance...")
    for topic in topic_trends['topic'].unique():
        print(f"\n{'='*50}")
        print(f"Topic: {topic}")
        
        topic_data = topic_trends[topic_trends['topic'] == topic]
        for _, row in topic_data.iterrows():
            try:
                quotes = row['quotes']
                if quotes:
                    scored_quotes = check_topic_relevance(topic, quotes, model)
                    
                    # 유사도 점수로 정렬
                    scored_quotes.sort(key=lambda x: x[1], reverse=True)
                    
                    print(f"\nDate: {row['date']}")
                    print("Top relevant quotes:")
                    for quote, score in scored_quotes:
                        if score > 0.3:  # 낮은 유사도 점수는 제외
                            print(f"\nScore: {score:.3f}")
                            print(f"Quote: {quote}")
                    
            except Exception as e:
                print(f"Error: {e}")
        print('='*50)

if __name__ == "__main__":
    main() 