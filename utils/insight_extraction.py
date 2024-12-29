import spacy
import re
import json
from textblob import TextBlob
import os
from pathlib import Path

def extract_complex_insights(text):
    """복합 키워드와 인사이트 추출"""
    insights = {
        "nodes": [],
        "edges": []
    }
    
    print(f"Processing text of length: {len(text)}")
    
    patterns = {
        'revenue': r'\$?\d+(?:\.\d+)?\s*(?:billion|million|B|M)?\s*(?:in|of)?\s*(?:revenue|income|earnings)',
        'growth': r'(?:up|down|increased|decreased|grew|declined)(?:\s+by)?\s+\d+(?:\.\d+)?%',
        'segment': r'(?:CIB|CCB|AWM|CB)(?:\s+[^.]*?)\$?\d+(?:\.\d+)?\s*(?:billion|million|B|M)',
        'credit': r'credit\s+(?:quality|metrics|performance|losses?)(?:[^.]*?)(?:strong|stable|improved|deteriorated)',
        'cost': r'(?:cost|expense)s?\s+(?:[^.]*?)\$?\d+(?:\.\d+)?\s*(?:billion|million|B|M)',
        'investment': r'invest(?:ing|ment|ed)?\s+(?:[^.]*?)\$?\d+(?:\.\d+)?\s*(?:billion|million|B|M)',
        'market': r'market\s+(?:share|position|leadership)(?:[^.]*?)(?:increased|decreased|improved|leading|strong)',
        'guidance': r'(?:expect|forecast|guidance|outlook)(?:[^.]*?)\d+(?:\.\d+)?%'
    }

    # 인사이트 빈도 추적을 위한 딕셔너리
    insight_frequencies = {}
    
    # 디버깅을 위해 원본 텍스트 일부 출력
    print("\nSample text:")
    print(text[:500])
    
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    print(f"\nFound {len(sentences)} sentences")
    
    node_id = 0
    for sentence in sentences:
        found_insights = []
        
        for category, pattern in patterns.items():
            matches = list(re.finditer(pattern, sentence, re.IGNORECASE))
            if matches:
                print(f"\nFound {category} in sentence: {sentence[:100]}...")
                for match in matches:
                    insight_text = match.group(0)
                    
                    # 더도 계산
                    if insight_text not in insight_frequencies:
                        # 전체 텍스트에서 해당 인사이트가 등장하는 횟수 계산
                        insight_frequencies[insight_text] = len(re.findall(re.escape(insight_text), text, re.IGNORECASE))
                    
                    # 더 넓은 컨텍스트 캡처
                    context_start = max(0, match.start() - 100)
                    context_end = min(len(sentence), match.end() + 100)
                    context = sentence[context_start:context_end].strip()
                    
                    print(f"Extracted: {insight_text} (frequency: {insight_frequencies[insight_text]})")
                    
                    node = {
                        "id": f"node_{node_id}",
                        "text": insight_text,
                        "category": category,
                        "context": context,
                        "sentiment": TextBlob(context).sentiment.polarity,
                        "frequency": insight_frequencies[insight_text]  # 빈도 추가
                    }
                    insights["nodes"].append(node)
                    found_insights.append(node)
                    node_id += 1
        
        # 관계 설정
        for i, node1 in enumerate(found_insights):
            for node2 in found_insights[i+1:]:
                # 두 노드가 같은 문장에 있을 때의 관계 강도 계산
                weight = 1.0
                
                # 카테고리가 관련있는 경우 가중치 증가
                related_categories = {
                    ('revenue', 'growth'): 1.5,
                    ('segment', 'revenue'): 1.3,
                    ('credit', 'segment'): 1.2,
                    ('cost', 'revenue'): 1.4,
                    ('investment', 'growth'): 1.5,
                    ('market', 'segment'): 1.3,
                    ('guidance', 'growth'): 1.4
                }
                
                cat_pair = (node1['category'], node2['category'])
                rev_pair = (node2['category'], node1['category'])
                
                if cat_pair in related_categories:
                    weight = related_categories[cat_pair]
                elif rev_pair in related_categories:
                    weight = related_categories[rev_pair]
                
                # 감성이 비슷한 경우 가중치 증가
                sentiment_diff = abs(node1['sentiment'] - node2['sentiment'])
                if sentiment_diff < 0.3:
                    weight *= 1.2
                
                insights["edges"].append({
                    "source": node1["id"],
                    "target": node2["id"],
                    "context": sentence,
                    "weight": weight  # 가중치 추가
                })
    
    print(f"\nFound {len(insights['nodes'])} insights and {len(insights['edges'])} relationships")
    return insights

def setup_directories():
    """필요한 디렉토리 구조 생성"""
    # 프로젝트 루트 디렉토리
    root_dir = Path(__file__).parent.parent
    
    # 필요한 디렉토리들
    directories = [
        'data',
        'data/raw',
        'data/processed'
    ]
    
    # 디렉토리 생성
    for dir_path in directories:
        full_path = root_dir / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {full_path}")

def main():
    # 디렉토리 설정
    setup_directories()
    
    # 어닝콜 텍스트 로드
    try:
        with open('data/raw/JPM_2024_Q3.txt', 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 인사이트 추출
        insights = extract_complex_insights(text)
        
        # JSON 파일로 저장
        output_path = Path('data/processed/insights_network.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(insights, f, indent=2, ensure_ascii=False)
        print(f"Successfully created: {output_path}")
        
    except FileNotFoundError:
        print("Error: Input file not found. Please ensure JPM_2024_Q3.txt exists in data/raw/")
    except Exception as e:
        print(f"Error during processing: {str(e)}")

FINANCIAL_PHRASES = {
    'Revenue & Growth': [
        'revenue', 'growth', 'sales', 'income', 'earnings', 'profit', 'margin',
        'net revenue', 'revenue growth', 'sales growth', 'organic growth',
        'market share', 'top line', 'bottom line', 'profitability'
    ],
    'Market & Trading': [
        'market', 'trading', 'volatility', 'spread', 'fee', 'commission',
        'fixed income', 'equity trading', 'market making', 'securities',
        'derivatives', 'fx', 'foreign exchange', 'treasury'
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
    'Digital & Technology': [
        'digital', 'technology', 'platform', 'mobile', 'online', 'payment',
        'digital banking', 'mobile app', 'digital platform', 'innovation',
        'fintech', 'cyber', 'cloud', 'automation', 'ai'
    ]
}

if __name__ == "__main__":
    main() 