import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from openai import OpenAI
import os
import plotly.express as px
import sys
from pathlib import Path
from textblob import TextBlob
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
from utils.topic_preprocessing import load_earnings_calls, analyze_topic_trends, extract_topics
from utils.insight_extraction import extract_complex_insights, FINANCIAL_PHRASES
import re
import altair as alt

# 현로젝트 루트 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# temporal_analysis.py에서 직접 함수 import
try:
    from utils.temporal_analysis import extract_temporal_data, identify_key_events
except ImportError as e:
    print(f"Import Error: {e}")
    print(f"Current directory: {current_dir}")
    print(f"Python path: {sys.path}")

# OpenAI 클라이언트 설정
client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

# GPT 호출 활성화/비활성화 플래그
USE_GPT = True  # GPT 호출 비활성화

def get_chatgpt_response(prompt, context):
    """GPT를 사용하여 응답 생성"""
    if not USE_GPT:
        return "GPT call has been disabled."

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a financial analyst assistant."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {prompt}"}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"

def create_topic_network(topic_info):
    """토픽-문구 네트워크 생성"""
    G = nx.Graph()
    
    # NaN 체크 추가하여 수정
    for _, row in topic_info.iterrows():
        topic_num = f"Topic {row['Topic_Num']}"
        
        # Top_Phrases가 문자열인 경우에만 처리
        if isinstance(row['Top_Phrases'], str):
            phrases = row['Top_Phrases'].split(' | ')
            
            # 토픽 노드 추가
            G.add_node(topic_num, node_type='topic', size=row['Size'])
            
            # 문구 노드 추가 및 엣지 연결
            for phrase in phrases:
                G.add_node(phrase, node_type='phrase', size=10)
                G.add_edge(topic_num, phrase)
    
    # 노드 위치 산
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # 엣지 트레이스
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # 네드 트레이스
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        node_size.append(G.nodes[node]['size'])
        # 노드 타입에 따른 색상 설정
        if G.nodes[node]['node_type'] == 'topic':
            node_color.append('#0066ff')
        else:
            node_color.append('#00ff88')
    
    # 시각화 생성
    fig = go.Figure()
    
    # 엣지 추가
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#666'),
        hoverinfo='none',
        mode='lines'
    ))
    
    # 노드 추가
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="bottom center",
        marker=dict(
            size=node_size,
            color=node_color,
            line_width=2
        )
    ))
    
    fig.update_layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0,l=0,r=0,t=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='#2d2d2d',
        paper_bgcolor='#2d2d2d'
    )
    
    return fig

def create_sentiment_viz():
    """감성 분석 시각화 생성"""
    # 데이터 로드
    sentiment_df = pd.read_csv('data/sentiment_analysis.csv')
    keywords_df = pd.read_csv('data/keyword_analysis.csv')
    
    # 1. 감성 분포 파이 차트
    sentiment_counts = {
        'Positive': len(sentiment_df[sentiment_df['sentiment'] > 0.2]),
        'Negative': len(sentiment_df[sentiment_df['sentiment'] < -0.2]),
        'Neutral': len(sentiment_df[(sentiment_df['sentiment'] >= -0.2) & 
                                  (sentiment_df['sentiment'] <= 0.2)])
    }
    
    pie_fig = go.Figure(data=[go.Pie(
        labels=list(sentiment_counts.keys()),
        values=list(sentiment_counts.values()),
        hole=.3,
        marker_colors=['#00ff88', '#ff4444', '#aaaaaa']
    )])
    
    pie_fig.update_layout(
        title="Sentiment Distribution",
        plot_bgcolor='#2d2d2d',
        paper_bgcolor='#2d2d2d',
        font=dict(color='white')
    )
    
    # 2. 키워드 바 차트
    keywords_fig = go.Figure()
    
    # 긍정 키워드 (상위 10개)
    pos_keywords = keywords_df[keywords_df['sentiment'] == 'positive'].nlargest(10, 'count')
    keywords_fig.add_trace(go.Bar(
        name='Positive',
        x=pos_keywords['keyword'],
        y=pos_keywords['count'],
        marker_color='#00ff88'
    ))
    
    # 부정 키워드 (상위 10개)
    neg_keywords = keywords_df[keywords_df['sentiment'] == 'negative'].nlargest(10, 'count')
    keywords_fig.add_trace(go.Bar(
        name='Negative',
        x=neg_keywords['keyword'],
        y=neg_keywords['count'],
        marker_color='#ff4444'
    ))
    
    keywords_fig.update_layout(
        barmode='group',
        title="Top Keywords by Sentiment",
        plot_bgcolor='#2d2d2d',
        paper_bgcolor='#2d2d2d',
        font=dict(color='white')
    )
    
    return pie_fig, keywords_fig

def create_enhanced_network():
    """향상된 인사이트 네트워크 시각화"""
    
    # JSON 파일에서 데이터 로드
    with open('data/processed/insights_network.json', 'r') as f:
        network_data = json.load(f)
    
    # 노드 색상 매핑
    color_map = {
        'revenue': '#1f77b4',      # 수익
        'growth': '#2ca02c',       # 성장
        'segment': '#ff7f0e',      # 사업부문
        'credit': '#d62728',       # 신용
        'cost': '#9467bd',         # 비용
        'investment': '#8c564b',   # 투자
        'market': '#e377c2',       # 시장
        'guidance': '#7f7f7f'      # 가이던스
    }
    
    # 노드 트레이스
    node_trace = go.Scatter(
        x=[], y=[], 
        mode='markers+text',
        hoverinfo='text',
        text=[],
        textposition="bottom center",
        marker=dict(
            showscale=True,
            colorscale='YlOrRd',
            reversescale=True,
            color=[],
            size=[],
            line_width=2
        )
    )
    
    # 엣지 트레이스
    edge_trace = go.Scatter(
        x=[], y=[],
        line=dict(width=0.5, color='#888'),
        hoverinfo='text',
        mode='lines',
        text=[]
    )
    
    # 네트워크 레이아웃 계산
    G = nx.Graph()
    
    # 노드 추가
    for node in network_data['nodes']:
        G.add_node(node['id'], 
                  text=node['text'],
                  category=node['category'],
                  sentiment=node['sentiment'],
                  frequency=node['frequency'])
    
    # 엣지 추가 - weight 처리 수정
    for edge in network_data['edges']:
        try:
            weight = edge.get('weight', 1.0)  # weight가 없으면 기본값 1.0 사용
        except:
            weight = 1.0
        G.add_edge(edge['source'], edge['target'], 
                  weight=weight,
                  context=edge['context'])
    
    # 레이아웃 계산
    pos = nx.spring_layout(G)
    
    # 엣지 데이터 추가
    edge_trace = go.Scatter(
        x=[], y=[],
        line=dict(width=0.5, color='#888'),
        hoverinfo='text',
        mode='lines',
        text=[]
    )
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += (x0, x1, None)
        edge_trace['y'] += (y0, y1, None)
        try:
            weight = G.edges[edge]['weight']
            context = G.edges[edge]['context']
            edge_trace['text'] += (f"Weight: {weight:.2f}<br>{context}",)
        except:
            edge_trace['text'] += (G.edges[edge].get('context', ''),)
    
    # 노드 데이터 추가
    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += (x,)
        node_trace['y'] += (y,)
        node_info = G.nodes[node]
        
        # 호버 텍스트
        hover_text = f"""
        {node_info['text']}
        Category: {node_info['category']}
        Sentiment: {node_info['sentiment']}
        Frequency: {node_info['frequency']}
        """
        node_trace['text'] += (hover_text,)
        
        # 노드 크기 (빈도 기반)
        node_trace['marker']['size'] += (20 + node_info['frequency'] * 5,)
        
        # 노드 색상 (테고리 기반)
        node_trace['marker']['color'] += (color_map[node_info['category']],)
    
    # 시각화
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title='Financial Insights Network',
                       titlefont_size=16,
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       plot_bgcolor='#2d2d2d',
                       paper_bgcolor='#2d2d2d',
                       font=dict(color='white'),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                   ))
    
    return fig

def create_temporal_sentiment_viz(transcript_data):
    """어닝콜 감성 분석 시각화"""
    
    temporal_df = extract_temporal_data(transcript_data)
    
    if temporal_df.empty:
        st.warning("No data could be extracted from the transcript.")
        return

    # 시간순 감성 추이 그래프
    fig = go.Figure()
    
    # 전체 어닝콜의 시간순 감성 점수
    fig.add_trace(go.Scatter(
        x=temporal_df.index,
        y=temporal_df['sentiment_score'].rolling(window=5).mean(),
        mode='lines',
        name='Sentiment Trend',
        line=dict(color='lightblue', width=2),
        hovertemplate="Time: %{x}<br>Sentiment: %{y:.2f}<br><extra></extra>"
    ))

    # 클릭 이벤트를 위한 설정
    fig.update_layout(
        title="Earnings Call Sentiment Trend",
        xaxis_title="Time",
        yaxis_title="Sentiment Score",
        plot_bgcolor='#2d2d2d',
        paper_bgcolor='#2d2d2d',
        font=dict(color='white'),
        height=300,
        margin=dict(t=30, l=60, r=30, b=60)
    )

    fig.update_yaxes(range=[-1, 1])

    return fig, temporal_df

def show_transcript_popup(temporal_df, selected_index):
    """선택된 시점의 트랜스크립트를 팝업으로 표시"""
    
    # CSS로 팝업 스타일 정의
    st.markdown("""
        <style>
        .transcript-popup {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 80%;
            height: 80%;
            background: #1e1e1e;
            border-radius: 10px;
            padding: 20px;
            overflow-y: auto;
            z-index: 1000;
        }
        .popup-close {
            position: absolute;
            top: 10px;
            right: 10px;
            cursor: pointer;
            color: white;
            font-size: 24px;
        }
        .highlighted-text {
            background-color: rgba(138, 180, 248, 0.2);
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        </style>
    """, unsafe_allow_html=True)

    # 팝업 내용
    st.markdown(f"""
        <div class="transcript-popup">
            <div class="popup-close" onclick="window.streamlit.setComponentValue('close_popup', true)">×</div>
            <h3>Transcript at {temporal_df.index[selected_index]}</h3>
            <div class="highlighted-text">
                <div style="color: #8ab4f8; margin-bottom: 5px;">
                    Sentiment Score: {temporal_df.iloc[selected_index]['sentiment_score']:.3f}
                </div>
                <div>{temporal_df.iloc[selected_index]['text']}</div>
            </div>
            <div style="margin-top: 20px;">
                <h4>Context:</h4>
                {temporal_df.iloc[max(0, selected_index-2):selected_index]['text'].str.cat(sep='<br><br>')}
                <br><br>
                {temporal_df.iloc[selected_index+1:selected_index+3]['text'].str.cat(sep='<br><br>')}
            </div>
        </div>
    """, unsafe_allow_html=True)

def get_finbert_sentiment(text):
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    positive = predictions[0][0].item()
    negative = predictions[0][1].item()
    neutral = predictions[0][2].item()
    
    # 점수를 -1에서 1 사이로 정규화
    sentiment_score = (positive - negative) / (positive + negative + neutral)
    
    return sentiment_score

def is_key_financial_insight(text):
    """핵심 금융 정보인지 판단"""
    # 금융 관련 키워드
    financial_keywords = {
        # 실적/성과 관련
        'revenue', 'income', 'earnings', 'profit', 'margin', 'growth',
        'billion', 'million', 'percent', 'basis points',
        
        # 사업 영역
        'banking', 'trading', 'loans', 'deposits', 'credit', 'investment',
        'assets', 'capital', 'markets',
        
        # 지표
        'ROE', 'ROTCE', 'CET1', 'NII', 'EPS'
    }
    
    # 불필요한 시작 문구
    skip_phrases = {
        'thank you', 'good morning', 'hello', 'hi everyone',
        'next question', 'operator', 'please go ahead'
    }
    
    text_lower = text.lower()
    
    # 불필요한 문구로 시작하면 제외
    if any(text_lower.startswith(phrase) for phrase in skip_phrases):
        return False
        
    # 금융 키워드를 포함하고 있는지 확인
    return any(keyword in text_lower for keyword in financial_keywords)

def get_financial_context(temporal_df, selected_index):
    """선택된 시점의 금융 관련 문맥 추출"""
    current_text = temporal_df.iloc[selected_index]['text']
    
    if not is_key_financial_insight(current_text):
        # 현재 텍스트가 금융 정보가 아니면 주변 문맥에서 찾기
        context_range = range(max(0, selected_index-5), min(len(temporal_df), selected_index+5))
        for idx in context_range:
            if is_key_financial_insight(temporal_df.iloc[idx]['text']):
                return temporal_df.iloc[idx]['text'], temporal_df.iloc[idx]['sentiment_score']
    
    return current_text, temporal_df.iloc[selected_index]['sentiment_score']

def extract_quarterly_metrics(text):
    """어닝콜 텍스트에서 주요 지표 추출"""
    metrics = {}
    
    # CIB 지표 추출
    if "IB fees were up" in text:
        ib_fees_match = re.search(r"IB fees were up (\d+)%", text)
        if ib_fees_match:
            metrics['CIB'] = int(ib_fees_match.group(1))
    
    # AWM 지표 추출
    if "AUM of $" in text:
        aum_match = re.search(r"AUM of \$(\d+\.\d+) trillion.*?up (\d+)%", text)
        if aum_match:
            metrics['AWM'] = int(aum_match.group(2))
    
    # CCB 지표 추출
    if "CCB reported net income" in text:
        ccb_match = re.search(r"revenue of \$(\d+\.\d+) billion, which was up (\d+)%", text)
        if ccb_match:
            metrics['CCB'] = int(ccb_match.group(2))
    
    return metrics

def extract_hiring_trends(text):
    """어닝콜 텍스트에서 채용 관련 정보 추출"""
    hiring_info = {
        'Private Banking': 0,
        'Technology': 0,
        'Operations': 0
    }
    
    # 채용 관련 텍스트 분석
    if "growth in our private banking advisor teams" in text:
        hiring_info['Private Banking'] += 1
    if "technology and marketing" in text:
        hiring_info['Technology'] += 1
    if "operations" in text.lower():
        hiring_info['Operations'] += 1
        
    return hiring_info

def extract_strategic_focus(text):
    """어닝콜 텍스트에서 전략적 투자 방향 추출"""
    strategic = {
        '🌐 International': '',
        '💻 Digital': '',
        '🏦 Network': ''
    }
    
    # International 전략
    if "AWM" in text and "net inflows" in text:
        inflow_match = re.search(r"net inflows were \$(\d+) billion", text)
        if inflow_match:
            strategic['🌐 International'] = f"Net inflows ${inflow_match.group(1)}B"
            
    # Digital 전략
    if "digital" in text.lower():
        digital_info = re.search(r"digital.*?([\w\s]+(?:up|increased)[\w\s]+\d+%)", text, re.I)
        if digital_info:
            strategic['💻 Digital'] = digital_info.group(1)
            
    # Branch Network
    if "retail deposit" in text.lower():
        strategic['🏦 Network'] = "Leading retail deposit share position"
        
    return strategic

def get_ai_keywords(text, period_type="quarterly"):
    """GPT를 사용하여 핵심 키워드와 가중치 추출"""
    if not USE_GPT:
        return {"example_keyword": 5}  # 기본 키워드 예시
    
    try:
        prompt = f"""Analyze this earnings call transcript and create a word cloud representation.
        Task:
        1. Extract the most significant financial terms and insights
        2. Return only the keywords with their importance weights
        3. Focus on {period_type} performance and trends
        4. Remove all common words, numbers, and company names
        5. Combine related concepts into compound terms (e.g., 'credit_quality', 'market_share')

        Format your response ONLY as:
        keyword1:weight
        keyword2:weight
        (weights from 1-10, higher = more important)

        Transcript: {text[:4000]}...
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a financial analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        keywords = {}
        for line in response.choices[0].message.content.strip().split('\n'):
            if ':' in line:
                word, weight = line.strip().split(':')
                keywords[word.strip()] = int(weight)
        
        return keywords
            
    except Exception as e:
        st.error(f"Error in AI analysis: {str(e)}")
        return {}

def main():
    # 페이지 설정
    st.set_page_config(
        page_title="JPM Earnings Call Analysis",
        page_icon="",
        layout="wide"
    )
    
    try:
        # 1년치 어닝콜 데이터 로드
        from utils.topic_preprocessing import load_earnings_calls, analyze_topic_trends, extract_topics
        earnings_calls = load_earnings_calls()
        
        # 최신 데이터의 토픽 정보
        latest_call = earnings_calls.iloc[-1]
        text_data = latest_call['text']
        
        # 토픽 분석
        topic_trends = analyze_topic_trends(earnings_calls)
        
        # 최신 데이터의 토픽 추출
        topics = extract_topics([text_data])
        
        # 감성 분석
        sentiment_fig, temporal_data = create_temporal_sentiment_viz(text_data)
        
        # 분기별 데이터 추출
        quarters = ['2024_Q3', '2024_Q2', '2024_Q1', '2023_Q4']
        quarterly_data = {
            '2023_Q4': {
                'Net Income': 9.3, 'EPS': 3.04, 'Revenue': 39.9
            },
            '2024_Q1': {
                'Net Income': 13.4, 'EPS': 4.44, 'Revenue': 42.5
            },
            '2024_Q2': {
                'Net Income': 18.1, 'EPS': 6.12, 'Revenue': 51.0
            },
            '2024_Q3': {
                'Net Income': 12.9, 'EPS': 4.37, 'Revenue': 43.3
            }
        }
        
        # 분기별 주요 내용 설정
        quarterly_highlights = {
            '2023_Q4': [
                "Return on Tangible Common Equity (ROTCE): 15%",  # ROTCE: Return on Tangible Common Equity
                "Full-year Net Income: $50 billion"
            ],
            '2024_Q1': [
                "ROTCE: 21%",  # ROTCE: Return on Tangible Common Equity
                "Strong performance in investment banking fees"
            ],
            '2024_Q2': [
                "Commercial & Investment Banking (CIB): saw a 50% increase in IB fees"  # CIB: Commercial & Investment Bank
            ],
            '2024_Q3': [
                "CIB: reported a 31% increase in IB fees"  # CIB: Commercial & Investment Bank
            ]
        }

        # 나머지 코드...
        
    except FileNotFoundError:
        st.error("데이터 파일을 찾을 수 없습니다. data/raw 폴더에 JPM_2024_Q3.txt 파일이 있는지 확인해주세요.")
    except Exception as e:
        st.error(f"오류가 발생했습니다: {str(e)}")
    
    # CSS 스타일 (기존 것 유지)
    st.markdown("""
        <style>
        .stApp {
            background-color: #1e1e1e;
        }
        .main-header {
            font-family: 'Helvetica Neue', sans-serif;
            padding: 1.5rem 0;
            text-align: center;
            background: linear-gradient(270deg, #0033cc 0%, #0066ff 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        h2, h3, .subtitle {
            color: white !important;
            margin-bottom: 1rem;
        }
        .metric-card {
            background: #2d2d2d;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            text-align: center;
            margin-bottom: 1rem;
            border: 1px solid #3d3d3d;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: #0066ff;
        }
        .metric-label {
            color: #ffffff;
            font-size: 0.9rem;
        }
        .insights-container {
            background: #2d2d2d;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            margin: 1rem 0 2rem 0;
            border-left: 5px solid #0066ff;
            color: white;
        }
        .chart-container {
            background: #2d2d2d;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            margin: 1rem 0;
            border: 1px solid #3d3d3d;
        }
        .stMarkdown {
            color: white !important;
        }
        [data-testid="stSidebar"] {
            background-color: #2d2d2d;
            border-right: 1px solid #3d3d3d;
        }
        .analysis-card {
            background: #363636;
            padding: 1.2rem;
            border-radius: 8px;
            border: 1px solid #444;
            height: 100%;
        }
        .analysis-card h4 {
            color: #0066ff;
            margin-top: 0;
            margin-bottom: 1rem;
            font-size: 1.1rem;
            border-bottom: 2px solid #444;
            padding-bottom: 0.5rem;
        }
        .analysis-content {
            font-size: 0.95rem;
            line-height: 1.5;
        }
        .quote {
            color: #aaa;
            font-style: italic;
            margin: 0.5rem 0;
            padding-left: 1rem;
            border-left: 3px solid #0066ff;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # 메인 
    st.markdown("""
        <div class='main-header'>
            <h1 style='margin:0;'>JPMorgan Chase Q3 2024 Earnings Call Snapshot</h1>
            <p style='margin:0.5rem 0 0 0;'>AI-Powered Insights from the Latest Earnings Call</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Financial Analysis 섹션
    if 'financial_analysis' not in st.session_state:
        core_metrics_prompt = """List 3 most critical financial metrics in bullet points:
        • Format: "↑ Revenue $40.7B (+21% YoY)"
        • Use arrows (↑/↓) to indicate direction
        • Keep each point under 30 characters
        • Numbers and percentages only"""
        
        future_outlook_prompt = """Extract 3 key forward-looking guidance points:
        • Format:
          📊 NII: $91B expected for 2024

          💰 Expenses: $92B target

          📈 Credit: 3.4% NCO rate

        • Focus on:
          1. NII forecast
          2. Expense target
          3. Credit outlook
        • Numbers only, no quotes
        • Keep each point under 25 characters"""
        
        strategy_risks_prompt = """List 3 main strategic priorities:
        • Format: "• Digital: +15% user growth"
        • One metric per strategy
        • Max 4 words per point
        • Focus on measurable goals"""
        
        st.session_state.financial_analysis = {
            'core_metrics': get_chatgpt_response(core_metrics_prompt, text_data),
            'future_outlook': get_chatgpt_response(future_outlook_prompt, text_data),
            'strategy_risks': get_chatgpt_response(strategy_risks_prompt, text_data)
        }
    
    core_metrics = st.session_state.financial_analysis['core_metrics'].replace('\n', '<br>')
    future_outlook = st.session_state.financial_analysis['future_outlook'].replace('\n', '<br>')
    strategy_risks = st.session_state.financial_analysis['strategy_risks'].replace('\n', '<br>')
    
    # JPMorgan Chase Q3 2024 Deep Dive 섹션
    st.markdown("""
        <div class='insights-container'>
            <div class='insights-title'>
                📈 JPMorgan Chase Q3 2024 Deep Dive
            </div>
            <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1rem;'>
                <div class='metric-card'>
                    <div class='metric-value'>$40.7B</div>
                    <div class='metric-label'>Revenue</div>
                    <div class='metric-change'>↑ 21% YoY</div>
                </div>
                <div class='metric-card'>
                    <div class='metric-value'>$13.2B</div>
                    <div class='metric-label'>Net Income</div>
                    <div class='metric-change'>↑ 35% YoY</div>
                </div>
                <div class='metric-card'>
                    <div class='metric-value'>16.8%</div>
                    <div class='metric-label'>ROTCE</div>
                    <div class='metric-change'>↑ 2.1pp YoY</div>
                </div>
                <div class='metric-card'>
                    <div class='metric-value'>$4.33</div>
                    <div class='metric-label'>EPS</div>
                    <div class='metric-change'>↑ 38% YoY</div>
                </div>
            </div>
            <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem;'>
                <div class='analysis-card'>
                    <h4>💰 Key Performance Metrics</h4>
                    <div class='analysis-content'>{}</div>
                </div>
                <div class='analysis-card'>
                    <h4>🔮 Management Outlook</h4>
                    <div class='analysis-content'>{}</div>
                </div>
                <div class='analysis-card'>
                    <h4>🎯 Strategic Insights & Risks</h4>
                    <div class='analysis-content'>{}</div>
                </div>
            </div>
        </div>
    """.format(core_metrics, future_outlook, strategy_risks), unsafe_allow_html=True)

    # CSS 스타일 추가
    st.markdown("""
        <style>
        .metric-card {
            background: #363636;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: bold;
            color: #0066ff;
            margin-bottom: 0.3rem;
        }
        .metric-label {
            color: #ffffff;
            font-size: 0.9rem;
            margin-bottom: 0.3rem;
        }
        .metric-change {
            color: #00ff88;
            font-size: 0.8rem;
        }
        .analysis-card {
            background: #363636;
            padding: 1.2rem;
            border-radius: 8px;
        }
        .analysis-card h4 {
            color: white;
            margin-bottom: 0.8rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # 사드바 챗봇
    with st.sidebar:
        st.markdown("""
            <div style='padding: 1rem; background: linear-gradient(180deg, #0033cc 0%, #0066ff 100%); 
                        color: white; border-radius: 10px; margin-bottom: 1rem;'>
                <h3 style='margin:0;'>💬 Earnings Insights Assistant</h3>
            </div>
        """, unsafe_allow_html=True)
        
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        context = f"""
        Topics and key phrases from the earnings calls:
        {topic_trends.to_string()}
        
        Main discussion points from the latest earnings call:
        {text_data[:500]}...
        """
        
        if prompt := st.chat_input("Ask about the earnings call..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                response = get_chatgpt_response(prompt, context)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
        
        with st.expander("💡 Sample Questions", expanded=False):
            st.markdown("""
            Try asking:
            - What are the main financial highlights?
            - How is JPMorgan's investment strategy?
            - What are the key risk factors?
            - Can you explain the revenue trends?
            - What's the outlook for next quarter?
            """)
    
    # 차트 영역
    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("📊 Financial Topics Evolution")
        
        topic_trend_fig = go.Figure()
        
        # 색상 매핑 - 더 다양한 색상으로 구분
        colors = {
            'Revenue & Growth': '#1f77b4',     # 파랑
            'Market & Trading': '#2ca02c',     # 초록
            'Credit & Risk': '#ff7f0e',        # 주황
            'Capital & Investment': '#d62728',  # 빨강
            'Digital & Technology': '#9467bd',  # 보라
            'Client & Service': '#17becf',      # 청록
            'Strategy & Outlook': '#bcbd22',    # 올리브
            'Cost & Efficiency': '#7f7f7f',     # 회색
            'Operational Performance': '#e377c2' # 분홍
        }

        # 토픽별 데이터 통합 및 중요도 계산
        topic_data_aggregated = (topic_trends.groupby(['date', 'topic'])
                               .agg({
                                   'importance': 'mean',  # 중요도 평균
                                   'coherence': 'mean'    # coherence 평균
                               })
                               .reset_index())
        
        # 각 분기의 top 5 토픽들을 연결하는 선 그리기
        for topic in topic_trends['topic'].unique():
            topic_data = topic_data_aggregated[topic_data_aggregated['topic'] == topic]
            topic_color = colors.get(topic, '#7f7f7f')
            
            # 해당 토픽이 각 분기의 top 5에 포함될 때만 데이터 포인트 추가
            x_values = []
            y_values = []
            
            for date in topic_data_aggregated['date'].unique():
                quarter_data = topic_data_aggregated[topic_data_aggregated['date'] == date]
                quarter_top5 = quarter_data.nlargest(5, 'importance')
                
                if topic in quarter_top5['topic'].values:
                    topic_importance = quarter_data[quarter_data['topic'] == topic]['importance'].iloc[0]
                    x_values.append(date)
                    y_values.append(topic_importance)
            
            if x_values:  # 데이터 포인트가 있는 경우만 trace 추가
                topic_trend_fig.add_trace(go.Scatter(
                    x=x_values,
                    y=y_values,
                    name=topic,
                    mode='lines+markers',
                    line=dict(color=topic_color),
                    hovertemplate=
                    "Topic: " + topic + "<br>" +
                    "Importance: %{y:.1%}<br>" +
                    "<extra></extra>",
                    hoverlabel=dict(namelength=-1)
                ))
        
        topic_trend_fig.update_layout(
            xaxis_title="Earnings Call Date",
            yaxis_title="Topic Importance (%)",  # y축 레이블 변경
            height=400,
            plot_bgcolor='#2d2d2d',
            paper_bgcolor='#2d2d2d',
            font=dict(color='white'),
            margin=dict(t=30),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(0,0,0,0.5)"
            )
        )
        
        st.plotly_chart(topic_trend_fig, use_container_width=True)
        
        st.caption("""
        **Topic Importance = Term Frequency (50%) + Section Weight (20%, Financial Highlights/Q&A) + 
        Speaker Role (20%, CEO/CFO/Analysts) + Topic Coherence (10%)**
        """)
    
    with col2:
        st.subheader("☁️ GPT Word Cloud")
        
        # 분기별 데이터 로드
        quarterly_files = {
            'Q3 2024': 'data/raw/JPM_2024_Q3.txt',
            'Q2 2024': 'data/raw/JPM_2024_Q2.txt',
            'Q1 2024': 'data/raw/JPM_2024_Q1.txt',
            'Q4 2023': 'data/raw/JPM_2023_Q4.txt'
        }
        
        # 탭 생성
        tabs = st.tabs(list(quarterly_files.keys()) + ["Yearly View"])
        
        # 전체 텍스트 저장 (yearly view용)
        all_texts = []
        
        # 분기별 탭 처리
        for quarter, filepath in quarterly_files.items():
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    quarter_text = file.read().strip()
                    
                if quarter_text:
                    all_texts.append(quarter_text)
                    
                    # 해당 분기 탭에서 워드클라우드 표시
                    with tabs[list(quarterly_files.keys()).index(quarter)]:
                        st.caption(f"AI Analysis of {quarter} Earnings Call")
                        keywords = get_ai_keywords(quarter_text, "quarterly")
                        
                        if keywords:
                            wordcloud = WordCloud(
                                width=800,
                                height=400,
                                background_color='#2d2d2d',
                                colormap='Blues',
                                prefer_horizontal=0.7,
                                min_font_size=10,
                                max_font_size=50
                            ).generate_from_frequencies(keywords)
                            
                            fig, ax = plt.subplots(figsize=(10,6))
                            ax.imshow(wordcloud)
                            ax.axis('off')
                            ax.set_facecolor('#2d2d2d')
                            fig.patch.set_facecolor('#2d2d2d')
                            st.pyplot(fig)
                            
            except Exception as e:
                st.error(f"Error processing {quarter}: {str(e)}")
        
        # Yearly View 탭
        with tabs[-1]:
            if all_texts:
                st.caption("AI Analysis of Full Year Earnings Calls")
                yearly_text = " ".join(all_texts)
                yearly_keywords = get_ai_keywords(yearly_text, "yearly")
                
                if yearly_keywords:
                    wordcloud = WordCloud(
                        width=800,
                        height=400,
                        background_color='#2d2d2d',
                        colormap='Blues',
                        prefer_horizontal=0.7,
                        min_font_size=10,
                        max_font_size=50
                    ).generate_from_frequencies(yearly_keywords)
                    
                    fig, ax = plt.subplots(figsize=(10,6))
                    ax.imshow(wordcloud)
                    ax.axis('off')
                    ax.set_facecolor('#2d2d2d')
                    fig.patch.set_facecolor('#2d2d2d')
                    st.pyplot(fig)
    
    # 새로운 시각화 섹션 추가
    st.subheader("🎭 Sentiment Analysis")
    pie_fig, keywords_fig = create_sentiment_viz()
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(pie_fig, use_container_width=True)
    with col2:
        st.plotly_chart(keywords_fig, use_container_width=True)
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Speaker Sentiment Analysis 섹션
    st.subheader("👥 Speaker Sentiment Analysis")
    
    # 발언자 목록과 데이터 가져오기
    sentiment_fig, temporal_data = create_temporal_sentiment_viz(text_data)

    # Jeremy Barnum의 발언만 필터링
    barnum_statements = temporal_data[temporal_data['speaker'].str.contains('Barnum', case=False, na=False)]
    
    if not barnum_statements.empty:
        st.subheader("CFO Jeremy Barnum's Statements")
        
        # 처음 3개의 발언 표시
        statements_shown = 0
        remaining_statements = []
        
        for _, row in barnum_statements.iterrows():
            text = row['text'].strip()
            if len(text.split()) > 6:
                if statements_shown < 3:
                    st.markdown(f"""
                        <div style='background: #363636; padding: 15px; border-radius: 5px; margin: 10px 0;'>
                            <div style='color: #8ab4f8; margin-bottom: 5px;'>Sentiment Score: {row['sentiment_score']:.3f}</div>
                            <div>{text}</div>
                        </div>
                    """, unsafe_allow_html=True)
                    statements_shown += 1
                else:
                    remaining_statements.append((text, row['sentiment_score']))
        
        # 나머지 발언들은 expander에 넣기
        if remaining_statements:
            with st.expander("Show More Statements", expanded=False):
                for text, score in remaining_statements:
                    st.markdown(f"""
                        <div style='background: #363636; padding: 15px; border-radius: 5px; margin: 10px 0;'>
                            <div style='color: #8ab4f8; margin-bottom: 5px;'>Sentiment Score: {score:.3f}</div>
                            <div>{text}</div>
                        </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("No statements found from Jeremy Barnum in this transcript.")
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    st.subheader("📈 JPM Growth & Hiring Trends")
    
    # 탭 생성
    trend_tabs = st.tabs(["Business Growth", "Hiring & Tech", "Strategic Focus"])
    
    with trend_tabs[1]:
        st.caption("Hiring & Technology Investment Trends")
        st.markdown("""
            ### Quotes
            
            #### Q3 2024
            - 💼 **Compensation & Hiring**: 
                - "continued growth in our private banking advisor teams"
                - "higher compensation, primarily revenue-related compensation"
            - 💻 **Technology**: 
                - "continued growth in technology and marketing"
                - "digital platform enhancement"
            
            #### Q2 2024
            - 💼 **Compensation & Hiring**: 
                - "expenses up 12% year-on-year, largely driven by higher compensation"
                - "field compensation and continued growth in technology and marketing"
            - 💻 **Technology**: 
                - "strong customer acquisition across checking accounts and card"
                - "record number of first-time investors"
            
            #### Q1 2024
            - 💼 **Compensation & Hiring**: 
                - "growth in our private banking advisor teams"
            - 💻 **Technology**: 
                - "digital platform growth"
                - "increased mobile adoption"
            
            #### Q4 2023
            - 💼 **Compensation & Hiring**: 
                - "higher compensation, primarily revenue-related compensation"
            - 💻 **Technology**: 
                - "technology infrastructure investments"
        """)
    
    with trend_tabs[2]:
        st.caption("Strategic Investment Focus Areas - Quarterly Comparison")
        strategic_focus = {
            'Q3 2024': {
                '🌐 International': 'Record quarterly revenues, $72B long-term net inflows',
                '💻 Digital': 'Ranked #1 in retail deposit share, strong customer acquisition',
                '🏦 Network': '#1 in retail deposit share for fourth straight year'
            },
            'Q2 2024': {
                '🌐 International': 'IB fees up 50% YoY, #1 wallet share of 9.5%',
                '💻 Digital': 'Record first-time investors, strong customer acquisition',
                '🏦 Network': 'Strong net inflows across AWM'
            },
            'Q1 2024': {
                '🌐 International': 'Strong net inflows led by equities and fixed income',
                '�� Digital': 'Digital platform growth, increased mobile adoption',
                '🏦 Network': 'Continued branch expansion in key markets'
            },
            'Q4 2023': {
                '🌐 International': 'Net inflows $52B, led by equities and fixed income',
                '💻 Digital': 'Digital users up 12%, AI projects initiated',
                '🏦 Network': 'Market share gains in deposits'
            }
        }
        
        # 분기별 전략 포커스 표시
        for quarter, strategies in strategic_focus.items():
            st.subheader(quarter)
            for area, details in strategies.items():
                st.markdown(f"**{area}**\n- {details}")
            st.markdown("---")

    # 시각화 및 텍스트 표시
    with trend_tabs[0]:
        st.caption("Quarter-over-Quarter Financial Performance")
        # Altair chart for financial metrics with units in tooltips
        metrics_df = pd.DataFrame([
            {'Quarter': quarter, 'Metric': 'Net Income', 'Value': data['Net Income'], 'Unit': 'billion'} for quarter, data in quarterly_data.items()
        ] + [
            {'Quarter': quarter, 'Metric': 'EPS', 'Value': data['EPS'], 'Unit': 'dollars'} for quarter, data in quarterly_data.items()
        ] + [
            {'Quarter': quarter, 'Metric': 'Revenue', 'Value': data['Revenue'], 'Unit': 'billion'} for quarter, data in quarterly_data.items()
        ])

        chart = alt.Chart(metrics_df).mark_line(point=True).encode(
            x='Quarter',
            y='Value',
            color='Metric',
            tooltip=['Quarter', 'Metric', alt.Tooltip('Value', format='.2f'), 'Unit']
        ).properties(
            width=600,
            height=400
        )

        st.altair_chart(chart, use_container_width=True)

        # 분기별 주요 내용 표시
        st.write("### Key Highlights for Each Quarter")
        for quarter, highlights in quarterly_highlights.items():
            st.subheader(quarter)
            st.write("\n".join(["- " + highlight for highlight in highlights]))

if __name__ == "__main__":
    main()

EARNINGS_CALL_INSIGHTS = {
    'financial_highlights': {
        'title': 'Financial Performance Highlights',
        'content': [
            {
                'topic': 'Revenue',
                'text': "Our revenue for the quarter was $40.7 billion, up 21% year-on-year...",
                'sentiment_score': 0.6,
                'timestamp': '10:05'
            },
            {
                'topic': 'Net Income',
                'text': "Net income of $13.2 billion reflected strong underlying performance...",
                'sentiment_score': 0.8,
                'timestamp': '10:08'
            }
        ]
    },
    'strategic_updates': {
        'title': 'Strategic Initiatives & Business Updates',
        'content': [
            {
                'topic': 'Digital Banking',
                'text': "Our digital platform saw significant growth with active users up 15%...",
                'sentiment_score': 0.7,
                'timestamp': '10:15'
            }
        ]
    },
    'future_outlook': {
        'title': 'Future Outlook & Guidance',
        'content': [
            {
                'topic': '2024 Outlook',
                'text': "We expect continued momentum in our core businesses...",
                'sentiment_score': 0.5,
                'timestamp': '10:45'
            }
        ]
    },
    'qa_highlights': {
        'title': 'Key Q&A Insights',
        'content': [
            {
                'topic': 'Credit Quality',
                'question': "Can you provide more color on credit quality trends?",
                'answer': "Credit quality remains strong across our portfolios...",
                'sentiment_score': 0.4,
                'timestamp': '11:15'
            }
        ]
    }
}
