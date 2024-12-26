import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import networkx as nx
import numpy as np
from itertools import combinations

def create_wordcloud(texts, additional_stopwords=['firm', 'quarter', 'year', 'company', 'business', 
                                                'million', 'billion', 'dollars', 'think', 'going', 
                                                'said', 'one', 'will', 'may', 'also', 'could',
                                                'would', 'see', 'well', 'make', 'made', 'really',
                                                'now', 'get', 'got', 'look', 'looking', 'thank',
                                                'thanks', 'good', 'great', 'right', 'first', 'second',
                                                'third', 'next', 'last', 'way', 'time', 'lot']):
    # texts를 문자열로 변환
    if isinstance(texts, list):
        text_data = ' '.join(texts)
    else:
        text_data = str(texts)
    
    # 불용어 설정
    stop_words = set(STOPWORDS)
    stop_words.update(additional_stopwords)
    
    # 워드클라우드 생성
    wordcloud = WordCloud(
        width=800, 
        height=400,
        background_color='white',
        stopwords=stop_words,
        min_font_size=10,
        max_font_size=100
    ).generate(text_data)
    
    # 플로틀리로 인터랙티브 워드클라우드 생성
    words = wordcloud.words_
    
    # 워드클라우드 데이터 준비
    word_list = [(word, freq) for word, freq in words.items()]
    word_list.sort(key=lambda x: x[1], reverse=True)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=[i for i in range(len(word_list))],
        y=[freq for word, freq in word_list],
        mode='text',
        text=[word for word, freq in word_list],
        textfont={
            'size': [freq * 50 for word, freq in word_list]
        },
        hoverinfo='text',
        textposition='middle center'
    ))
    
    fig.update_layout(
        showlegend=False,
        hovermode='closest',
        xaxis={'showgrid': False, 'zeroline': False, 'visible': False},
        yaxis={'showgrid': False, 'zeroline': False, 'visible': False},
        margin=dict(t=20, b=20, l=20, r=20)
    )
    
    return fig

# 메인 앱에 추가할 콜백 함수
def show_word_context(word, texts):
    contexts = []
    for text in texts:
        if word.lower() in text.lower():
            # 단어 주변 컨텍스트 추출 (단어 앞뒤 50자)
            idx = text.lower().find(word.lower())
            start = max(0, idx - 50)
            end = min(len(text), idx + len(word) + 50)
            context = text[start:end]
            if start > 0:
                context = f"...{context}"
            if end < len(text):
                context = f"{context}..."
            contexts.append(context)
    
    return contexts 

def create_financial_insights_viz(financial_data):
    fig = go.Figure()
    
    # 재무 지표 시각화
    fig.add_trace(go.Scatter(
        x=financial_data['date'],
        y=financial_data['value'],
        mode='lines+markers',
        name='Financial Metric'
    ))
    
    fig.update_layout(
        title='Financial Performance Over Time',
        xaxis_title='Date',
        yaxis_title='Value',
        template='plotly_white'
    )
    
    return fig

def create_sentiment_viz(sentiment_scores):
    fig = go.Figure()
    
    # 감성 분석 결과 시각화
    fig.add_trace(go.Bar(
        x=['Positive', 'Neutral', 'Negative'],
        y=[
            sentiment_scores['pos'],
            sentiment_scores['neu'],
            sentiment_scores['neg']
        ]
    ))
    
    fig.update_layout(
        title='Sentiment Analysis Results',
        xaxis_title='Sentiment',
        yaxis_title='Score',
        template='plotly_white'
    )
    
    return fig

def create_topic_network(topics, min_edge_weight=0.3):
    """금융 도메인 특화 토픽 네트워크 시각화"""
    # 네트워크 그래프 생성
    G = nx.Graph()
    
    # 노드 추가 (토픽)
    for topic in topics:
        G.add_node(topic['label'], 
                  size=1.0,  # 기본 크기
                  terms=', '.join([term for term, _ in topic['terms'][:5]]))  # 상위 5개 단어
    
    # 엣지 추가 (토픽 간 관계)
    for topic1, topic2 in combinations(topics, 2):
        terms1 = set([term for term, _ in topic1['terms']])
        terms2 = set([term for term, _ in topic2['terms']])
        
        # 자카드 유사도 계산
        similarity = len(terms1 & terms2) / len(terms1 | terms2)
        
        if similarity >= min_edge_weight:
            G.add_edge(topic1['label'], topic2['label'], weight=similarity)
    
    # 네트워크 레이아웃 계산
    pos = nx.spring_layout(G)
    
    # 엣지 트레이스
    edge_x = []
    edge_y = []
    edge_weights = []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_weights.append(edge[2]['weight'])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines')
    
    # 노드 트레이스
    node_x = []
    node_y = []
    node_labels = []
    node_terms = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_labels.append(node)
        node_terms.append(G.nodes[node]['terms'])
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_labels,
        textposition="top center",
        hovertext=[f"{label}<br>Terms: {terms}" for label, terms in zip(node_labels, node_terms)],
        marker=dict(
            showscale=True,
            colorscale='YlOrRd',
            size=30,
            colorbar=dict(
                thickness=15,
                title='Topic Centrality',
                xanchor='left',
                titleside='right'
            )
        )
    )
    
    # 노드 중심성 계산 및 색상 설정
    centrality = nx.degree_centrality(G)
    node_trace.marker.color = [centrality[node] for node in G.nodes()]
    
    # 레이아웃 설정
    layout = go.Layout(
        title='Financial Topic Network',
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        annotations=[ dict(
            text="",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.005, y=-0.002 ) ],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        template='plotly_white'
    )
    
    # 최종 그래프 생성
    fig = go.Figure(data=[edge_trace, node_trace], layout=layout)
    
    return fig 