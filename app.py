import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from openai import OpenAI
import os
import plotly.express as px

# OpenAI 클라이언트 설정
client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

def get_chatgpt_response(prompt, context):
    """GPT를 사용하여 응답 생성"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a financial analyst assistant. Use the provided context to answer questions about the earnings call."},
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
    
    # 노드 위치 계산
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
    """향상된 키워드 네트워크 시각화"""
    network_df = pd.read_csv('data/network_data.csv')
    
    # 네트워크 생성
    G = nx.from_pandas_edgelist(network_df, 'source', 'target', 'weight')
    
    # 노드 가중치 계산 수정
    node_weights = {}
    for node in G.nodes():
        # 연결된 엣지들의 가중치 합계 계산
        weight_sum = sum(G[node][neighbor]['weight'] for neighbor in G[node])
        node_weights[node] = weight_sum
    
    # 노드 위치 계산
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # 엣지 트레이스
    edge_x = []
    edge_y = []
    edge_weights = []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_weights.extend([edge[2]['weight']] * 3)
    
    # 노드 트레이스
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{node}<br>Weight: {node_weights[node]}")
        node_size.append(node_weights[node] * 10)
    
    # 시각화 생성
    network_fig = go.Figure()
    
    # 엣지 추가
    network_fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#666'),
        hoverinfo='none',
        mode='lines'
    ))
    
    # 노드 추가
    network_fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="bottom center",
        marker=dict(
            size=node_size,
            color='#0066ff',
            line_width=2,
            line=dict(color='#444')
        )
    ))
    
    network_fig.update_layout(
        title="Keyword Relationship Network",
        showlegend=False,
        plot_bgcolor='#2d2d2d',
        paper_bgcolor='#2d2d2d',
        font=dict(color='white'),
        height=600
    )
    
    return network_fig

def main():
    # 페이지 설정
    st.set_page_config(
        page_title="JPM Earnings Call Analysis",
        page_icon="📊",
        layout="wide"
    )
    
    # 데이터 로드
    try:
        # 워드클라우드용 텍스트 파일 로딩
        with open('data/processed/JPM_2024_Q3_wordcloud.txt', 'r') as file:
            text_data = file.read()
            
        # 토픽 모델링을 위한 텍스트 전처리
        from utils.preprocessing import preprocess_text, extract_topics
        processed_text = preprocess_text(text_data)
        topics = extract_topics([processed_text])
        
        # 토픽 정보를 DataFrame으로 변환
        topic_info = pd.DataFrame([{
            'Topic_Num': i+1,
            'Top_Phrases': ' | '.join([term for term, _ in topic['terms']]),
            'Size': sum([score for _, score in topic['terms']]),
            'Label': topic['label']
        } for i, topic in enumerate(topics)])
        
    except FileNotFoundError:
        st.error("필요한 데이터 파일을 찾을 수 없습니다.")
        return
    
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
    
    # 메인 더
    st.markdown("""
        <div class='main-header'>
            <h1 style='margin:0;'>JPMorgan Chase Q3 2024 Earnings Call Analysis</h1>
            <p style='margin:0.5rem 0 0 0;'>AI-Powered Insights from the Latest Earnings Call</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Financial Analysis 섹션
    if 'financial_analysis' not in st.session_state:
        core_metrics_prompt = """Extract 3-4 key financial metrics from the earnings call:
        • Focus on exact numbers and YoY changes
        • Start each point with the most important number/change
        • Format: "Revenue up 15% YoY to $12.3B"
        • Keep each point to one line
        • Include only the most significant metrics"""
        
        future_outlook_prompt = """Extract 3-4 key points about future outlook:
        • Focus on specific guidance and targets
        • Start with the most important prediction/target
        • Include direct quotes from CEO/CFO
        • Format: "2024 Target: $X revenue, citing 'relevant quote'"
        • Keep each point to one line"""
        
        strategy_risks_prompt = """Extract 3-4 key strategic points and risks:
        • Focus on concrete plans and specific challenges
        • Start each point with the key strategy/risk
        • Include management's direct responses
        • Format: "Digital Banking: Investing $XB in tech, 'relevant quote'"
        • Keep each point to one line"""
        
        st.session_state.financial_analysis = {
            'core_metrics': get_chatgpt_response(core_metrics_prompt, text_data),
            'future_outlook': get_chatgpt_response(future_outlook_prompt, text_data),
            'strategy_risks': get_chatgpt_response(strategy_risks_prompt, text_data)
        }
    
    core_metrics = st.session_state.financial_analysis['core_metrics'].replace('\n', '<br>')
    future_outlook = st.session_state.financial_analysis['future_outlook'].replace('\n', '<br>')
    strategy_risks = st.session_state.financial_analysis['strategy_risks'].replace('\n', '<br>')
    
    st.markdown("""
        <div class='insights-container'>
            <div class='insights-title'>
                📈 JPMorgan Chase Q3 2024 Deep Dive
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
    
    # 사이드바 챗봇
    with st.sidebar:
        st.markdown("""
            <div style='padding: 1rem; background: linear-gradient(180deg, #0033cc 0%, #0066ff 100%); 
                        color: white; border-radius: 10px; margin-bottom: 1rem;'>
                <h3 style='margin:0;'>💬 AI Financial Analyst</h3>
            </div>
        """, unsafe_allow_html=True)
        
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        context = f"""
        Topics and key phrases from the earnings call:
        {topic_info['Top_Phrases'].to_string()}
        
        Main discussion points based on topic sizes:
        {topic_info[['Topic_Num', 'Size', 'Top_Phrases']].to_string()}
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
    
    # 주요 지표 행
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
            <div class='metric-card'>
                <div class='metric-value'>{}</div>
                <div class='metric-label'>Topics Identified</div>
            </div>
        """.format(len(topic_info)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='metric-card'>
                <div class='metric-value'>{}</div>
                <div class='metric-label'>Total Phrases Analyzed</div>
            </div>
        """.format(topic_info['Size'].sum()), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class='metric-card'>
                <div class='metric-value'>{}</div>
                <div class='metric-label'>Key Topics</div>
            </div>
        """.format(len(topic_info[topic_info['Size'] > topic_info['Size'].mean()])), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
            <div class='metric-card'>
                <div class='metric-value'>Q3 2024</div>
                <div class='metric-label'>Earnings Period</div>
            </div>
        """, unsafe_allow_html=True)
    
    # 차트 영역
    col1, col2 = st.columns([2,1])
    
    with col1:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.subheader("📊 Financial Topics Distribution")
        
        # 토픽 분포 시각화
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f"{row['Label']}<br>Topic {row['Topic_Num']}" for _, row in topic_info.iterrows()],
            y=topic_info['Size'],
            text=topic_info['Top_Phrases'],
            textposition='auto',
            marker_color='#0066ff'
        ))
        
        fig.update_layout(
            xaxis_title="Topics",
            yaxis_title="Importance Score",
            height=400,
            plot_bgcolor='#2d2d2d',
            paper_bgcolor='#2d2d2d',
            font=dict(color='white'),
            margin=dict(t=30)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.subheader("☁️ Word Cloud")
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='#2d2d2d',
            colormap='Blues'
        ).generate(text_data)
        
        fig, ax = plt.subplots(figsize=(10,6))
        ax.imshow(wordcloud)
        ax.axis('off')
        ax.set_facecolor('#2d2d2d')
        fig.patch.set_facecolor('#2d2d2d')
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # 네트워크 맵
    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
    st.subheader("🔄 Financial Topic Network")
    from utils.visualization import create_topic_network
    network_fig = create_topic_network(topics)
    st.plotly_chart(network_fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # 새로운 시각화 섹션 추가 (차트 영역 다음에)
    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
    st.subheader("🎭 Sentiment Analysis")
    
    # 감성 분석 시각화
    pie_fig, keywords_fig = create_sentiment_viz()
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(pie_fig, use_container_width=True)
    with col2:
        st.plotly_chart(keywords_fig, use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # 향상된 키워드 네트워크
    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
    st.subheader("🔄 Enhanced Keyword Network")
    network_fig = create_enhanced_network()
    st.plotly_chart(network_fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

