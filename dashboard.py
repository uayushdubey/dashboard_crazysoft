import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import numpy as np

# Load the dataset
df = pd.read_excel(r'D:\CrazySoft\Model_Output_Screen\stock_prediction_results_20250721_062729.xlsx', parse_dates=['Datetime'])

# Set page config for a modern, immersive look
st.set_page_config(layout="wide", page_title="CrazySoft ML Model Outputs", initial_sidebar_state="expanded")

# Custom CSS for an amazing, futuristic design with color-coded probabilities
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #e0e0e0;
        font-family: 'Orbitron', 'Arial', sans-serif;
    }
    .stApp > header {
        background: linear-gradient(90deg, #1a73e8, #4a90e2);
        color: white;
        padding: 10px;
        border-bottom: 2px solid #4a90e2;
    }
    .stButton>button {
        background: linear-gradient(45deg, #1a73e8, #4a90e2);
        color: white;
        border-radius: 12px;
        padding: 10px 20px;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(26, 115, 232, 0.4);
    }
    .stButton>button:hover {
        background: linear-gradient(45deg, #1557b0, #357abd);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(26, 115, 232, 0.6);
    }
    .metric-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(5px);
        padding: 15px;
        border-radius: 15px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.2);
        animation: fadeIn 0.5s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        text-align: center;
        padding: 10px;
        background: linear-gradient(90deg, #1a73e8, #4a90e2);
        color: white;
        font-size: 12px;
        box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.2);
    }
    .chart-container {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        background: rgba(255, 255, 255, 0.05);
    }
    .up-prob { color: #4caf50; font-weight: bold; }
    .down-prob { color: #f44336; font-weight: bold; }
    .neutral-prob { color: #9e9e9e; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# Sidebar for filters and interactivity
st.sidebar.title("Control Panel")
all_stocks = st.sidebar.checkbox("Show All Stocks Analysis", value=False)
stock_filter = st.sidebar.selectbox("Select Stock", options=['All'] + df['STOCK'].unique().tolist(), index=df['STOCK'].unique().tolist().index('ACC24SEPFUT') if 'ACC24SEPFUT' in df['STOCK'].unique() else 0)
time_frames = {'1 Day': 1, '1 Week': 7, '1 Month': 30, '3 Months': 90, '1 Year': 365, 'All': None}
time_frame = st.sidebar.selectbox("Time Frame", options=list(time_frames.keys()), index=5)
prob_threshold = st.sidebar.slider("Min Probability Threshold (%)", 0, 100, 30, 5)
show_open = st.sidebar.checkbox("Show Open Lines", value=True)
refresh = st.sidebar.button("Refresh Data")

# Calculate date range based on time frame
end_date = datetime.now()
start_date = end_date - timedelta(days=time_frames[time_frame]) if time_frames[time_frame] else df['Datetime'].min()

# Simulate data refresh
if refresh:
    time.sleep(1)  # Simulate loading
    st.experimental_rerun()

# Filter the data
if all_stocks:
    filtered_df = df[
        (df['Datetime'].dt.date >= start_date.date()) &
        (df['Datetime'].dt.date <= end_date.date()) &
        ((df['UP_Prob'] >= prob_threshold / 100) | (df['DOWN_Prob'] >= prob_threshold / 100) | (df['NEUTRAL_Prob'] >= prob_threshold / 100))
    ]
else:
    filtered_df = df[
        (df['STOCK'] == stock_filter) &
        (df['Datetime'].dt.date >= start_date.date()) &
        (df['Datetime'].dt.date <= end_date.date()) &
        ((df['UP_Prob'] >= prob_threshold / 100) | (df['DOWN_Prob'] >= prob_threshold / 100) | (df['NEUTRAL_Prob'] >= prob_threshold / 100))
    ]

# Calculate up/down/neutral analysis
filtered_df['Success'] = np.where(
    ((filtered_df['Prediction'] == 'UP') & (filtered_df['Actual_Close'] > filtered_df['Predicted_Close'])) |
    ((filtered_df['Prediction'] == 'DOWN') & (filtered_df['Actual_Close'] < filtered_df['Predicted_Close'])) |
    ((filtered_df['Prediction'] == 'NEUTRAL') & (abs(filtered_df['Actual_Close'] - filtered_df['Predicted_Close']) <= 10)),
    1, 0
)
up_count = len(filtered_df[filtered_df['Prediction'] == 'UP'])
down_count = len(filtered_df[filtered_df['Prediction'] == 'DOWN'])
neutral_count = len(filtered_df[filtered_df['Prediction'] == 'NEUTRAL'])
up_success_rate = (filtered_df[filtered_df['Prediction'] == 'UP']['Success'].mean() * 100) if up_count > 0 else 0
down_success_rate = (filtered_df[filtered_df['Prediction'] == 'DOWN']['Success'].mean() * 100) if down_count > 0 else 0
neutral_success_rate = (filtered_df[filtered_df['Prediction'] == 'NEUTRAL']['Success'].mean() * 100) if neutral_count > 0 else 0

# Main dashboard layout
st.title("CrazySoft ML Model Outputs")

# Key Metrics Section with Analysis
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric("UP Predictions", f"{up_count} ({up_success_rate:.1f}%)")
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric("DOWN Predictions", f"{down_count} ({down_success_rate:.1f}%)")
    st.markdown('</div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric("NEUTRAL Predictions", f"{neutral_count} ({neutral_success_rate:.1f}%)")
    st.markdown('</div>', unsafe_allow_html=True)

# Interactive Data Table with Color-Coded Probabilities
st.subheader("Prediction Insights")
styled_df = filtered_df.copy()
styled_df['UP_Prob'] = styled_df['UP_Prob'].apply(lambda x: f'<span class="up-prob">{x:.2f}%</span>')
styled_df['DOWN_Prob'] = styled_df['DOWN_Prob'].apply(lambda x: f'<span class="down-prob">{x:.2f}%</span>')
styled_df['NEUTRAL_Prob'] = styled_df['NEUTRAL_Prob'].apply(lambda x: f'<span class="neutral-prob">{x:.2f}%</span>')
st.write(styled_df[['STOCK', 'Datetime', 'Predicted_Open', 'Actual_Open', 'Predicted_Close', 'Actual_Close', 'Predicted_High', 'Actual_High', 'Predicted_Low', 'Actual_Low', 'UP_Prob', 'DOWN_Prob', 'NEUTRAL_Prob', 'Prediction']].to_html(escape=False, index=False), unsafe_allow_html=True)

# Charts Section with Enhanced Visualizations
col4, col5, col6 = st.columns(3)

with col4:
    st.subheader("Price Dynamics")
    price_fig = go.Figure()
    if show_open:
        price_fig.add_trace(go.Scatter(x=filtered_df['Datetime'], y=filtered_df['Predicted_Open'], name='Predicted Open', line=dict(color='#1a73e8', width=2.5), mode='lines+markers'))
        price_fig.add_trace(go.Scatter(x=filtered_df['Datetime'], y=filtered_df['Actual_Open'], name='Actual Open', line=dict(color='#ffff00', width=2.5), mode='lines+markers'))
    price_fig.add_trace(go.Scatter(x=filtered_df['Datetime'], y=filtered_df['Predicted_Close'], name='Predicted Close', line=dict(color='#f44336', width=2.5), mode='lines+markers'))
    price_fig.add_trace(go.Scatter(x=filtered_df['Datetime'], y=filtered_df['Actual_Close'], name='Actual Close', line=dict(color='#ffffff', width=2.5), mode='lines+markers'))
    price_fig.add_trace(go.Scatter(x=filtered_df['Datetime'], y=filtered_df['Predicted_High'], name='Predicted High', line=dict(color='#4caf50', width=2.5), mode='lines+markers'))
    price_fig.add_trace(go.Scatter(x=filtered_df['Datetime'], y=filtered_df['Actual_High'], name='Actual High', line=dict(color='#f44336', width=2.5), mode='lines+markers'))
    price_fig.add_trace(go.Scatter(x=filtered_df['Datetime'], y=filtered_df['Predicted_Low'], name='Predicted Low', line=dict(color='#e91e63', width=2.5), mode='lines+markers'))
    price_fig.add_trace(go.Scatter(x=filtered_df['Datetime'], y=filtered_df['Actual_Low'], name='Actual Low', line=dict(color='#ffd700', width=2.5), mode='lines+markers'))
    price_fig.update_layout(
        xaxis_title="Date", yaxis_title="Price", template="plotly_dark",
        height=350, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    st.plotly_chart(price_fig, use_container_width=True, config={'displayModeBar': True})

with col5:
    st.subheader("Trend Analysis")
    trend_fig = go.Figure()
    trend_fig.add_trace(go.Bar(x=['UP'], y=[up_count], name='UP', marker_color='#4caf50', text=[f"{up_success_rate:.1f}%"], textposition='auto'))
    trend_fig.add_trace(go.Bar(x=['DOWN'], y=[down_count], name='DOWN', marker_color='#f44336', text=[f"{down_success_rate:.1f}%"], textposition='auto'))
    trend_fig.add_trace(go.Bar(x=['NEUTRAL'], y=[neutral_count], name='NEUTRAL', marker_color='#9e9e9e', text=[f"{neutral_success_rate:.1f}%"], textposition='auto'))
    trend_fig.update_layout(
        xaxis_title="Trend", yaxis_title="Count", template="plotly_dark",
        height=350, showlegend=True, barmode='group',
        hovermode="x unified", bargap=0.2
    )
    st.plotly_chart(trend_fig, use_container_width=True, config={'displayModeBar': True})

with col6:
    st.subheader("Probability Distribution")
    prob_fig = go.Figure()
    prob_fig.add_trace(go.Scatter(x=filtered_df['Datetime'], y=filtered_df['UP_Prob'], name='Up Prob', stackgroup='one', line=dict(color='#4caf50', width=2), fill='tozeroy'))
    prob_fig.add_trace(go.Scatter(x=filtered_df['Datetime'], y=filtered_df['DOWN_Prob'], name='Down Prob', stackgroup='one', line=dict(color='#f44336', width=2), fill='tozeroy'))
    prob_fig.add_trace(go.Scatter(x=filtered_df['Datetime'], y=filtered_df['NEUTRAL_Prob'], name='Neutral Prob', stackgroup='one', line=dict(color='#9e9e9e', width=2), fill='tozeroy'))
    prob_fig.update_layout(
        xaxis_title="Date", yaxis_title="Probability %", template="plotly_dark",
        height=350, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    st.plotly_chart(prob_fig, use_container_width=True, config={'displayModeBar': True})

# Footer
st.markdown('<div class="footer">2025 CrazySoft All Rights Reserved</div>', unsafe_allow_html=True)