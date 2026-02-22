"""NCT Real-time Visualization Dashboard - Streamlit Web Interface
NeuroConscious Transformer Real-time Dashboard

Features:
1. Real-time monitoring of Φ value, Free Energy, Attention Weights
2. Interactive parameter adjustment
3. Experiment data visualization comparison
4. One-click comparison with paper results

Usage:
    streamlit run nct_dashboard.py
    
Dependencies:
    pip install streamlit plotly pandas
"""

import sys
import os
import numpy as np
import torch
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# 添加 NCT 模块到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nct_modules import NCTManager, NCTConfig


def generate_continuous_sensory(cycle_idx, noise_level=0.1):
    """生成连续性感觉输入（模拟真实世界的时序相关性）
    
    Args:
        cycle_idx: 当前周期索引
        noise_level: 噪声水平（0-1）
    
    Returns:
        sensory_data: 连续变化的感觉输入
    """
    # 使用正弦波 + 缓慢漂移 + 少量噪声，模拟自然刺激
    t = cycle_idx * 0.2  # 时间缩放因子
    
    # 视觉输入：基础模式 + 时间调制
    base_visual = np.sin(t) * 0.5 + 0.5  # [0, 1] 范围
    visual_pattern = np.ones((1, 28, 28)) * base_visual
    # 添加空间变化
    x, y = np.meshgrid(np.linspace(-1, 1, 28), np.linspace(-1, 1, 28))
    spatial_modulation = np.sin(x * 3 + t) * np.cos(y * 3 - t) * 0.3
    visual_pattern += spatial_modulation
    visual_pattern = np.clip(visual_pattern, 0, 1)
    
    # 听觉输入：多频率组合
    audio_freq1 = np.sin(t * 1.5) * 0.4 + 0.5
    audio_freq2 = np.sin(t * 0.8 + 1) * 0.3 + 0.5
    audio_pattern = (audio_freq1 + audio_freq2) / 2
    audio_pattern = audio_pattern + np.random.randn(10, 10) * noise_level * 0.1
    audio_pattern = np.clip(audio_pattern, 0, 1)
    
    # 内感受输入：缓慢变化的生理信号
    intero_pattern = np.sin(t * 0.5) * 0.3 + 0.5
    intero_pattern = intero_pattern + np.random.randn(10) * noise_level * 0.05
    intero_pattern = np.clip(intero_pattern, -1, 1)
    
    return {
        'visual': visual_pattern.astype(np.float32),
        'auditory': audio_pattern.astype(np.float32),
        'interoceptive': intero_pattern.astype(np.float32),
    }
from nct_modules.nct_metrics import PhiFromAttention

# ============================================================================
# Internationalization (i18n) Support
# ============================================================================

TRANSLATIONS = {
    'en': {
        # Page config
        'page_title': 'NCT Real-time Dashboard',
        
        # Sidebar
        'param_config': '⚙️ Parameter Configuration',
        'arch_params': '🏗️ Architecture Parameters',
        'd_model': 'Model Dimension (d_model)',
        'n_heads': 'Number of Attention Heads',
        'n_layers': 'Number of Transformer Layers',
        'gamma_freq': 'γ-wave Frequency (Hz)',
        'exp_params': '🔬 Experiment Parameters',
        'n_cycles': 'Number of Consciousness Cycles',
        'noise_level': 'Input Noise Level',
        'noise_help': 'Controls the random noise intensity of input signals (lower = smoother)',
        'show_phi': 'Show Φ Value Calculation',
        'show_fe': 'Show Free Energy',
        'show_attention': 'Show Attention Heatmap',
        'control_panel': '🎮 Control Panel',
        'start_btn': '▶️ Start Running',
        'stop_btn': '⏹️ Stop',
        'reset_btn': '🔄 Reset',
        'paper_comparison': '📊 Paper Reference',
        'show_paper_ref': 'Show Paper Φ Reference (d=768)',
        'lang_settings': '🌐 Language',
        'lang_select': 'Select Language',
        
        # Main interface
        'main_header': '🧠 NCT Real-time Visualization Dashboard',
        'running_status': 'Running - Cycle {}/{}',
        'complete_msg': '✅ Completed {} consciousness cycles!',
        'stopped_msg': '⏹️ Stopped',
        'reset_msg': '🔄 Reset complete',
        'view_details': '📋 View Detailed Data',
        'download_csv': '📥 Download CSV Data',
        
        # Charts
        'metrics_chart_title': '📈 Dynamic Changes in Consciousness Metrics',
        'cycle': 'Cycle',
        'phi_value': 'Φ Value',
        'free_energy': 'Free Energy',
        'paper_phi_note': 'Paper Φ (d=768)',
        'paper_fe_note': 'Paper FE Final',
        
        # Attention heatmap
        'attention_title': '🎯 Multi-candidate Competition - Attention Weight Distribution',
        'attention_subtitle': '4 candidates compete in global workspace, winner broadcasts conscious content',
        'attention_title_sim': '🎯 Multi-candidate Competition - Attention Weight Distribution (Simulated Data)',
        'candidate_content': 'Candidate Content',
        'attention_weight': 'Attention Weight',
        'winner': '🏆 Winner',
        'integrated_repr': 'Integrated Repr',
        'visual_feature': 'Visual Feature',
        'auditory_feature': 'Auditory Feature',
        'intero_feature': 'Interoceptive Feature',
        
        # Confidence gauge
        'confidence': '🎯 Confidence',
        
        # Metrics
        'salience': 'Salience',
        
        # Footer
        'version': 'Version',
        'paper': 'Paper',
        'paper_status': 'arXiv:xxxx.xxxxx (Coming soon)',
    },
    'zh': {
        # Page config
        'page_title': 'NCT 实时仪表盘',
        
        # Sidebar
        'param_config': '⚙️ 参数配置',
        'arch_params': '🏗️ 架构参数',
        'd_model': '模型维度 (d_model)',
        'n_heads': '注意力头数',
        'n_layers': 'Transformer 层数',
        'gamma_freq': 'γ波频率 (Hz)',
        'exp_params': '🔬 实验参数',
        'n_cycles': '意识周期数',
        'noise_level': '输入噪声水平',
        'noise_help': '控制输入信号的随机噪声强度（越小越平滑）',
        'show_phi': '显示 Φ值计算',
        'show_fe': '显示自由能',
        'show_attention': '显示注意力热力图',
        'control_panel': '🎮 控制面板',
        'start_btn': '▶️ 开始运行',
        'stop_btn': '⏹️ 停止',
        'reset_btn': '🔄 重置',
        'paper_comparison': '📊 论文参考',
        'show_paper_ref': '显示论文 Φ 参考值 (d=768)',
        'lang_settings': '🌐 语言',
        'lang_select': '选择语言',
        
        # Main interface
        'main_header': '🧠 NCT 实时可视化仪表盘',
        'running_status': '运行中 - 周期 {}/{}',
        'complete_msg': '✅ 完成 {} 个意识周期！',
        'stopped_msg': '⏹️ 已停止运行',
        'reset_msg': '🔄 已重置',
        'view_details': '📋 查看详细数据',
        'download_csv': '📥 下载 CSV 数据',
        
        # Charts
        'metrics_chart_title': '📈 意识指标动态变化',
        'cycle': '周期',
        'phi_value': 'Φ值',
        'free_energy': '自由能',
        'paper_phi_note': '论文Φ值 (d=768)',
        'paper_fe_note': '论文 FE 终值',
        
        # Attention heatmap
        'attention_title': '🎯 多候选竞争 - 注意力权重分布',
        'attention_subtitle': '4 个候选在全局工作空间中竞争，胜者获得意识内容广播权',
        'attention_title_sim': '🎯 多候选竞争 - 注意力权重分布（模拟数据）',
        'candidate_content': '候选内容',
        'attention_weight': '注意力权重',
        'winner': '🏆 获胜者',
        'integrated_repr': '整合表征',
        'visual_feature': '视觉特征',
        'auditory_feature': '听觉特征',
        'intero_feature': '内感受特征',
        
        # Confidence gauge
        'confidence': '🎯 自信度',
        
        # Metrics
        'salience': '显著性',
        
        # Footer
        'version': '版本',
        'paper': '论文',
        'paper_status': 'arXiv:xxxx.xxxxx (即将提交)',
    }
}

def get_text(key, lang='en'):
    """Get translated text by key and language"""
    return TRANSLATIONS.get(lang, TRANSLATIONS['en']).get(key, key)

# ============================================================================
# Streamlit Page Configuration
# ============================================================================
import streamlit as st

st.set_page_config(
    page_title="NCT Real-time Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styles
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 1rem;
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 0.5rem;
    color: white;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Language Selection (at the top of sidebar)
# ============================================================================
if 'language' not in st.session_state:
    st.session_state.language = 'en'  # Default to English

st.sidebar.subheader(get_text('lang_settings', st.session_state.language))
lang_options = {'English': 'en', '中文': 'zh'}
lang_display = {v: k for k, v in lang_options.items()}
selected_lang_display = st.sidebar.selectbox(
    get_text('lang_select', st.session_state.language),
    options=list(lang_options.keys()),
    index=0 if st.session_state.language == 'en' else 1
)
st.session_state.language = lang_options[selected_lang_display]
lang = st.session_state.language

# ============================================================================
# Sidebar - Parameter Configuration
# ============================================================================
st.sidebar.title(get_text('param_config', lang))

# Model architecture parameters
st.sidebar.subheader(get_text('arch_params', lang))
d_model = st.sidebar.slider(get_text('d_model', lang), 64, 768, 256, step=64)
n_heads = st.sidebar.slider(get_text('n_heads', lang), 4, 16, 8)
n_layers = st.sidebar.slider(get_text('n_layers', lang), 2, 8, 4)
gamma_freq = st.sidebar.slider(get_text('gamma_freq', lang), 30.0, 50.0, 40.0, step=5.0)

# Experiment parameters
st.sidebar.subheader(get_text('exp_params', lang))
n_cycles = st.sidebar.slider(get_text('n_cycles', lang), 5, 100, 20)
noise_level = st.sidebar.slider(
    get_text('noise_level', lang),
    min_value=0.0,
    max_value=0.5,
    value=0.15,
    step=0.05,
    help=get_text('noise_help', lang)
)
show_phi = st.sidebar.checkbox(get_text('show_phi', lang), value=True)
show_fe = st.sidebar.checkbox(get_text('show_fe', lang), value=True)
show_attention = st.sidebar.checkbox(get_text('show_attention', lang), value=True)

# Control buttons
st.sidebar.subheader(get_text('control_panel', lang))
start_btn = st.sidebar.button(get_text('start_btn', lang), type="primary")
stop_btn = st.sidebar.button(get_text('stop_btn', lang), type="secondary")
reset_btn = st.sidebar.button(get_text('reset_btn', lang), type="secondary")

# Paper data comparison
st.sidebar.subheader(get_text('paper_comparison', lang))
show_paper_comparison = st.sidebar.checkbox(get_text('show_paper_ref', lang), value=False)

# ============================================================================
# Main Interface
# ============================================================================
st.markdown(f'<p class="main-header">{get_text("main_header", lang)}</p>', unsafe_allow_html=True)
st.markdown("---")

# 初始化状态
if 'running' not in st.session_state:
    st.session_state.running = False
if 'results_history' not in st.session_state:
    st.session_state.results_history = []
if 'cycle_count' not in st.session_state:
    st.session_state.cycle_count = 0

# 创建占位符
metrics_placeholder = st.empty()
charts_placeholder = st.empty()
log_placeholder = st.empty()

# ============================================================================
# 核心功能函数
# ============================================================================

def create_nct_manager():
    """创建 NCT 管理器"""
    config = NCTConfig(
        n_heads=n_heads,
        n_layers=n_layers,
        d_model=d_model,
        gamma_freq=gamma_freq,
    )
    return NCTManager(config)


def run_cycle(manager, cycle_idx):
    """运行单个意识周期"""
    # 生成连续性感觉输入（替代完全随机输入）
    sensory_data = generate_continuous_sensory(cycle_idx, noise_level=noise_level)
    
    # 处理周期
    state = manager.process_cycle(sensory_data)
    
    # 关键新增：保存注意力权重和 workspace_info 到 session_state
    if hasattr(state, 'diagnostics') and 'workspace' in state.diagnostics:
        workspace_info = state.diagnostics['workspace']
        print(f"💾 保存 workspace 信息")
        st.session_state.last_workspace_info = workspace_info
        
        # 保存注意力 maps
        if 'attention_weights' in workspace_info:
            attn_weights = workspace_info['attention_weights']
            # 转为 tensor 格式 [1, H, 1, N]
            if isinstance(attn_weights, np.ndarray):
                attn_tensor = torch.from_numpy(attn_weights).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, N]
                # 扩展到多头
                attn_tensor = attn_tensor.repeat(1, n_heads, 1, 1)  # [1, H, 1, N]
                st.session_state.last_attention_maps = attn_tensor
    
    # 提取指标
    result = {
        'cycle': cycle_idx,
        'timestamp': datetime.now().strftime('%H:%M:%S'),
        'phi_value': state.consciousness_metrics.get('phi_value', 0),
        'free_energy': state.self_representation['free_energy'],
        'confidence': state.self_representation['confidence'],
        'awareness_level': state.awareness_level,
        'salience': state.workspace_content.salience if state.workspace_content else 0,
    }
    
    return result


def plot_metrics_chart(results_df, show_paper=False, lang='en'):
    """Plot metrics trend chart"""
    fig = go.Figure()
    
    # Φ value curve
    fig.add_trace(go.Scatter(
        x=results_df['cycle'],
        y=results_df['phi_value'],
        mode='lines+markers',
        name=get_text('phi_value', lang),
        line=dict(color='#FF6B6B', width=3),
        marker=dict(size=8, symbol='circle'),
    ))
    
    # Free energy curve (dual Y axis)
    fig.add_trace(go.Scatter(
        x=results_df['cycle'],
        y=results_df['free_energy'],
        mode='lines+markers',
        name=get_text('free_energy', lang),
        line=dict(color='#4ECDC4', width=3, dash='dot'),
        yaxis='y2',
    ))
    
    # Paper reference values (if enabled)
    if show_paper:
        # Φ value reference line (primary Y axis) - Paper: d=768, structured attention
        fig.add_hline(y=0.329, line_dash="dash", line_color="green", 
                     annotation_text=get_text('paper_phi_note', lang), annotation_position="top right")
        # Note: Free energy reference (0.57) is from PredictiveHierarchy after 100-step optimization,
        # which is different from the instant prediction error shown here. Reference line removed.
    
    fig.update_layout(
        title=get_text('metrics_chart_title', lang),
        xaxis_title=get_text('cycle', lang),
        yaxis_title=get_text('phi_value', lang),
        yaxis2=dict(title=get_text('free_energy', lang), overlaying='y', side='right'),
        legend=dict(x=0, y=1.1, orientation='h'),
        height=400,
        hovermode='x unified'
    )
    
    return fig


def plot_attention_heatmap(manager, lang='en'):
    """Plot attention weight distribution (multi-candidate competition version)"""
    # Get candidate names based on language
    candidate_names = [
        get_text('integrated_repr', lang),
        get_text('visual_feature', lang),
        get_text('auditory_feature', lang),
        get_text('intero_feature', lang)
    ]
    
    # Get real attention weights from session_state
    if hasattr(st.session_state, 'last_attention_maps') and st.session_state.last_attention_maps is not None:
        attention_maps = st.session_state.last_attention_maps
        print(f"✅ Using real attention data, shape: {attention_maps.shape}")
        
        # Get all candidates' salience (if workspace_info exists)
        all_salience = []
        if hasattr(st.session_state, 'last_workspace_info'):
            all_salience = st.session_state.last_workspace_info.get('all_candidates_salience', [])
        
        n_candidates = len(all_salience) if all_salience else attention_maps.shape[3]
        candidate_names = candidate_names[:n_candidates]
        
        # Draw bar chart: show attention weight for each candidate
        fig = go.Figure()
        
        # Use average attention weight across all heads
        avg_attention = attention_maps[0, :, 0, :].mean(dim=0).cpu().numpy()  # [N_candidates]
        
        fig.add_trace(go.Bar(
            x=candidate_names,
            y=avg_attention.tolist(),
            marker=dict(color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'][:n_candidates]),
            text=[f'{w:.3f}' for w in avg_attention],
            textposition='auto'
        ))
        
        # Mark the winner
        if hasattr(st.session_state, 'last_workspace_info'):
            winner_idx = st.session_state.last_workspace_info.get('winner_idx', -1)
            if 0 <= winner_idx < n_candidates:
                # Add marker above the winner
                fig.add_annotation(
                    x=candidate_names[winner_idx],
                    y=max(avg_attention) * 1.1,
                    text=get_text('winner', lang),
                    showarrow=False,
                    font=dict(size=16, color='#FFD700')
                )
        
        fig.update_layout(
            title=f'{get_text("attention_title", lang)}<br><span style="font-size:12px;color:#666">{get_text("attention_subtitle", lang)}</span>',
            xaxis_title=get_text('candidate_content', lang),
            yaxis_title=get_text('attention_weight', lang),
            height=450,
            showlegend=False,
            yaxis=dict(range=[0, max(0.5, max(avg_attention) * 1.3)])
        )
        
        return fig
        
    else:
        # If no real data, generate simulated data
        n_candidates = 4
        # Simulate sparse attention
        np.random.seed(42)
        avg_attention = np.random.rand(n_candidates) * 0.3 + 0.2
        avg_attention[0] += 0.2  # Make integrated representation slightly higher
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=candidate_names,
            y=avg_attention.tolist(),
            marker=dict(color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']),
            text=[f'{w:.3f}' for w in avg_attention],
            textposition='auto'
        ))
        
        fig.update_layout(
            title=get_text('attention_title_sim', lang),
            xaxis_title=get_text('candidate_content', lang),
            yaxis_title=get_text('attention_weight', lang),
            height=450,
            showlegend=False,
            yaxis=dict(range=[0, max(0.5, max(avg_attention) * 1.3)])
        )
        
        return fig


def plot_confidence_gauge(confidence, lang='en'):
    """Plot confidence gauge"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': get_text('confidence', lang), 'font': {'size': 24}},
        delta={'reference': 0.5, 'increasing': None, 'decreasing': None},
        gauge={
            'axis': {'range': [None, 1]},
            'bar': {'color': "#FF6B6B"},
            'steps': [
                {'range': [0, 0.3], 'color': "#ffebee"},
                {'range': [0.3, 0.7], 'color': "#fff3e0"},
                {'range': [0.7, 1], 'color': "#e8f5e9"}
            ],
        }
    ))
    
    fig.update_layout(height=300)
    return fig


# ============================================================================
# 运行逻辑
# ============================================================================

if start_btn and not st.session_state.running:
    st.session_state.running = True
    st.session_state.results_history = []
    st.session_state.cycle_count = 0
    
    # Create manager
    manager = create_nct_manager()
    manager.start()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Run specified number of cycles
    for cycle in range(n_cycles):
        result = run_cycle(manager, cycle + 1)
        st.session_state.results_history.append(result)
        st.session_state.cycle_count += 1
        
        # Update progress
        progress_bar.progress((cycle + 1) / n_cycles)
        status_text.text(get_text('running_status', lang).format(cycle + 1, n_cycles))
        
        # Real-time chart update (every 5 cycles)
        if (cycle + 1) % 5 == 0 or cycle == 0:
            results_df = pd.DataFrame(st.session_state.results_history)
            
            with charts_placeholder.container():
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(plot_metrics_chart(results_df, show_paper_comparison, lang), width="stretch", key=f"metrics_chart_{cycle}")
                
                with col2:
                    if show_attention:
                        st.plotly_chart(plot_attention_heatmap(manager, lang), width="stretch", key=f"attention_heatmap_{cycle}")
                    else:
                        st.plotly_chart(plot_confidence_gauge(result['confidence'], lang), width="stretch", key=f"confidence_gauge_{cycle}")
            
            # Update metric cards
            with metrics_placeholder.container():
                col1, col2, col3, col4 = st.columns(4)
                
                latest = results_df.iloc[-1]
                col1.metric(get_text('phi_value', lang), f"{latest['phi_value']:.3f}", delta=None)
                col2.metric(get_text('free_energy', lang), f"{latest['free_energy']:.4f}", delta=f"{latest['free_energy'] - results_df.iloc[0]['free_energy']:.4f}")
                col3.metric(get_text('confidence', lang).replace('🎯 ', ''), f"{latest['confidence']:.3f}")
                col4.metric(get_text('salience', lang), f"{latest['salience']:.3f}")
    
    manager.stop()
    progress_bar.empty()
    status_text.empty()
    
    st.success(get_text('complete_msg', lang).format(n_cycles))
    
    # Show final data table
    with st.expander(get_text('view_details', lang)):
        st.dataframe(pd.DataFrame(st.session_state.results_history))
    
    # Export button
    csv = pd.DataFrame(st.session_state.results_history).to_csv(index=False)
    st.download_button(
        label=get_text('download_csv', lang),
        data=csv,
        file_name=f'nct_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
        mime='text/csv'
    )

elif stop_btn:
    st.session_state.running = False
    st.warning(get_text('stopped_msg', lang))

elif reset_btn:
    st.session_state.running = False
    st.session_state.results_history = []
    st.session_state.cycle_count = 0
    metrics_placeholder.empty()
    charts_placeholder.empty()
    log_placeholder.empty()
    st.info(get_text('reset_msg', lang))

# ============================================================================
# Display existing data (when not running but has history)
# ============================================================================
if not st.session_state.running and st.session_state.results_history:
    results_df = pd.DataFrame(st.session_state.results_history)
    
    with charts_placeholder.container():
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(plot_metrics_chart(results_df, show_paper_comparison, lang), use_container_width=True, key="metrics_chart_static")
        
        with col2:
            if show_attention:
                st.plotly_chart(plot_attention_heatmap(None, lang), use_container_width=True, key="attention_heatmap_static")
            else:
                latest = results_df.iloc[-1]
                st.plotly_chart(plot_confidence_gauge(latest['confidence'], lang), use_container_width=True, key="confidence_gauge_static")
    
    # Update metric cards
    with metrics_placeholder.container():
        col1, col2, col3, col4 = st.columns(4)
        
        latest = results_df.iloc[-1]
        col1.metric(get_text('phi_value', lang), f"{latest['phi_value']:.3f}", delta=None)
        col2.metric(get_text('free_energy', lang), f"{latest['free_energy']:.4f}", delta=f"{latest['free_energy'] - results_df.iloc[0]['free_energy']:.4f}")
        col3.metric(get_text('confidence', lang).replace('🎯 ', ''), f"{latest['confidence']:.3f}")
        col4.metric(get_text('salience', lang), f"{latest['salience']:.3f}")

# ============================================================================
# Footer Information
# ============================================================================
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.write("**GitHub:** https://github.com/wyg5208/nct")
with col2:
    st.write(f"**{get_text('version', lang)}:** v3.1.0")
with col3:
    st.write(f"**{get_text('paper', lang)}:** {get_text('paper_status', lang)}")
