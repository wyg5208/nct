"""
NCT 实时可视化仪表盘 - Streamlit Web Interface
NeuroConscious Transformer Real-time Dashboard

功能:
1. 实时监控 Φ值、自由能、注意力权重
2. 交互式参数调整
3. 实验数据可视化对比
4. 与论文结果一键对比

运行方式:
    streamlit run nct_dashboard.py
    
安装依赖:
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
# Streamlit 页面配置
# ============================================================================
import streamlit as st

st.set_page_config(
    page_title="NCT 实时仪表盘",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义 CSS 样式
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
# 侧边栏 - 参数配置
# ============================================================================
st.sidebar.title("⚙️ 参数配置")

# 模型架构参数
st.sidebar.subheader("🏗️ 架构参数")
d_model = st.sidebar.slider("模型维度 (d_model)", 64, 768, 256, step=64)
n_heads = st.sidebar.slider("注意力头数", 4, 16, 8)
n_layers = st.sidebar.slider("Transformer 层数", 2, 8, 4)
gamma_freq = st.sidebar.slider("γ波频率 (Hz)", 30.0, 50.0, 40.0, step=5.0)

# 实验参数
st.sidebar.subheader("🔬 实验参数")
n_cycles = st.sidebar.slider("意识周期数", 5, 100, 20)
noise_level = st.sidebar.slider(
    "输入噪声水平",
    min_value=0.0,
    max_value=0.5,
    value=0.15,
    step=0.05,
    help="控制输入信号的随机噪声强度（越小越平滑）"
)
show_phi = st.sidebar.checkbox("显示 Φ值计算", value=True)
show_fe = st.sidebar.checkbox("显示自由能", value=True)
show_attention = st.sidebar.checkbox("显示注意力热力图", value=True)

# 控制按钮
st.sidebar.subheader("🎮 控制面板")
start_btn = st.sidebar.button("▶️ 开始运行", type="primary")
stop_btn = st.sidebar.button("⏹️ 停止", type="secondary")
reset_btn = st.sidebar.button("🔄 重置", type="secondary")

# 论文数据对比
st.sidebar.subheader("📊 论文数据对比")
show_paper_comparison = st.sidebar.checkbox("显示论文参考值", value=False)

# ============================================================================
# 主界面
# ============================================================================
st.markdown('<p class="main-header">🧠 NCT 实时可视化仪表盘</p>', unsafe_allow_html=True)
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


def plot_metrics_chart(results_df, show_paper=False):
    """绘制指标趋势图"""
    fig = go.Figure()
    
    # Φ值曲线
    fig.add_trace(go.Scatter(
        x=results_df['cycle'],
        y=results_df['phi_value'],
        mode='lines+markers',
        name='Φ值',
        line=dict(color='#FF6B6B', width=3),
        marker=dict(size=8, symbol='circle'),
    ))
    
    # 自由能曲线（双 Y 轴）
    fig.add_trace(go.Scatter(
        x=results_df['cycle'],
        y=results_df['free_energy'],
        mode='lines+markers',
        name='自由能',
        line=dict(color='#4ECDC4', width=3, dash='dot'),
        yaxis='y2',
    ))
    
    # 论文参考值（如果启用）
    if show_paper:
        fig.add_hline(y=0.329, line_dash="dash", line_color="green", 
                     annotation_text="论文Φ值 (d=768)", annotation_position="top right")
        fig.add_hline(y=0.57, line_dash="dash", line_color="orange",
                     annotation_text="论文 FE 终值", annotation_position="bottom right")
    
    fig.update_layout(
        title='📈 意识指标动态变化',
        xaxis_title='周期',
        yaxis_title='Φ值',
        yaxis2=dict(title='自由能', overlaying='y', side='right'),
        legend=dict(x=0, y=1.1, orientation='h'),
        height=400,
        hovermode='x unified'
    )
    
    return fig


def plot_attention_heatmap(manager):
    """绘制注意力权重分布图（多候选竞争版本）"""
    # 从 session_state 中获取真实的注意力权重
    if hasattr(st.session_state, 'last_attention_maps') and st.session_state.last_attention_maps is not None:
        attention_maps = st.session_state.last_attention_maps
        print(f"✅ 使用真实注意力数据，shape: {attention_maps.shape}")
        
        # 获取所有候选的显著性（如果有 workspace_info）
        all_salience = []
        if hasattr(st.session_state, 'last_workspace_info'):
            all_salience = st.session_state.last_workspace_info.get('all_candidates_salience', [])
        
        n_candidates = len(all_salience) if all_salience else attention_maps.shape[3]
        candidate_names = ['整合表征', '视觉特征', '听觉特征', '内感受特征'][:n_candidates]
        
        # 绘制条形图：展示每个候选的注意力权重
        fig = go.Figure()
        
        # 使用所有头的平均注意力权重
        avg_attention = attention_maps[0, :, 0, :].mean(dim=0).cpu().numpy()  # [N_candidates]
        
        fig.add_trace(go.Bar(
            x=candidate_names,
            y=avg_attention.tolist(),
            marker=dict(color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'][:n_candidates]),
            text=[f'{w:.3f}' for w in avg_attention],
            textposition='auto'
        ))
        
        # 标记获胜者
        if hasattr(st.session_state, 'last_workspace_info'):
            winner_idx = st.session_state.last_workspace_info.get('winner_idx', -1)
            if 0 <= winner_idx < n_candidates:
                # 在获胜者上方添加标记
                fig.add_annotation(
                    x=candidate_names[winner_idx],
                    y=max(avg_attention) * 1.1,
                    text='🏆 获胜者',
                    showarrow=False,
                    font=dict(size=16, color='#FFD700')
                )
        
        fig.update_layout(
            title='🎯 多候选竞争 - 注意力权重分布\n<span style="font-size:12px;color:#666">4 个候选在全局工作空间中竞争，胜者获得意识内容广播权</span>',
            xaxis_title='候选内容',
            yaxis_title='注意力权重',
            height=450,
            showlegend=False,
            yaxis=dict(range=[0, max(0.5, max(avg_attention) * 1.3)])
        )
        
        return fig
        
    else:
        # 如果没有真实数据，生成模拟数据
        n_candidates = 4
        candidate_names = ['整合表征', '视觉特征', '听觉特征', '内感受特征']
        # 模拟稀疏注意力
        np.random.seed(42)
        avg_attention = np.random.rand(n_candidates) * 0.3 + 0.2
        avg_attention[0] += 0.2  # 让整合表征略高
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=candidate_names,
            y=avg_attention.tolist(),
            marker=dict(color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']),
            text=[f'{w:.3f}' for w in avg_attention],
            textposition='auto'
        ))
        
        fig.update_layout(
            title='🎯 多候选竞争 - 注意力权重分布（模拟数据）',
            xaxis_title='候选内容',
            yaxis_title='注意力权重',
            height=450,
            showlegend=False,
            yaxis=dict(range=[0, max(0.5, max(avg_attention) * 1.3)])
        )
        
        return fig


def plot_confidence_gauge(confidence):
    """绘制自信度仪表盘"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "🎯 自信度", 'font': {'size': 24}},
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
    
    # 创建管理器
    manager = create_nct_manager()
    manager.start()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 运行指定周期数
    for cycle in range(n_cycles):
        result = run_cycle(manager, cycle + 1)
        st.session_state.results_history.append(result)
        st.session_state.cycle_count += 1
        
        # 更新进度
        progress_bar.progress((cycle + 1) / n_cycles)
        status_text.text(f"运行中 - 周期 {cycle + 1}/{n_cycles}")
        
        # 实时更新图表（每 5 个周期）
        if (cycle + 1) % 5 == 0 or cycle == 0:
            results_df = pd.DataFrame(st.session_state.results_history)
            
            with charts_placeholder.container():
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(plot_metrics_chart(results_df, show_paper_comparison), width="stretch", key=f"metrics_chart_{cycle}")
                
                with col2:
                    if show_attention:
                        st.plotly_chart(plot_attention_heatmap(manager), width="stretch", key=f"attention_heatmap_{cycle}")
                    else:
                        st.plotly_chart(plot_confidence_gauge(result['confidence']), width="stretch", key=f"confidence_gauge_{cycle}")
            
            # 更新指标卡片
            with metrics_placeholder.container():
                col1, col2, col3, col4 = st.columns(4)
                
                latest = results_df.iloc[-1]
                col1.metric("Φ值", f"{latest['phi_value']:.3f}", delta=None)
                col2.metric("自由能", f"{latest['free_energy']:.4f}", delta=f"{latest['free_energy'] - results_df.iloc[0]['free_energy']:.4f}")
                col3.metric("自信度", f"{latest['confidence']:.3f}")
                col4.metric("显著性", f"{latest['salience']:.3f}")
    
    manager.stop()
    progress_bar.empty()
    status_text.empty()
    
    st.success(f"✅ 完成 {n_cycles} 个意识周期！")
    
    # 显示最终数据表格
    with st.expander("📋 查看详细数据"):
        st.dataframe(pd.DataFrame(st.session_state.results_history))
    
    # 导出按钮
    csv = pd.DataFrame(st.session_state.results_history).to_csv(index=False)
    st.download_button(
        label="📥 下载 CSV 数据",
        data=csv,
        file_name=f'nct_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
        mime='text/csv'
    )

elif stop_btn:
    st.session_state.running = False
    st.warning("⏹️ 已停止运行")

elif reset_btn:
    st.session_state.running = False
    st.session_state.results_history = []
    st.session_state.cycle_count = 0
    metrics_placeholder.empty()
    charts_placeholder.empty()
    log_placeholder.empty()
    st.info("🔄 已重置")

# ============================================================================
# 页脚信息
# ============================================================================
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.write("**GitHub:** https://github.com/wyg5208/nct")
with col2:
    st.write("**版本:** v3.1.0")
with col3:
    st.write("**论文:** arXiv:xxxx.xxxxx (即将提交)")
