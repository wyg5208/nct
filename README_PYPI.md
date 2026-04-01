# 🧠 NeuroConscious Transformer (NCT)

[![PyPI](https://img.shields.io/pypi/v/neuroconscious-transformer?style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/neuroconscious-transformer/)
[![Python](https://img.shields.io/pypi/pyversions/neuroconscious-transformer?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

**Version**: v3.2.0 | **版本**: v3.2.0 
**Author**: NCT LAB Team  

---

## 📦 Installation / 安装

### Basic Installation / 基础安装

```bash
pip install neuroconscious-transformer
```

### Complete Installation (with Dashboard) / 完整安装（含可视化工具）

```bash
pip install neuroconscious-transformer[dashboard]
```

### Verify Installation / 验证安装

```bash
python -c "from nct_modules import NCTManager; print('✓ Installation successful')"
```

### Requirements / 依赖要求

- Python 3.9+
- PyTorch 2.0+
- NumPy 1.24+
- SciPy 1.10+

---

## 🚀 Quick Start / 快速开始

### Example 1: Basic Usage / 基础使用

```python
from nct_modules import NCTManager, NCTConfig
import numpy as np

# Create configuration / 创建配置
config = NCTConfig(
    n_heads=8,           # Number of attention heads / 注意力头数
    n_layers=4,          # Number of Transformer layers / Transformer 层数
    d_model=768,         # Representation dimension / 表征维度
    gamma_freq=40.0,     # Gamma wave frequency (Hz) / γ波频率
)

# Initialize manager / 初始化管理器
manager = NCTManager(config)
manager.start()

# Process consciousness cycles / 处理意识周期
for cycle in range(10):
    # Prepare sensory input / 准备感觉输入
    sensory_data = {
        'visual': np.random.randn(28, 28).astype(np.float32),
        'auditory': np.random.randn(10, 10).astype(np.float32),
        'interoceptive': np.random.randn(10).astype(np.float32),
    }
    
    # Process cycle / 处理周期
    state = manager.process_cycle(sensory_data)
    
    # View results / 查看结果
    print(f"Cycle {cycle + 1}:")
    print(f"  Φ Value (Integrated Information): {state.consciousness_metrics.get('phi_value', 0):.3f}")
    print(f"  Free Energy (Prediction Error): {state.self_representation['free_energy']:.4f}")

# Stop / 停止
manager.stop()
```

### Example 2: Multimodal Encoding / 多模态编码

```python
from nct_modules import MultiModalEncoder
import torch

encoder = MultiModalEncoder(
    visual_embed_dim=256,
    audio_embed_dim=256,
    intero_embed_dim=256,
)

# Prepare inputs / 准备输入
visual_input = torch.randn(1, 3, 28, 28)  # RGB image / RGB 图像
audio_input = torch.randn(1, 10, 10)       # Audio spectrogram / 音频频谱
intero_input = torch.randn(1, 10)          # Interoceptive signals / 内感受信号

sensory_tensors = {
    'visual': visual_input,
    'auditory': audio_input,
    'interoceptive': intero_input,
}

# Encode / 编码
embeddings = encoder(sensory_tensors)
print(f"Visual embedding shape: {embeddings['visual'].shape}")  # [1, 768]
```

### Example 3: Φ Value Computation / Φ值计算

```python
from nct_modules import PhiFromAttention
import torch

phi_calc = PhiFromAttention()

# Simulate attention maps / 模拟注意力图
attention_maps = torch.rand(8, 768, 768)  # [heads, seq_len, seq_len]

# Compute Φ value / 计算 Φ值
phi_value = phi_calc.compute_phi(attention_maps)
print(f"Integrated Information Φ: {phi_value:.3f}")
print(f"Consciousness level: {'High' if phi_value > 0.5 else 'Medium' if phi_value > 0.2 else 'Low'}")
```

### Example 4: Transformer-STDP Hybrid Learning / 混合学习

```python
from nct_modules import TransformerSTDP, STDPEvent
import torch

stdp_learner = TransformerSTDP(n_neurons=768, d_model=768)

# Create STDP event / 创建 STDP 事件
event = STDPEvent(
    pre_idx=10,      # Pre-synaptic neuron index / 突触前神经元索引
    post_idx=20,     # Post-synaptic neuron index / 突触后神经元索引
    delta_t=0.015,   # Time difference (seconds) / 时间差
)

# Update synapse / 更新突触
synaptic_update = stdp_learner.update(event)
print(f"Synaptic strength change: Δw = {synaptic_update.delta_w:.6f}")
```

### Example 5: Complete Experiment / 完整实验

```python
"""
NCT Consciousness Computation Experiment
展示：Multimodal Fusion + Φ Monitoring + STDP Learning
"""
from nct_modules import NCTManager, NCTConfig
import numpy as np
import matplotlib.pyplot as plt

# Configuration / 配置
config = NCTConfig(
    n_heads=8,
    n_layers=4,
    d_model=768,
    stdp_learning_rate=0.01,
)

manager = NCTManager(config)
manager.start()

# Record data / 记录数据
phi_values = []
free_energies = []

# Run 100 cycles / 运行 100 个周期
for cycle in range(100):
    # Generate meaningful sensory input (with patterns) / 生成有意义的感觉输入
    visual = np.sin(np.linspace(0, cycle * 0.1, 28 * 28)).reshape(28, 28).astype(np.float32)
    audio = np.cos(np.linspace(0, cycle * 0.05, 10 * 10)).reshape(10, 10).astype(np.float32)
    intero = np.random.randn(10).astype(np.float32) * 0.5
    
    sensory_data = {
        'visual': visual,
        'auditory': audio,
        'interoceptive': intero,
    }
    
    # Process / 处理
    state = manager.process_cycle(sensory_data)
    
    # Record / 记录
    phi_values.append(state.consciousness_metrics.get('phi_value', 0))
    free_energies.append(state.self_representation['free_energy'])
    
    # Print every 10 cycles / 每 10 个周期打印
    if (cycle + 1) % 10 == 0:
        print(f"Cycle {cycle + 1}/100 | Φ={phi_values[-1]:.3f} | FE={free_energies[-1]:.4f}")

manager.stop()

# Visualization / 可视化
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(phi_values)
plt.title('Φ Value (Integrated Information)')
plt.xlabel('Cycle')
plt.ylabel('Φ')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(free_energies)
plt.title('Free Energy (Prediction Error)')
plt.xlabel('Cycle')
plt.ylabel('FE')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('nct_experiment_results.png', dpi=300)
plt.show()

print("\n✅ Experiment complete! Results saved to nct_experiment_results.png")
```

---

## 🔧 Core API Reference / 核心 API 速查

### NCTConfig - Configuration Class / 配置类

```python
from nct_modules import NCTConfig

config = NCTConfig(
    # Core parameters / 核心参数
    n_heads=8,                    # Attention heads (workspace capacity) / 注意力头数
    n_layers=4,                   # Transformer layers / Transformer 层数
    d_model=768,                  # Representation dimension / 表征维度
    dim_ff=3072,                  # Feed-forward network dimension / 前馈网络维度
    
    # Neuroscience parameters / 神经科学参数
    gamma_freq=40.0,              # Gamma wave frequency (Hz) / γ波频率
    stdp_learning_rate=0.01,      # STDP learning rate / STDP 学习率
    attention_modulation_lambda=0.1,  # Attention modulation coefficient / 注意力调制系数
)
```

### NCTManager - Manager Class / 管理器类

```python
from nct_modules import NCTManager

# Initialize / 初始化
manager = NCTManager(config)

# Start system / 启动系统
manager.start()

# Process one consciousness cycle / 处理一个意识周期
state = manager.process_cycle(sensory_data)

# Stop system / 停止系统
manager.stop()

# Get statistics / 获取统计信息
stats = manager.get_stats()

# Save model / 保存模型
torch.save(manager.state_dict(), 'model.pth')

# Load model / 加载模型
manager.load_state_dict(torch.load('model.pth'))
```

### ConsciousnessState - State Object / 状态对象

```python
# Access consciousness metrics / 访问意识度量
state.awareness_level              # Consciousness level (low/moderate/high) / 意识水平
state.consciousness_metrics        # Metrics dictionary (includes Φ value) / 意识度量字典
state.self_representation          # Self-representation (free energy, confidence) / 自我表征
state.workspace_content            # Global workspace content / 全局工作空间内容

# Example usage / 示例用法
print(f"Consciousness level: {state.awareness_level}")
print(f"Φ value: {state.consciousness_metrics['phi_value']:.3f}")
print(f"Free energy: {state.self_representation['free_energy']:.4f}")
```

### Other Core Components / 其他核心组件

#### MultiModalEncoder / 多模态编码器
```python
from nct_modules import MultiModalEncoder

encoder = MultiModalEncoder(
    visual_embed_dim=256,
    audio_embed_dim=256,
    intero_embed_dim=256,
)

embeddings = encoder(sensory_tensors)
```

#### PhiFromAttention / Φ值计算器
```python
from nct_modules import PhiFromAttention

phi_calc = PhiFromAttention()
phi_value = phi_calc.compute_phi(attention_maps)
```

#### TransformerSTDP / 混合学习
```python
from nct_modules import TransformerSTDP, STDPEvent

stdp = TransformerSTDP(n_neurons=768, d_model=768)
event = STDPEvent(pre_idx=10, post_idx=20, delta_t=0.015)
update = stdp.update(event)
```

#### NCTWorkspace / 全局工作空间
```python
from nct_modules import NCTWorkspace

workspace = NCTWorkspace(n_heads=8, d_model=768)
global_content = workspace(attention_maps, query)
```

---

## 🎨 Advanced Features / 高级功能

### GPU Acceleration / GPU 加速

```python
import torch
torch.set_default_device('cuda')  # Automatically use GPU / 自动使用 GPU

# All computations will now run on GPU / 所有计算将在 GPU 上运行
config = NCTConfig()
manager = NCTManager(config)
```

### Custom Sensory Inputs / 自定义感觉输入

```python
# Supports arbitrary shapes (automatically adapts) / 支持任意形状（自动适配）
sensory_data = {
    'visual': your_image,           # [H, W] or [C, H, W]
    'auditory': your_audio,         # [T, F]
    'interoceptive': your_signals,  # [N]
}
```

### Model Checkpointing / 模型检查点

```python
import torch
from nct_modules import NCTManager, NCTConfig

# Save checkpoint / 保存检查点
checkpoint = {
    'config': config.to_dict(),
    'model_state': manager.state_dict(),
    'metrics': manager.get_stats(),
    'cycle': current_cycle,
}
torch.save(checkpoint, 'nct_checkpoint.pth')

# Load checkpoint / 加载检查点
loaded = torch.load('nct_checkpoint.pth')
loaded_config = NCTConfig.from_dict(loaded['config'])
loaded_manager = NCTManager(loaded_config)
loaded_manager.load_state_dict(loaded['model_state'])

print(f"✓ Checkpoint loaded at cycle {loaded['cycle']}")
```

### Preset Configurations / 预设配置

```python
# Large-scale configuration (research) / 大规模配置（研究用）
large_config = NCTConfig(
    n_heads=12,
    n_layers=6,
    d_model=1024,
    dim_ff=4096,
)

# Lightweight configuration (real-time) / 轻量级配置（实时应用）
small_config = NCTConfig(
    n_heads=4,
    n_layers=2,
    d_model=256,
    gamma_freq=30.0,
)

# Temporal learning configuration / 时序关联学习配置
temporal_config = NCTConfig(
    n_heads=8,
    n_layers=4,
    d_model=512,
    stdp_learning_rate=0.05,
    attention_modulation_lambda=0.2,
)
```

---

## 📊 Common Issues / 常见问题

### Q1: How to improve Φ value? / 如何提高 Φ值？

- Increase `d_model` (representation dimension) / 增加 `d_model`（表征维度）
- Increase `n_heads` (attention head count) / 增加 `n_heads`（注意力头数）
- Use more meaningful sensory inputs (patterned data) / 使用更有意义的感觉输入

### Q2: How to use custom modalities? / 如何使用自定义模态？

```python
# Add any modality you want / 添加任意模态
sensory_data = {
    'custom_modality_1': tensor1,
    'custom_modality_2': tensor2,
}

# The encoder will automatically adapt / 编码器会自动适配
```

### Q3: Dashboard not launching? / Dashboard 无法启动？

```bash
# Install dashboard dependencies / 安装 Dashboard 依赖
pip install neuroconscious-transformer[dashboard]

# Or manually install / 或手动安装
pip install streamlit plotly pandas

# Launch / 启动
nct-dashboard
```

---

## 📚 Further Reading / 延伸阅读

### Technical Blog Series / 技术博客系列

- [Consciousness: From Philosophy to Engineering](https://blog.csdn.net/yweng18/article/details/158291728?spm=1001.2014.3001.5501)
- [Attention as Global Workspace](https://blog.csdn.net/yweng18/article/details/158291816?spm=1001.2014.3001.5501)
- [Transformer-STDP Integration](https://blog.csdn.net/yweng18/article/details/158320528?spm=1001.2014.3001.5501)
- [Predictive Coding & Free Energy](https://blog.csdn.net/yweng18/article/details/158321001?spm=1001.2014.3001.5501)
- [大白话说清 NCT——意识是如何“算“出来的？](https://blog.csdn.net/yweng18/article/details/158322623?spm=1001.2014.3001.5501)

### Academic Paper / 学术论文

- arXiv preprint: [NeuroConscious Transformer](https://arxiv.org/) (coming soon)
- Technical report with full experiments and ablation studies

### GitHub Repository / GitHub 仓库

For source code, examples, and development documentation:  
https://github.com/wyg5208/NCT

---

## 👨‍💻 Author & License / 作者与许可

**Author**: NCT LAB Team  

**License / 许可证**: MIT License  
**Homepage / 官方网站**: https://neuroconscious.link
---

## 🙏 Acknowledgments / 致谢

This project integrates insights from:
- Transformer architecture (Vaswani et al., 2017)
- STDP mechanisms (Bi & Poo, 1998)
- Predictive Coding theory (Rao & Ballard, 1999)
- Integrated Information Theory (Tononi et al.)
- Global Workspace Theory (Baars, Dehaene)

本项目融合了以下理论洞见：
- Transformer 架构（Vaswani 等，2017）
- STDP 机制（Bi & Poo, 1998）
- 预测编码理论（Rao & Ballard, 1999）
- 整合信息论（Tononi 等）
- 全局工作空间理论（Baars, Dehaene）

---

## 📬 Contact / 联系

- **GitHub Issues**: https://github.com/wyg5208/NCT/issues
- **PyPI**: https://pypi.org/project/neuroconscious-transformer/

**Welcome to join the NeuroConscious community! 🧠✨**  
**欢迎加入 NeuroConscious 社区！🧠✨**
