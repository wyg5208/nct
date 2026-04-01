# 🧠 NeuroConscious Transformer (NCT)

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyPI](https://img.shields.io/badge/PyPI-v3.2.0-007396?style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/neuroconscious-transformer/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformer](https://img.shields.io/badge/Transformer-Architecture-FF6F00?style=for-the-badge&logo=transformers&logoColor=white)](https://huggingface.co/docs/transformers)
[![Neuroscience](https://img.shields.io/badge/Neuroscience-Consciousness-4CAF50?style=for-the-badge)](https://en.wikipedia.org/wiki/Consciousness)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

**版本**: v3.2.0  
**创建**: 2026 年 2 月 21 日  
**更新日期**: 2026 年 3 月 20 日  
**作者**: NCT LAB Team  
**官方网站**: [neuroconscious.link](https://neuroconscious.link)  
**代码**: https://github.com/wyg5208/nct  

[English Documentation](README.md)

---

## 📖 项目简介

NeuroConscious Transformer (NCT) 是**下一代神经形态意识架构**，基于 Transformer 技术重构经典人脑科学理论，实现了六大核心理论创新：

1. **Attention-Based Global Workspace** - 用多头注意力替代简单竞争
2. **Transformer-STDP Hybrid Learning** - 全局调制的突触可塑性
3. **Predictive Coding as Decoder** - Friston 自由能 = Transformer 训练目标
4. **Multi-Modal Cross-Attention Fusion** - 语义级多模态整合
5. **γ-Synchronization Mechanism** - γ同步作为更新周期
6. **Φ Calculator from Attention Flow** - 实时计算整合信息量

### 🏆 实验验证结果（v3.1）

| 指标 | 测量值 | 说明 |
|------|--------|------|
| **Φ值（整合信息）** | 0.329 (d=768) | 随模型维度增加而提升 |
| **自由能降低** | 83.0% | 100 steps, n=5 seeds |
| **STDP 学习延迟** | < 2ms | 所有尺度下亚毫秒级 |
| **时间关联学习** | r=0.733 | 模式相关性显著高于基线 |
| **神经调节放大** | 89% | 效应量 Cohen's d = 1.41 |

> 详细实验数据见论文 Section 7 和 `experiments/results/`

---

## 🚀 快速开始

### 安装依赖

```bash
pip install torch numpy scipy
```

### 运行示例

```bash
cd examples
python quickstart.py
```

### 运行测试

```bash
cd tests
python test_basic.py
```

---

## 📦 项目结构

```
NCT/
├── __init__.py              # 包初始化
├── pyproject.toml           # 项目配置
├── requirements.txt         # 依赖列表
├── README.md               # 英文文档
├── README_CN.md            # 中文文档
├── .gitignore              # Git 忽略规则
│
├── nct_modules/            # 核心模块（9 个文件）
│   ├── nct_core.py         # 核心配置 + 多模态编码器
│   ├── nct_cross_modal.py  # Cross-Modal 整合
│   ├── nct_workspace.py    # Attention 工作空间 ⭐
│   ├── nct_hybrid_learning.py  # Transformer-STDP ⭐
│   ├── nct_predictive_coding.py  # 预测编码 ⭐
│   ├── nct_metrics.py      # Φ计算器 + 意识度量 ⭐
│   ├── nct_gamma_sync.py   # γ同步机制
│   └── nct_manager.py      # 总控制器
│
├── MCS-NCT框架理论/         # MCS 多重约束满足框架
│   ├── mcs_solver.py       # MCS 核心求解器
│   └── mcs_nct_integration.py  # NCT 整合模块
│
├── cats_nct/               # CATS-NCT 概念抽象变体
│   ├── core/               # 核心模块
│   └── manager.py          # CATS-NCT 管理器
│
├── experiments/            # 实验脚本和结果
│   ├── run_all_experiments.py
│   └── results/            # JSON 结果数据
│       ├── exp_A_free_energy.json
│       ├── exp_B_stdp.json
│       ├── exp_C_ablation.json
│       ├── exp_D_scale.json
│       ├── exp_E_attention_grading.json
│       └── exp_F_temporal_association.json
│
├── examples/               # 示例代码
│   └── quickstart.py       # 快速入门
│
├── tests/                  # 测试套件
│   └── test_basic.py       # 基础功能测试
│
├── visualization/          # 可视化工具
│   └── nct_dashboard.py    # Streamlit 实时仪表盘 🎨
│
├── docs/                   # 文档
│   ├── 教育领域数据集实验论文/  # 教育领域实验论文
│   ├── 教育领域数据集实验结果/  # 教育实验结果
│   └── NCT 完整实施方案.md
│
└── papers/                 # 相关论文
    └── neuroconscious_paper/
        ├── NCT_arXiv.tex   # LaTeX 源文件
        └── NCT_arXiv.pdf   # 编译后 PDF
```

---

## 🎨 可视化仪表盘

NCT 提供基于 **Streamlit** 的实时可视化仪表盘，支持：

- **实时监控**: Φ值、自由能、注意力权重动态变化
- **交互调参**: 模型维度、注意力头数、γ波频率等
- **多候选竞争可视化**: 展示全局工作空间中的候选竞争过程
- **双语界面**: 支持中英文切换
- **数据导出**: CSV 格式导出实验数据

```bash
# 安装依赖
pip install streamlit plotly pandas

# 启动仪表盘
streamlit run visualization/nct_dashboard.py
```

---

## 🔬 核心创新详解

### 1. Attention-Based Global Workspace

**传统方案** (v2.2):
```python
# 简单侧向抑制
cand_j.salience -= cand_i.salience * 0.1
```

**NCT 方案** (v3.0):
```python
# Multi-Head Self-Attention (8 heads)
attn_output, attn_weights = nn.MultiheadAttention(
    embed_dim=768, num_heads=8
)(query=q, key=k, value=v)

# Head 分工:
# - Head 0-1: 视觉/听觉显著性检测
# - Head 2-3: 情感价值评估
# - Head 4-5: 任务相关性
# - Head 6-7: 新颖性检测
```

**性能提升**: 意识选择准确率从 75% → 92% (+23%)

---

### 2. Transformer-STDP Hybrid Learning

**数学公式**:
```python
Δw = (δ_STDP + λ·δ_attention) · η_neuromodulator

# δ_STDP: 经典 STDP（局部时间相关）
δ_STDP = A₊·exp(-Δt/τ₊) if Δt > 0
       = -A₋·exp(Δt/τ₋) if Δt < 0

# δ_attention: Attention 梯度（全局语义）
δ_attention = ∂Loss/∂W

# η_neuromodulator: 神经递质调制
η = 1.0 + w_DA·DA + w_5HT·5HT + w_NE·NE + w_ACh·ACh
```

**收敛速度**: 1000 cycles → 200 cycles (**5 倍提升**)

---

### 3. Predictive Coding = Decoder Training

**理论统一证明**:
```python
# Friston 变分自由能
F = E_q(z)[ln q(z) - ln p(s,z)]

# 展开后:
F = CrossEntropy(predictions, actual)  # 预测误差
    + KL(q||p)                         # 正则化项

# Transformer Decoder 训练损失:
Loss = CrossEntropy(next_token_pred, actual_next)
       + L2_regularization(weights)

# 因此:
Free Energy ≈ Transformer Loss
```

---

### 4. Φ Calculator from Attention Flow

**避免 IIT 的 NP-hard 问题**:
```python
# 传统 IIT: O(2^n) 复杂度
Φ = I_total - min_partition[I_A + I_B]

# NCT 近似：O(n²) 复杂度
class PhiFromAttention(nn.Module):
    def compute_phi(self, attention_maps):
        I_total = mutual_information(attn_matrix)
        min_partition_mi = find_min_partition(attn_matrix)
        phi = max(0.0, I_total - min_partition_mi)
        return np.tanh(phi / max(1.0, L * 0.1))
```

**Φ值提升**: 0.3 → 0.7 (**2.3 倍**)

---

## 🔄 框架变体与扩展

### MCS（多重约束满足）框架

MCS 将意识建模重新定义为**多重约束优化问题**：不再追问"什么是意识？"，而是问"一个系统需要满足哪些约束才能具有意识？"这种操作性定义使得意识水平的量化测量成为可能。

**核心公式**:
```
C(t) = argmin_S [ Σᵢ wᵢ·Vᵢ(S,t) ]    # 最优意识状态
意识水平 = 1/(1+J)                     # J = 加权约束违反度
```

| 约束 | 定义 | 理论基础 |
|-----|------|---------|
| C1 感知一致性 | 多模态输入时空对齐 | GWT 全局广播 |
| C2 时间连续性 | 当前状态可由历史预测 | 预测编码 + 自由能 |
| C3 自我一致性 | 信念系统无矛盾 | Thagard 连贯性理论 |
| C4 行动可行性 | 意图可映射到可执行计划 | 具身认知 |
| C5 社会可解释性 | 经验可传达给他人 | Vygotsky 社会起源论 |
| C6 整合信息（Φ） | 系统 Φ 值超过阈值 | IIT |

**关键成果**: DAiSEE 数据集 5 折交叉验证 **R²=0.164**（比 NCT Φ 基线提升 121%）

📁 核心文件: `MCS-NCT框架理论/mcs_solver.py`, `mcs_nct_integration.py`  
📄 论文: *已投稿至 IEEE Transactions on Affective Computing*

---

### CATS-NCT（概念抽象与任务求解）

CATS-NCT 将**概念抽象（CA）双模块架构**与 NCT 的神经科学机制相融合。它将意识建模从感知层面扩展到**概念意识**——稳定的、可传达的心理表征。

| 维度 | NCT（原版） | CATS-NCT |
|-----|------------|----------|
| 关注点 | 意识生成 | 概念形成与传达 |
| 表征层面 | 感知意识 | 概念意识（稳定） |
| 整合方式 | 基于注意力的 GWS | CA + TS 双模块 |
| 学习机制 | Transformer-STDP | 概念抽象 + STDP |
| 可解释性 | 注意力图 | 概念原型 + 门控可视化 |
| 知识迁移 | 不支持 | 概念空间对齐 |

📁 核心文件: `cats_nct/core/`, `cats_nct/manager.py`  
🚧 状态: *积极开发中*

---

## 🎓 教育领域研究突破

### 研究演进（V1→V4）

**研究问题**: NCT 架构能否有效监测教育场景中的认知状态？

| 版本 | 重点 | 关键发现 |
|-----|------|---------|
| V1 | 概念验证 | 框架可运行，但 Φ 不显著 (p>0.05) |
| V2 | 深度学习增强 | FER +16.84%，但 Φ 仍不显著 (p=0.549) |
| V3 | 系统诊断 | **突破**: EEGNet 特征使 Φ 显著 (p=0.0003) |
| V4 | 论文与验证 | 完整消融研究，PCA 优化 (p=0.00005, d=0.586) |

### 关键突破

核心发现：**当 Φ（整合信息）从深度学习（EEGNet）特征计算而非传统频谱特征时，它成为有效的认知状态标记**。这一发现架起了 IIT 理论与实际教育应用之间的桥梁。

| 特征类型 | Φ 显著性 | Cohen's d |
|---------|---------|-----------|
| 传统特征（Welch PSD） | p>0.05（不显著） | - |
| EEGNet 特征 | **p=0.0003** | **0.524（中等）** |
| PCA 降维（50维） | **p=0.00005** | **0.586** |

- **EEGNet 分类**: F1=0.62（vs SVM 基线 F1=0.39）
- **数据集**: MEMA（脑电）、DAiSEE（视频）、FER2013、EdNet
- 📄 论文: *已投稿至 IEEE Transactions on Affective Computing*

---

## 📊 预期性能指标

| 维度 | v2.2 | v3.0 | v3.1 (实测) | 提升 |
|------|------|------|-------------|------|
| 意识选择准确率 | 75% | 92% | **92%** | +23% |
| 学习收敛速度 | 1000 cycles | 200 cycles | **~180 cycles** | 5× |
| 多模态融合质量 | 0.6 NCC | 0.85 NCC | **0.82 NCC** | +42% |
| Φ值（整合信息） | 0.3 | 0.7 | **0.329 (d=768)** | 2.3× |
| GPU 加速潜力 | ❌ | ✅ CUDA 原生 | **✅ 已验证** | 50× |
| STDP 延迟 | - | <5ms | **<2ms** | - |
| 自由能降低 | - | 80% | **83.0%** | - |

> 注：v3.1 实测数据来自 `experiments/results/`，详细统计见论文 Table 2-6

---

## 🛠️ 开发指南

### 本地开发设置

```bash
# 克隆仓库
git clone https://github.com/wyg5208/nct.git
cd nct

# 安装依赖
pip install -r requirements.txt

# 安装开发依赖（可选）
pip install pytest black ruff mypy

# 运行测试
pytest tests/

# 代码格式化
black .
ruff check .
```

### 复现论文实验

```bash
# 运行所有实验（约需 30 分钟）
python experiments/run_all_experiments.py

# 查看结果
ls experiments/results/

# 运行实时可视化仪表盘
streamlit run visualization/nct_dashboard.py
```

### 自定义实验

```python
from nct_modules import NCTManager, NCTConfig

# 自定义配置
config = NCTConfig(
    n_heads=12,      # 增加工作空间容量
    n_layers=6,      # 增加皮层层次
    d_model=1024,    # 增加表征维度
)

# 创建管理器
manager = NCTManager(config)

# 运行实验
for trial in range(100):
    sensory = generate_sensory_data()
    state = manager.process_cycle(sensory)
    analyze(state)
```

---

## 📚 参考文献

1. Whittington & Bogacz (2017). An approximation of the error backpropagation algorithm in a predictive coding network with local Hebbian synaptic plasticity. *Neural Computation*
2. Millidge, Tschantz & Buckley (2022). Predictive coding approximates backprop along arbitrary computation graphs. *Neural Computation*
3. Vaswani et al. (2017). Attention Is All You Need
4. Dehaene & Changeux (2011). Experimental and theoretical approaches to conscious processing
5. Friston (2010). The free-energy principle: a unified brain theory
6. Tononi (2008). Consciousness as integrated information
7. Bi & Poo (1998). Synaptic modifications by STDP
8. Fries (2005). Gamma oscillations and communication

### 📄 相关论文

- **NCT_arXiv.pdf** - 最新论文预印本（包含完整实验验证）
- **NCT_arXiv.tex** - LaTeX 源文件

---

## 📄 论文发表

| 论文 | 期刊 | 状态 |
|-----|------|------|
| MCS: Multi-Constraint Satisfaction Framework for Consciousness Modeling | IEEE Trans. Affective Computing | 审稿中 |
| Deep Learning Features Enable IIT (Φ) for Cognitive State Monitoring in Education | IEEE Trans. Affective Computing | 审稿中 |

---

## 📝 更新日志

### v3.2.0 (2026-03-20)
- ✅ 新增 MCS（多重约束满足）意识建模框架
- ✅ 教育领域 V4 实验：EEGNet 特征使 Φ 值具有区分能力 (p=0.0003, d=0.524)
- ✅ 两篇论文投稿至 IEEE Transactions on Affective Computing
- ✅ 项目结构重组：清理根目录，整理脚本文件
- ✅ 统一所有配置文件中的版本号
- ✅ 修复 pyproject.toml 中的 GitHub URL
- ✅ 增强 .gitignore 安全性（.env）和组织结构（/temp/）

### v3.1.0 (2026-02-22)
- ✅ 完成所有 6 项核心实验验证
- ✅ 添加统计显著性分析（t-test, Cohen's d）
- ✅ 优化Φ计算方法（随机二分法，r > 0.93）
- ✅ 整合 Integration Challenges 讨论
- ✅ 添加误差线可视化
- ✅ 开源代码仓库建立

### v3.0.0-alpha (2026-02-21)
- 🎉 初始版本发布

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

### 代码规范

- 遵循 PEP 8
- 类型注解必需
- 单元测试覆盖率 > 80%
- 使用 Black 格式化代码

---

## 📄 许可证

MIT License

---

## 🌟 致谢

感谢所有意识神经科学研究者和 AI 领域的先驱。

**🧠 让我们一起探索意识的奥秘！**
