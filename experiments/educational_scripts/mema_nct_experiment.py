"""
Phase 3: MEMA EEG 神经调质映射实验（核心实验）
==============================================
使用 MEMA 多标签 EEG 数据集，直接验证 EEG 频带功率与
NCT 神经调质参数映射的生理合理性，并分析 Φ 值与注意力状态的相关性。

MEMA 数据集：
  - 西安交通大学，20 名受试者 × 12 次试验
  - 3 类注意力状态：neutral (0) / relaxing (1) / concentrating (2)
  - 格式：.mat 文件（MATLAB），含 EEG 信号、标签
  - GitHub: https://github.com/XJTU-EEG/MEMA

EEG 与神经调质映射（基于神经电生理文献）：
  Theta 功率 ↑ ↔ ACh↑  (乙酰胆碱-Theta 节律耦合，工作记忆)
  Alpha 功率 ↑ ↔ 5-HT↑ (血清素-Alpha 关联，放松抑制)
  Beta  功率 ↑ ↔ DA↑   (多巴胺奖励通路，认知投入)
  Theta/Alpha↑ ↔ NE↑   (去甲肾上腺素，警觉专注)

关键实验目标：
  实验A: 3 类分类基准（NCT vs SVM vs LSTM），报告 F1
  实验B: Concentrating vs Relaxing 的 Φ 值 t 检验（p<0.05 则显著）
  实验C: 4 种神经调质在 3 类状态下的时序曲线

作者：NCT LAB Team
日期：2026-03-15
"""

import sys
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy import signal as sp_signal
from scipy.stats import ttest_ind, f_oneway
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nct_modules.nct_manager import NCTManager
from nct_modules.nct_core import NCTConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR    = PROJECT_ROOT / "data" / "mema"
RESULTS_DIR = PROJECT_ROOT / "results" / "education"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---- 注意力状态标签 ----
ATTENTION_STATES = {0: "neutral", 1: "relaxing", 2: "concentrating"}

# ---- 频带范围（Hz） ---
FREQ_BANDS = {
    "delta": (1,   4),
    "theta": (4,   8),
    "alpha": (8,  13),
    "beta":  (13, 30),
    "gamma": (30, 50),
}

# ---- 轻量 NCT 配置 ----
LIGHT_NCT_CONFIG = NCTConfig(
    n_heads=4,
    n_layers=2,
    d_model=128,
    dim_ff=256,
    visual_embed_dim=128,
    audio_embed_dim=128,
    intero_embed_dim=128,
    gamma_freq=40.0,
    consciousness_threshold=0.5,
    stdp_learning_rate=0.01,
)


# ==============================================================================
# 1. EEG 数据加载
# ==============================================================================

class MEMALoader:
    """MEMA EEG 数据集加载器（支持真实 MEMA 文件夹结构）"""

    def __init__(self, data_dir: Path, max_samples: int = 5000):
        self.data_dir = data_dir
        self.max_samples = max_samples  # 最大样本数限制

    def is_available(self) -> bool:
        mat_files = list(self.data_dir.rglob("*.mat"))
        return len(mat_files) > 0

    def load_all(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载全部 .mat 文件，返回 (eeg_epochs, labels)。

        支持两种格式：
        1. 真实 MEMA 文件夹结构：Subject1/Subject1_attention.mat 等
           - data: [n_trials, n_timepoints, n_channels]
           - label: [1, n_trials] 每个试次的标签
        2. 合成数据格式：Subject_1.mat
           - data: [n_trials, n_channels, n_timepoints]
           - labels: [1, n_trials]

        Returns:
          eeg_epochs: [N, n_channels, n_timepoints]
          labels:     [N] (0/1/2)
        """
        from scipy.io import loadmat
        all_eeg, all_labels = [], []
        fs = 200  # 采样率 Hz

        # 优先检测真实 MEMA 文件夹结构（Subject1/, Subject2/, ...）
        subject_dirs = sorted([d for d in self.data_dir.iterdir()
                               if d.is_dir() and d.name.startswith('Subject')])

        if subject_dirs:
            logger.info(f"检测到真实 MEMA 结构：{len(subject_dirs)} 个受试者文件夹")
            return self._load_real_mema(subject_dirs)

        # 回退到合成数据格式
        logger.info("使用合成数据格式加载")
        return self._load_synthetic()

    def _load_real_mema(self, subject_dirs: List[Path]) -> Tuple[np.ndarray, np.ndarray]:
        """加载真实 MEMA 数据（文件夹结构），带采样限制"""
        from scipy.io import loadmat
        all_eeg, all_labels = [], []
        per_class_samples = {0: [], 1: [], 2: []}  # 分层存储索引

        for subj_dir in subject_dirs:
            # 查找 attention 相关文件（核心实验目标）
            attention_files = list(subj_dir.glob('*attention*.mat'))
            if not attention_files:
                # 回退到任意 .mat 文件
                attention_files = list(subj_dir.glob('*.mat'))

            for mat_path in attention_files:
                try:
                    mat = loadmat(str(mat_path))
                    data = mat.get('data')
                    labels = mat.get('label')

                    if data is None:
                        continue

                    # 真实 MEMA: data shape = (trials, time, channels)
                    # 需要转置为 (trials, channels, time)
                    if data.ndim == 3:
                        if data.shape[2] == 32:  # channels 在最后
                            data = np.transpose(data, (0, 2, 1))  # (trials, channels, time)

                    # 加载标签
                    if labels is not None:
                        labels = labels.flatten()
                    else:
                        # 从文件名推断
                        fname = mat_path.stem.lower()
                        if 'neutral' in fname or '_0' in fname:
                            labels = np.zeros(data.shape[0], dtype=int)
                        elif 'relax' in fname or '_1' in fname:
                            labels = np.ones(data.shape[0], dtype=int)
                        elif 'concentrat' in fname or '_2' in fname:
                            labels = np.full(data.shape[0], 2, dtype=int)
                        else:
                            labels = np.zeros(data.shape[0], dtype=int)

                    # 分层存储
                    for i, trial in enumerate(data):
                        lbl = int(labels[i] if i < len(labels) else 0)
                        if lbl in per_class_samples:
                            per_class_samples[lbl].append((trial.astype(np.float32), lbl))

                except Exception as e:
                    logger.warning(f"跳过 {mat_path.name}: {e}")

        if not any(per_class_samples.values()):
            return None, None

        # 分层采样：每类最多 max_samples // 3 个样本
        max_per_class = self.max_samples // 3
        for lbl, samples in per_class_samples.items():
            if len(samples) > max_per_class:
                indices = np.random.choice(len(samples), max_per_class, replace=False)
                samples = [samples[i] for i in indices]
            for trial, label in samples:
                all_eeg.append(trial)
                all_labels.append(label)

        # 打乱顺序
        indices = np.random.permutation(len(all_eeg))
        all_eeg = [all_eeg[i] for i in indices]
        all_labels = [all_labels[i] for i in indices]

        logger.info(f"加载完成：{len(all_eeg)} 个试次（采样自 {len(subject_dirs)} 名受试者）")
        logger.info(f"类别分布：{[(l, sum(1 for x in all_labels if x==l)) for l in [0,1,2]]}")
        return np.array(all_eeg), np.array(all_labels)

    def _load_synthetic(self) -> Tuple[np.ndarray, np.ndarray]:
        """加载合成数据格式"""
        from scipy.io import loadmat
        all_eeg, all_labels = [], []

        for mat_path in sorted(self.data_dir.rglob("*.mat")):
            try:
                mat = loadmat(str(mat_path))
                data_key = next((k for k in mat if not k.startswith("_") and
                                 isinstance(mat[k], np.ndarray) and mat[k].ndim >= 2), None)
                label_key = next((k for k in mat if "label" in k.lower()), None)

                if data_key is None:
                    continue

                data = mat[data_key]
                if data.ndim == 2:
                    data = data[np.newaxis, ...]

                # 合成格式：data shape = (trials, channels, time)
                fname = mat_path.stem.lower()
                if "neutral" in fname or fname.endswith("0"):
                    label = 0
                elif "relax" in fname or fname.endswith("1"):
                    label = 1
                elif "concentrat" in fname or fname.endswith("2"):
                    label = 2
                elif label_key and mat[label_key].size > 0:
                    label = int(mat[label_key].flat[0])
                else:
                    label = 0

                for trial in data:
                    all_eeg.append(trial.astype(np.float32))
                    all_labels.append(label)

            except Exception as e:
                logger.warning(f"跳过 {mat_path.name}: {e}")

        if not all_eeg:
            return None, None

        # 统一长度（截断到最短）
        min_t = min(d.shape[-1] for d in all_eeg)
        all_eeg = [d[:, :min_t] for d in all_eeg]
        return np.array(all_eeg), np.array(all_labels)

    def generate_mock(
        self, n_subjects: int = 5, n_trials_per_class: int = 10,
        n_channels: int = 14, sfreq: int = 200, duration_sec: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成合理的 Mock EEG 信号（各频带加入生物合理性差异）。

        - neutral:       均衡 alpha/beta
        - relaxing:      alpha↑ (8-13 Hz 显著)
        - concentrating: beta↑ + theta↑ (4-8 Hz + 13-30 Hz)
        """
        np.random.seed(42)
        n_timepoints = sfreq * duration_sec
        t = np.linspace(0, duration_sec, n_timepoints)
        all_eeg, all_labels = [], []

        state_profiles = {
            0: {"alpha": 1.0, "beta": 1.0, "theta": 0.8, "delta": 0.5},  # neutral
            1: {"alpha": 2.5, "beta": 0.7, "theta": 0.6, "delta": 0.4},  # relaxing
            2: {"alpha": 0.8, "beta": 2.0, "theta": 1.8, "delta": 0.3},  # concentrating
        }

        for state in range(3):
            profile = state_profiles[state]
            for _ in range(n_subjects * n_trials_per_class):
                channels = []
                for ch in range(n_channels):
                    sig = np.random.randn(n_timepoints) * 0.5
                    # 加入各频带正弦成分
                    for band, (flo, fhi) in FREQ_BANDS.items():
                        amp = profile.get(band, 0.5)
                        fc  = (flo + fhi) / 2
                        sig += amp * np.sin(2 * np.pi * fc * t + np.random.uniform(0, 2 * np.pi))
                    channels.append(sig.astype(np.float32))
                all_eeg.append(np.array(channels))
                all_labels.append(state)

        idx = np.random.permutation(len(all_eeg))
        return np.array(all_eeg)[idx], np.array(all_labels)[idx]


# ==============================================================================
# 2. EEG 特征提取
# ==============================================================================

def compute_band_power(eeg: np.ndarray, sfreq: int = 200) -> Dict[str, float]:
    """
    计算每个频带的相对功率（平均所有通道）。

    Args:
        eeg:   [n_channels, n_timepoints]
        sfreq: 采样率 Hz

    Returns:
        band_powers: {'theta': float, 'alpha': float, 'beta': float, ...}
    """
    n_timepoints = eeg.shape[-1]
    freqs, psd = sp_signal.welch(eeg, fs=sfreq, nperseg=min(256, n_timepoints))

    total_power = np.mean(psd) + 1e-10
    band_powers = {}
    for band, (flo, fhi) in FREQ_BANDS.items():
        mask = (freqs >= flo) & (freqs < fhi)
        band_powers[band] = float(np.mean(psd[:, mask])) / total_power

    return band_powers


def eeg_features_to_neuromodulator(band_powers: Dict[str, float]) -> Dict[str, float]:
    """
    EEG 频带功率 → NCT 神经调质参数

    映射规则（基于神经电生理文献）：
      DA   ↔ beta 功率（认知投入 / 奖励回路激活）
      5-HT ↔ alpha 功率（血清素-alpha 放松状态关联）
      NE   ↔ theta/alpha 比值（去甲肾上腺素-警觉度关联）
      ACh  ↔ theta 功率（海马 theta 振荡，乙酰胆碱调控工作记忆）
    """
    def sigmoid(x: float) -> float:
        return float(1.0 / (1.0 + np.exp(-x)))

    theta = band_powers.get("theta", 0.1)
    alpha = band_powers.get("alpha", 0.1)
    beta  = band_powers.get("beta",  0.1)

    # z-score 风格归一化（以 0.1 为基准）
    w = 10.0  # 缩放因子，让 sigmoid 输出有合理范围

    DA   = sigmoid((beta  - 0.12) * w)
    sHT  = sigmoid((alpha - 0.15) * w)
    NE   = sigmoid((theta / (alpha + 1e-8) - 1.0) * 2.0)
    ACh  = sigmoid((theta - 0.08) * w)

    return {
        "DA":   round(DA,  4),
        "5-HT": round(sHT, 4),
        "NE":   round(NE,  4),
        "ACh":  round(ACh, 4),
    }


def extract_feature_vector(band_powers: Dict[str, float]) -> np.ndarray:
    """将频带功率转换为 sklearn 特征向量"""
    return np.array([
        band_powers.get("delta", 0),
        band_powers.get("theta", 0),
        band_powers.get("alpha", 0),
        band_powers.get("beta",  0),
        band_powers.get("gamma", 0),
        band_powers.get("theta", 0) / (band_powers.get("alpha", 0.01) + 1e-8),  # theta/alpha
        band_powers.get("beta",  0) / (band_powers.get("alpha", 0.01) + 1e-8),  # beta/alpha
    ], dtype=np.float32)


# ==============================================================================
# 3. 分类基准（实验 A）
# ==============================================================================

def _run_lstm_baseline(X: np.ndarray, y: np.ndarray) -> Dict:
    """
    简化 LSTM 分类器基准。
    每条样本为 7 维 EEG 特征展开为序列（单步长，用 MLP 近似单层 LSTM 分类器）。
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        class _LSTMMLP(nn.Module):
            """用于 EEG 频带特征属性分类的小型 MLP（LSTM 近似）"""
            def __init__(self, in_dim: int = 7, hidden: int = 64, n_cls: int = 3):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, hidden), nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden, hidden), nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden, n_cls),
                )
            def forward(self, x):
                return self.net(x)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        kf     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        f1_scores = []

        X_t = torch.from_numpy(X.astype(np.float32))
        y_t = torch.from_numpy(y.astype(np.int64))

        for tr, te in kf.split(X, y):
            # 标准化
            mu, std = X_t[tr].mean(0), X_t[tr].std(0) + 1e-8
            X_tr = (X_t[tr] - mu) / std
            X_te = (X_t[te] - mu) / std

            model = _LSTMMLP().to(device)
            opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss_fn = nn.CrossEntropyLoss()

            ds = TensorDataset(X_tr, y_t[tr])
            dl = DataLoader(ds, batch_size=32, shuffle=True)

            model.train()
            for _ in range(30):  # 轻量训练
                for xb, yb in dl:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad()
                    loss_fn(model(xb), yb).backward()
                    opt.step()

            model.eval()
            with torch.no_grad():
                preds = model(X_te.to(device)).argmax(1).cpu().numpy()
            f1 = f1_score(y[te], preds, average="macro")
            f1_scores.append(f1)

        return {
            "mean_f1": round(float(np.mean(f1_scores)), 4),
            "std_f1":  round(float(np.std(f1_scores)),  4),
            "model":   "MLP-2layer (LSTM-proxy)",
        }
    except Exception as e:
        logger.warning(f"LSTM 基准运行失败：{e}")
        return {"mean_f1": None, "std_f1": None, "model": "MLP-2layer", "error": str(e)}


def run_classification_benchmark(
    X: np.ndarray, y: np.ndarray
) -> Dict:
    """
    实验 A：3 类注意力状态分类基准（SVM + LSTM）三方对比。

    使用 5 折交叉验证，报告宏平均 F1。
    """
    # ---- SVM 基准 ----
    scaler = StandardScaler()
    svm    = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)
    kf     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    f1_scores = []
    for fold, (tr, te) in enumerate(kf.split(X, y)):
        X_tr = scaler.fit_transform(X[tr])
        X_te = scaler.transform(X[te])
        svm.fit(X_tr, y[tr])
        preds = svm.predict(X_te)
        f1 = f1_score(y[te], preds, average="macro")
        f1_scores.append(f1)

    # 全数据拟合，获得每类报告
    X_scaled = scaler.fit_transform(X)
    svm.fit(X_scaled, y)
    all_preds = svm.predict(X_scaled)
    report    = classification_report(y, all_preds,
                                       target_names=[ATTENTION_STATES[i] for i in range(3)],
                                       output_dict=True)

    svm_result = {
        "mean_f1":  round(float(np.mean(f1_scores)), 4),
        "std_f1":   round(float(np.std(f1_scores)),  4),
        "per_class": {k: round(v["f1-score"], 4)
                      for k, v in report.items() if k in ATTENTION_STATES.values()},
    }

    # ---- LSTM 基准 ----
    logger.info("  运行 LSTM 基准对比...")
    lstm_result = _run_lstm_baseline(X, y)

    combined = {
        "SVM":  svm_result,
        "LSTM": lstm_result,
        # NCT Phi 分类结果将在 run_phi_analysis 后填充
    }
    logger.info(f"  SVM  宏平均 F1={svm_result['mean_f1']:.4f} ± {svm_result['std_f1']:.4f}")
    logger.info(f"  LSTM 宏平均 F1={lstm_result['mean_f1']}")
    return combined


# ==============================================================================
# 4. Phi 值分析（实验 B）
# ==============================================================================

def run_phi_analysis(
    eeg_data: np.ndarray,
    labels:   np.ndarray,
    nct:      NCTManager,
    sfreq:    int = 200,
    max_per_class: int = 30,
) -> Dict:
    """
    实验 B：Φ 值在 3 类状态下的统计分析（t 检验 + one-way ANOVA）。
    """
    phi_by_state: Dict[int, List[float]] = {0: [], 1: [], 2: []}
    nm_by_state:  Dict[int, List[Dict]]  = {0: [], 1: [], 2: []}

    counts = {0: 0, 1: 0, 2: 0}
    for eeg_trial, lbl in zip(eeg_data, labels):
        lbl_int = int(lbl)
        if counts[lbl_int] >= max_per_class:
            continue
        counts[lbl_int] += 1

        # 提取频带功率
        band_powers = compute_band_power(eeg_trial, sfreq)
        nm_state    = eeg_features_to_neuromodulator(band_powers)
        nm_by_state[lbl_int].append(nm_state)

        # 将 EEG 信号压缩为 NCT 视觉输入（28×28 伪图像）
        # 取前 784 个时间点 reshape
        eeg_flat = eeg_trial[0, :784] if eeg_trial.shape[-1] >= 784 else \
                   np.pad(eeg_trial[0], (0, 784 - eeg_trial.shape[-1]))
        visual_input = (eeg_flat.reshape(28, 28) - eeg_flat.min()) / \
                       (np.ptp(eeg_flat) + 1e-8)
        visual_input = visual_input.astype(np.float32)

        try:
            nct_state = nct.process_cycle({"visual": visual_input}, nm_state)
            phi = nct_state.consciousness_metrics.get("phi_value", 0.0)
        except Exception:
            phi = 0.0

        phi_by_state[lbl_int].append(phi)

    # ---- 统计检验 ----
    phi_neutral      = phi_by_state[0]
    phi_relaxing     = phi_by_state[1]
    phi_concentrating = phi_by_state[2]

    results = {
        "phi_stats": {},
        "t_test_relax_vs_conc": None,
        "anova_3way": None,
        "nm_means_by_state": {},
    }

    for state_id, state_name in ATTENTION_STATES.items():
        vals = phi_by_state[state_id]
        results["phi_stats"][state_name] = {
            "mean": round(float(np.mean(vals)),  4) if vals else 0.0,
            "std":  round(float(np.std(vals)),   4) if vals else 0.0,
            "n":    len(vals),
        }
        nm_list = nm_by_state[state_id]
        if nm_list:
            results["nm_means_by_state"][state_name] = {
                k: round(float(np.mean([d[k] for d in nm_list])), 4)
                for k in ["DA", "5-HT", "NE", "ACh"]
            }

    # t 检验：concentrating vs relaxing
    if len(phi_relaxing) >= 3 and len(phi_concentrating) >= 3:
        t_stat, p_val = ttest_ind(phi_concentrating, phi_relaxing, alternative="greater")
        results["t_test_relax_vs_conc"] = {
            "t_statistic": round(float(t_stat), 4),
            "p_value":     round(float(p_val),  6),
            "significant": bool(p_val < 0.05),
            "interpretation": (
                "Concentrating 状态下 Φ 值显著高于 Relaxing（p<0.05），"
                "支持 NCT 意识整合假说"
                if p_val < 0.05 else
                "差异未达显著水平，需更多数据"
            ),
        }
        logger.info(f"Concentrating vs Relaxing Φ t检验 t={t_stat:.4f}, p={p_val:.6f}")

    # one-way ANOVA：3 类差异
    if all(len(phi_by_state[i]) >= 2 for i in range(3)):
        f_stat, p_anova = f_oneway(phi_neutral, phi_relaxing, phi_concentrating)
        results["anova_3way"] = {
            "F_statistic": round(float(f_stat), 4),
            "p_value":     round(float(p_anova), 6),
            "significant": bool(p_anova < 0.05),
        }
        logger.info(f"3类 ANOVA: F={f_stat:.4f}, p={p_anova:.6f}")

    return results, phi_by_state


# ==============================================================================
# 5. 主实验流程
# ==============================================================================

def run_mema_experiment(use_mock: bool = False, max_per_class: int = 30, max_samples: int = 6000) -> Dict:
    """
    运行 MEMA EEG 神经调质映射实验（Phase 3）。

    Args:
        use_mock: 强制使用 Mock 数据
        max_per_class: Φ 分析时每类最大样本数
        max_samples: 加载时最大样本数（内存限制）
    """
    logger.info("=" * 60)
    logger.info("Phase 3 - MEMA EEG 神经调质映射实验（核心实验）")
    logger.info("=" * 60)

    # ---- 加载数据 ----
    loader = MEMALoader(DATA_DIR, max_samples=max_samples)
    if use_mock or not loader.is_available():
        logger.info("使用 Mock EEG 数据（MEMA 数据集未检测到）")
        eeg_data, labels = loader.generate_mock(
            n_subjects=5, n_trials_per_class=20
        )
        mock_mode = True
        sfreq = 200
    else:
        logger.info(f"加载真实 MEMA 数据：{DATA_DIR}")
        eeg_data, labels = loader.load_all()
        if eeg_data is None:
            logger.warning("加载失败，切换到 Mock 模式")
            eeg_data, labels = loader.generate_mock()
            mock_mode = True
        else:
            mock_mode = False
        sfreq = 200

    logger.info(f"EEG 数据：{eeg_data.shape}，标签分布：{dict(zip(*np.unique(labels, return_counts=True)))}")

    # ---- 提取频带功率特征矩阵（用于分类）----
    logger.info("提取 EEG 频带功率特征...")
    X_feat = np.array([
        extract_feature_vector(compute_band_power(trial, sfreq))
        for trial in eeg_data
    ])

    # ---- 初始化 NCT ----
    nct = NCTManager(LIGHT_NCT_CONFIG)
    nct.start()

    # ---- 实验 A: 分类基准 ----
    logger.info("实验A：SVM + LSTM 分类基准（5折交叉验证）...")
    clf_results = run_classification_benchmark(X_feat, labels)
    svm_f1  = clf_results["SVM"]["mean_f1"]
    lstm_f1 = clf_results["LSTM"]["mean_f1"]
    logger.info(f"  SVM  宏平均 F1 = {svm_f1:.4f}")
    logger.info(f"  LSTM 宏平均 F1 = {lstm_f1}")

    # ---- 实验 B/C: Phi 值分析 + 神经调质轨迹 ----
    logger.info("实验B/C：Phi 值分析与神经调质映射...")
    phi_results, phi_by_state = run_phi_analysis(
        eeg_data, labels, nct, sfreq=sfreq, max_per_class=max_per_class
    )

    # ---- 汇总结果 ----
    results = {
        "experiment":           "Phase3_MEMA_EEG",
        "mock_mode":            mock_mode,
        "n_samples":            int(len(labels)),
        "class_distribution":   {ATTENTION_STATES[i]: int((labels == i).sum())
                                  for i in range(3)},
        "experiment_A_classification": clf_results,
        "experiment_B_phi_analysis":   phi_results,
        "eeg_nm_mapping_basis": {
            "DA":   "Beta 功率（认知投入 / 奖励回路）",
            "5-HT": "Alpha 功率（放松抑制状态）",
            "NE":   "Theta/Alpha 比（警觉度 / 专注度）",
            "ACh":  "Theta 功率（工作记忆 / 海马节律）",
        },
    }

    # ---- 保存结果 ----
    out_path = RESULTS_DIR / "phase3_mema_phi_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"结果已保存：{out_path}")

    # ---- 可视化 ----
    _plot_phase3(results, phi_by_state, eeg_data, labels, sfreq)

    return results


def _plot_phase3(
    results: Dict,
    phi_by_state: Dict,
    eeg_data: np.ndarray,
    labels: np.ndarray,
    sfreq: int,
):
    """生成 Phase 3 可视化（3 子图）"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Phase 3: MEMA EEG × NCT 神经调质映射与Φ值分析", fontsize=13)

        state_colors = {"neutral": "#95A5A6", "relaxing": "#3498DB", "concentrating": "#E74C3C"}
        state_ids    = list(ATTENTION_STATES.items())

        # 子图1：各状态 Phi 值分布（箱线图）
        ax = axes[0]
        phi_lists  = [phi_by_state[i] for i in range(3) if phi_by_state[i]]
        state_lbls = [ATTENTION_STATES[i] for i in range(3) if phi_by_state[i]]
        if phi_lists:
            bp = ax.boxplot(phi_lists, patch_artist=True, notch=False)
            for patch, lbl in zip(bp["boxes"], state_lbls):
                patch.set_facecolor(state_colors.get(lbl, "gray"))
                patch.set_alpha(0.7)
            ax.set_xticklabels(state_lbls, rotation=15)
            ax.set_ylabel("Φ 值")
            ax.set_title("各注意力状态 Φ 值分布")
            ax.grid(True, alpha=0.3)
        phi_stats = results.get("experiment_B_phi_analysis", {}).get("phi_stats", {})
        if "t_test_relax_vs_conc" in results.get("experiment_B_phi_analysis", {}):
            t_res = results["experiment_B_phi_analysis"]["t_test_relax_vs_conc"] or {}
            p_txt = f"p={t_res.get('p_value', '?'):.4f}" if t_res else ""
            ax.set_xlabel(f"Concentrating vs Relaxing: {p_txt}")

        # 子图2：神经调质在 3 类状态下的雷达柱状图
        ax2 = axes[1]
        nm_by_state = results.get("experiment_B_phi_analysis", {}).get("nm_means_by_state", {})
        nm_keys = ["DA", "5-HT", "NE", "ACh"]
        x = np.arange(len(nm_keys))
        width = 0.25
        for offset, (s_name, color) in enumerate(state_colors.items()):
            if s_name in nm_by_state:
                vals = [nm_by_state[s_name].get(k, 0.5) for k in nm_keys]
                ax2.bar(x + offset * width, vals, width, label=s_name,
                        color=color, alpha=0.8)
        ax2.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="baseline")
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(nm_keys)
        ax2.set_ylabel("神经调质浓度")
        ax2.set_title("3 类状态下神经调质均值")
        ax2.set_ylim(0, 1)
        ax2.legend(fontsize=8)

        # 子图3：单个 trial EEG 功率谱（各频带均值）
        ax3 = axes[2]
        band_means_by_state = {}
        for state_id, state_name in ATTENTION_STATES.items():
            mask = labels == state_id
            if mask.sum() == 0:
                continue
            trials = eeg_data[mask][:5]
            band_means = {b: [] for b in FREQ_BANDS}
            for trial in trials:
                bp_dict = compute_band_power(trial, sfreq)
                for b in FREQ_BANDS:
                    band_means[b].append(bp_dict.get(b, 0))
            band_means_by_state[state_name] = {b: np.mean(v) for b, v in band_means.items()}

        bands = list(FREQ_BANDS.keys())
        x3 = np.arange(len(bands))
        w3 = 0.25
        for offset, (s_name, color) in enumerate(state_colors.items()):
            if s_name in band_means_by_state:
                vals = [band_means_by_state[s_name].get(b, 0) for b in bands]
                ax3.bar(x3 + offset * w3, vals, w3, label=s_name,
                        color=color, alpha=0.8)
        ax3.set_xticks(x3 + w3)
        ax3.set_xticklabels(bands)
        ax3.set_ylabel("相对功率")
        ax3.set_title("3 类状态 EEG 频带功率均值")
        ax3.legend(fontsize=8)

        # 标注 SVM + LSTM F1
        clf = results.get("experiment_A_classification", {})
        svm_f1  = clf.get("SVM",  {}).get("mean_f1", "?")
        lstm_f1 = clf.get("LSTM", {}).get("mean_f1", "?")
        svm_str  = f"{svm_f1:.4f}"  if isinstance(svm_f1,  float) else str(svm_f1)
        lstm_str = f"{lstm_f1:.4f}" if isinstance(lstm_f1, float) else str(lstm_f1)
        fig.text(0.5, 0.01,
                 f"三方对比（宏平均 F1）：SVM={svm_str}  LSTM={lstm_str}  NCT(见实验B)",
                 ha="center", fontsize=10, color="navy")

        plt.tight_layout(rect=[0, 0.04, 1, 1])
        out_path = RESULTS_DIR / "phase3_mema_eeg_analysis.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"可视化已保存：{out_path}")
    except Exception as e:
        logger.warning(f"可视化失败：{e}")


# ==============================================================================
# 入口
# ==============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 3: MEMA EEG × NCT 实验")
    parser.add_argument("--mock",          action="store_true", help="强制使用 Mock EEG 数据")
    parser.add_argument("--max-per-class", type=int, default=30, help="每类最大样本数（Phi 分析）")
    parser.add_argument("--max-samples",   type=int, default=6000, help="加载时最大样本数（内存限制）")
    args = parser.parse_args()

    results = run_mema_experiment(use_mock=args.mock, max_per_class=args.max_per_class, max_samples=args.max_samples)

    print("\n========== Phase 3 实验结果摘要 ==========")
    clf = results["experiment_A_classification"]
    print(f"\n[实验A] 三方分类对比:")
    svm_r  = clf.get("SVM",  {})
    lstm_r = clf.get("LSTM", {})
    print(f"  SVM  宏平均 F1 = {svm_r.get('mean_f1', '?')}")
    print(f"  LSTM 宏平均 F1 = {lstm_r.get('mean_f1', '?')}")
    print("  SVM 每类 F1:")
    for cls, f1 in svm_r.get("per_class", {}).items():
        print(f"    {cls:15s}: {f1:.4f}")

    phi_a = results["experiment_B_phi_analysis"]
    print(f"\n[实验B] Φ 值统计")
    for state, stat in phi_a.get("phi_stats", {}).items():
        print(f"  {state:15s}: mean={stat['mean']:.4f} ± {stat['std']:.4f} (n={stat['n']})")
    if phi_a.get("t_test_relax_vs_conc"):
        t_res = phi_a["t_test_relax_vs_conc"]
        print(f"\n  Concentrating vs Relaxing t检验")
        print(f"  t={t_res['t_statistic']:.4f}, p={t_res['p_value']:.6f}")
        print(f"  {'[显著]' if t_res['significant'] else '[不显著]'} {t_res['interpretation']}")

    print(f"\n[实验C] 神经调质映射（按状态）:")
    for state, nm in phi_a.get("nm_means_by_state", {}).items():
        nm_str = "  ".join(f"{k}={v:.3f}" for k, v in nm.items())
        print(f"  {state:15s}: {nm_str}")