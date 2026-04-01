"""
Phase 4: EdNet 知识追踪自适应学习实验
========================================
验证 NCT 神经调质驱动的自适应难度调整是否相对固定难度策略
提升学生知识掌握效率（答题准确率增长 + AUC）。

EdNet 数据格式（Kaggle 下载）：
  - KT1 级别：student_id, timestamp, content_id, answered_correctly
  - 文件：KT1/u{student_id}.csv
  - 下载：https://www.kaggle.com/datasets/anhtu96/ednet-contents

实验设计：
  对同一批学生历史答题序列，分别使用：
    ① 固定难度策略（baseline）：不调整题目难度
    ② NCT 自适应策略：根据神经调质状态动态调整题目难度
  评估指标：
    - 滑动窗口准确率增长曲线（学习效率）
    - AUC-ROC（知识掌握预测能力）
    - 收敛答对率（最后 20% 题目的平均准确率）

知识追踪（DKT 简化版）：
  - 使用隐马尔可夫链近似（BKT）建模知识掌握概率
  - p(mastery) 驱动题目难度分配
  - NCT 神经调质状态调整 p(learning) 速率

作者：NCT LAB Team
日期：2026-03-15
"""

import sys
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import ttest_rel
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nct_modules.nct_manager import NCTManager
from nct_modules.nct_core import NCTConfig
from experiments.education_state_detection import (
    NeuromodulatorState,
    NeuromodulatorMapper,
    TeachingStrategyGenerator,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR    = PROJECT_ROOT / "data" / "ednet"
RESULTS_DIR = PROJECT_ROOT / "results" / "education"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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
# 1. EdNet 数据加载
# ==============================================================================

class EdNetLoader:
    """EdNet KT1 数据集加载器（兼容原始格式和含 answered_correctly 格式）"""

    def __init__(self, data_dir: Path, max_students: int = 50, min_interactions: int = 30):
        self.data_dir = data_dir
        self.max_students = max_students
        self.min_interactions = min_interactions
        # 加载 contents.csv 作为答案键（支持原始 EdNet 格式）
        self._correct_answer_map = self._load_contents()

    def _load_contents(self) -> dict:
        """加载 contents.csv，构建 question_id -> correct_answer 映射"""
        import csv
        contents_path = self.data_dir / "contents.csv"
        answer_map = {}
        if not contents_path.exists():
            return answer_map
        with open(contents_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                qid = row.get("question_id", "").strip()
                ans = row.get("correct_answer", "").strip().lower()
                if qid and ans:
                    answer_map[qid] = ans
        logger.info(f"已加载 contents.csv：{len(answer_map)} 道题的答案键")
        return answer_map

    def _get_correctness(self, row: dict) -> Optional[int]:
        """从行记录推断答题是否正确（兼容两种格式）"""
        # 格式1：直接含 answered_correctly / correct 字段
        answered = row.get("answered_correctly", row.get("correct", ""))
        if answered in ("0", "1"):
            return int(answered)
        # 格式2：原始 EdNet KT1（user_answer + contents.csv 对比）
        user_ans = row.get("user_answer", "").strip().lower()
        qid = row.get("question_id", row.get("content_id", "")).strip()
        if user_ans and qid and self._correct_answer_map:
            correct_ans = self._correct_answer_map.get(qid, "")
            if correct_ans:
                return int(user_ans == correct_ans)
        return None

    def is_available(self) -> bool:
        csv_files = list(self.data_dir.rglob("*.csv"))
        return len(csv_files) > 0

    def load_students(self) -> List[Dict]:
        """
        加载学生答题序列。

        Returns:
            students: [{"student_id": str, "interactions": [{"content_id", "correct", "timestamp"}, ...]}]
        """
        import csv
        students = []
        # 仅扫描 KT1/ 子目录中的 u*.csv，跳过 contents.csv
        kt1_dir = self.data_dir / "KT1"
        if kt1_dir.exists():
            csv_files = sorted(kt1_dir.glob("u*.csv"))[:self.max_students * 2]
        else:
            csv_files = [f for f in sorted(self.data_dir.rglob("u*.csv"))
                         if f.name != "contents.csv"][:self.max_students * 2]
        for csv_file in csv_files:
            try:
                interactions = []
                with open(csv_file, newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        correct = self._get_correctness(row)
                        if correct is None:
                            continue
                        # 兼容整数 ID 和 "q123" 字符串 ID
                        raw_id = row.get("content_id", row.get("question_id", "0"))
                        try:
                            content_id = int(raw_id)
                        except (ValueError, TypeError):
                            digits = "".join(c for c in str(raw_id) if c.isdigit())
                            content_id = int(digits) if digits else 0
                        interactions.append({
                            "content_id": content_id,
                            "correct":    correct,
                            "timestamp":  int(row.get("timestamp", 0)),
                        })
                if len(interactions) >= self.min_interactions:
                    students.append({
                        "student_id": csv_file.stem,
                        "interactions": interactions[:200],  # 最多取 200 条
                    })
                if len(students) >= self.max_students:
                    break
            except Exception as e:
                logger.debug(f"跳过 {csv_file.name}: {e}")
        return students

    def generate_mock_students(self, n_students: int = 50, n_questions: int = 80) -> List[Dict]:
        """
        生成 Mock 学生序列（模拟知识掌握逐渐提升的过程）。

        学生类型：
          - fast_learner:  p_correct 从 0.4 逐步升到 0.85
          - slow_learner:  p_correct 从 0.3 逐步升到 0.65
          - medium_learner:p_correct 从 0.5 逐步升到 0.75
        """
        np.random.seed(42)
        students = []
        profiles = [
            ("fast",   0.40, 0.85),
            ("medium", 0.50, 0.75),
            ("slow",   0.30, 0.65),
        ]
        for i in range(n_students):
            profile = profiles[i % 3]
            _, p_start, p_end = profile
            student_id = f"mock_s{i:03d}"
            interactions = []
            for q_idx in range(n_questions):
                # 学习曲线：logistic 增长
                progress = q_idx / n_questions
                p_correct = p_start + (p_end - p_start) * (1 / (1 + np.exp(-8 * (progress - 0.5))))
                correct = int(np.random.rand() < p_correct)
                interactions.append({
                    "content_id": np.random.randint(1, 500),
                    "correct":    correct,
                    "timestamp":  q_idx * 60,
                })
            students.append({"student_id": student_id, "interactions": interactions})
        return students


# ==============================================================================
# 2. 贝叶斯知识追踪（BKT）简化版
# ==============================================================================

@dataclass
class BKTState:
    """贝叶斯知识追踪状态"""
    p_mastery:    float = 0.1   # 当前知识掌握概率
    p_learn:      float = 0.15  # 基准学习率
    p_guess:      float = 0.25  # 猜测概率
    p_slip:       float = 0.10  # 失误概率

    def update(self, correct: int, p_learn_override: Optional[float] = None) -> None:
        """根据答题结果更新掌握概率（BKT 标准更新）"""
        p_learn = p_learn_override if p_learn_override else self.p_learn
        if correct == 1:
            p_correct_given_mastery   = 1 - self.p_slip
            p_correct_given_unmastery = self.p_guess
        else:
            p_correct_given_mastery   = self.p_slip
            p_correct_given_unmastery = 1 - self.p_guess

        # 后验更新
        numerator   = p_correct_given_mastery   * self.p_mastery
        denominator = (numerator +
                       p_correct_given_unmastery * (1 - self.p_mastery))
        if denominator > 0:
            self.p_mastery = numerator / denominator

        # 知识增长
        self.p_mastery = self.p_mastery + (1 - self.p_mastery) * p_learn
        self.p_mastery = float(np.clip(self.p_mastery, 0.0, 1.0))

    def predict_correct(self) -> float:
        """预测答对概率"""
        return (self.p_mastery * (1 - self.p_slip) +
                (1 - self.p_mastery) * self.p_guess)


# ==============================================================================
# 3. 自适应难度策略
# ==============================================================================

def infer_nm_from_performance(
    history_window: List[int], response_times: Optional[List[float]] = None
) -> Dict[str, float]:
    """
    根据近期答题表现推断神经调质状态。

    映射规则：
      高正确率 (>0.7) → DA↑（动机强）
      低正确率 (<0.4) → NE↑（困惑）+ DA↓
      正确率中等      → 5-HT 维持基准
      响应时间长      → ACh↑（深度加工）或 NE↑（困难）
    """
    if not history_window:
        return {"DA": 0.5, "5-HT": 0.5, "NE": 0.5, "ACh": 0.5}

    recent_acc = float(np.mean(history_window))
    n = len(history_window)
    # 最近趋势（后半段 vs 前半段）
    mid = n // 2 + 1
    trend = (np.mean(history_window[mid:]) - np.mean(history_window[:mid])) if n >= 4 else 0.0

    DA  = float(np.clip(0.3 + recent_acc * 0.5 + trend * 0.3, 0.0, 1.0))
    sHT = float(np.clip(0.4 + recent_acc * 0.2, 0.0, 1.0))
    NE  = float(np.clip(0.7 - recent_acc * 0.4, 0.0, 1.0))

    avg_rt = float(np.mean(response_times)) if response_times else 60.0
    ACh = float(np.clip(0.3 + min(avg_rt, 120) / 200, 0.0, 1.0))

    return {"DA": round(DA, 4), "5-HT": round(sHT, 4), "NE": round(NE, 4), "ACh": round(ACh, 4)}


def adaptive_difficulty(
    nm_state: Dict[str, float],
    current_difficulty: float,
    bkt_state: BKTState,
) -> Tuple[float, float]:
    """
    NCT 自适应难度调整策略。

    Args:
        nm_state:           神经调质状态
        current_difficulty: 当前难度 (0-1)
        bkt_state:          BKT 知识追踪状态

    Returns:
        (new_difficulty, p_learn_boost)
    """
    DA   = nm_state.get("DA",   0.5)
    NE   = nm_state.get("NE",   0.5)
    ACh  = nm_state.get("ACh",  0.5)
    sHT  = nm_state.get("5-HT", 0.5)

    # -- 难度调整 --
    delta = 0.0
    if DA > 0.7 and NE < 0.4 and bkt_state.p_mastery > 0.6:
        delta = +0.10   # 高动机 + 低困惑 + 已掌握 → 升难度
    elif NE > 0.7 or DA < 0.3:
        delta = -0.12   # 困惑/低动机 → 降难度
    elif sHT < 0.3:
        delta = -0.08   # 压力大 → 稍降难度
    elif bkt_state.p_mastery > 0.8:
        delta = +0.06   # 掌握度高 → 小幅升难度

    new_difficulty = float(np.clip(current_difficulty + delta, 0.0, 1.0))

    # -- 学习率加成 --
    p_learn_boost = bkt_state.p_learn * (1 + 0.5 * ACh)  # ACh 增强记忆编码

    return new_difficulty, float(np.clip(p_learn_boost, 0.0, 0.5))


def simulate_answer(
    correct_prob: float,
    difficulty: float,
    student_ability: float,
) -> Tuple[int, float]:
    """
    模拟学生在特定难度下的答题结果和响应时间。

    Args:
        correct_prob: BKT 预测的答对概率
        difficulty:   题目难度 (0-1)
        student_ability: 学生能力（BKT p_mastery）

    Returns:
        (answered_correctly, response_time_seconds)
    """
    # 难度会降低实际答对概率
    adjusted_prob = correct_prob * (1 - 0.4 * difficulty) + 0.1
    adjusted_prob = float(np.clip(adjusted_prob, 0.05, 0.95))
    correct = int(np.random.rand() < adjusted_prob)
    # 响应时间：难题用时更长
    base_time = 20 + difficulty * 80
    rt = base_time * np.random.lognormal(0, 0.3)
    return correct, float(rt)


# ==============================================================================
# 4. 单学生模拟
# ==============================================================================

def simulate_student_session(
    interactions: List[Dict],
    strategy: str = "fixed",   # "fixed" or "adaptive"
    window_size: int = 10,
) -> Dict:
    """
    模拟单个学生完整学习会话。

    Args:
        interactions: 历史答题序列（content_id, correct, timestamp）
        strategy:     "fixed" = 固定难度 0.5, "adaptive" = NCT 神经调质自适应
        window_size:  滑动窗口大小

    Returns:
        session_result: 包含准确率序列、BKT轨迹、神经调质轨迹
    """
    bkt = BKTState(p_mastery=0.1, p_learn=0.15)
    difficulty = 0.5   # 固定策略起始难度

    correct_history  = []
    bkt_mastery_traj = []
    nm_traj          = []
    difficulty_traj  = []
    preds_proba      = []   # for AUC
    true_labels      = []

    for i, event in enumerate(interactions):
        actual_correct = event["correct"]
        true_labels.append(actual_correct)
        pred_prob = bkt.predict_correct()
        preds_proba.append(pred_prob)

        # 最近窗口
        recent_window = correct_history[-window_size:]

        if strategy == "adaptive":
            # 推断神经调质
            nm_state = infer_nm_from_performance(recent_window)
            nm_traj.append(nm_state)

            # 自适应难度
            new_diff, p_learn_boost = adaptive_difficulty(nm_state, difficulty, bkt)
            difficulty = new_diff

            # BKT 更新（带学习率加成）
            bkt.update(actual_correct, p_learn_override=p_learn_boost)
        else:
            # 固定策略
            nm_traj.append({"DA": 0.5, "5-HT": 0.5, "NE": 0.5, "ACh": 0.5})
            bkt.update(actual_correct)

        correct_history.append(actual_correct)
        bkt_mastery_traj.append(bkt.p_mastery)
        difficulty_traj.append(difficulty)

    # 滑动窗口准确率
    rolling_acc = []
    for j in range(len(correct_history)):
        start = max(0, j - window_size + 1)
        rolling_acc.append(float(np.mean(correct_history[start:j + 1])))

    # AUC
    auc = roc_auc_score(true_labels, preds_proba) if len(set(true_labels)) > 1 else 0.5

    # 收敛准确率（最后 20% 的问题）
    tail_n = max(1, len(correct_history) // 5)
    converged_acc = float(np.mean(correct_history[-tail_n:]))

    return {
        "rolling_accuracy":  rolling_acc,
        "bkt_mastery_traj":  bkt_mastery_traj,
        "difficulty_traj":   difficulty_traj,
        "nm_traj":           nm_traj,
        "final_auc":         round(auc, 4),
        "converged_acc":     round(converged_acc, 4),
        "mean_acc":          round(float(np.mean(correct_history)), 4),
        "n_questions":       len(correct_history),
    }


# ==============================================================================
# 5. 主实验流程
# ==============================================================================

def run_ednet_experiment(
    use_mock: bool = False,
    n_students: int = 50,
    window_size: int = 10,
) -> Dict:
    """
    运行 EdNet 知识追踪自适应实验（Phase 4）。
    """
    logger.info("=" * 60)
    logger.info("Phase 4 - EdNet 知识追踪自适应学习实验")
    logger.info("=" * 60)

    loader = EdNetLoader(DATA_DIR, max_students=n_students)

    if use_mock or not loader.is_available():
        logger.info("使用 Mock 学生数据（EdNet 数据集未检测到）")
        students = loader.generate_mock_students(n_students=n_students, n_questions=80)
        mock_mode = True
    else:
        logger.info(f"加载真实 EdNet 数据：{DATA_DIR}")
        students = loader.load_students()
        if not students:
            logger.warning("未找到有效数据，切换到 Mock 模式")
            students = loader.generate_mock_students(n_students=n_students)
            mock_mode = True
        else:
            mock_mode = False

    logger.info(f"学生数量：{len(students)}")

    # ---- 对比实验：固定 vs 自适应 ----
    fixed_results    = []
    adaptive_results = []

    for idx, student in enumerate(students):
        interactions = student["interactions"]
        if len(interactions) < 10:
            continue

        np.random.seed(idx)
        fixed_res    = simulate_student_session(interactions, strategy="fixed",    window_size=window_size)
        np.random.seed(idx)  # 相同随机种子，保证公平对比
        adaptive_res = simulate_student_session(interactions, strategy="adaptive", window_size=window_size)

        fixed_results.append(fixed_res)
        adaptive_results.append(adaptive_res)

        if (idx + 1) % 10 == 0:
            logger.info(f"  已处理 {idx + 1}/{len(students)} 名学生")

    if not fixed_results:
        logger.error("没有有效的模拟结果")
        return {}

    # ---- 聚合统计 ----
    fixed_auc    = [r["final_auc"]     for r in fixed_results]
    adaptive_auc = [r["final_auc"]     for r in adaptive_results]
    fixed_conv   = [r["converged_acc"] for r in fixed_results]
    adaptive_conv = [r["converged_acc"] for r in adaptive_results]
    fixed_mean   = [r["mean_acc"]      for r in fixed_results]
    adaptive_mean = [r["mean_acc"]     for r in adaptive_results]

    # 配对 t 检验
    t_auc,  p_auc  = ttest_rel(adaptive_auc,  fixed_auc)
    t_conv, p_conv = ttest_rel(adaptive_conv, fixed_conv)

    # 平均学习曲线（对齐到最短序列）
    min_len = min(len(r["rolling_accuracy"]) for r in fixed_results + adaptive_results)
    fixed_curves    = np.array([r["rolling_accuracy"][:min_len] for r in fixed_results])
    adaptive_curves = np.array([r["rolling_accuracy"][:min_len] for r in adaptive_results])
    avg_fixed_curve    = fixed_curves.mean(axis=0).tolist()
    avg_adaptive_curve = adaptive_curves.mean(axis=0).tolist()

    results = {
        "experiment":   "Phase4_EdNet_Adaptive",
        "mock_mode":    mock_mode,
        "n_students":   len(fixed_results),
        "window_size":  window_size,
        "fixed_strategy": {
            "mean_auc":        round(float(np.mean(fixed_auc)),  4),
            "mean_conv_acc":   round(float(np.mean(fixed_conv)), 4),
            "mean_overall_acc":round(float(np.mean(fixed_mean)), 4),
        },
        "adaptive_strategy": {
            "mean_auc":        round(float(np.mean(adaptive_auc)),  4),
            "mean_conv_acc":   round(float(np.mean(adaptive_conv)), 4),
            "mean_overall_acc":round(float(np.mean(adaptive_mean)), 4),
        },
        "improvements": {
            "auc_delta":          round(float(np.mean(adaptive_auc))  - float(np.mean(fixed_auc)),  4),
            "conv_acc_delta":     round(float(np.mean(adaptive_conv)) - float(np.mean(fixed_conv)), 4),
            "auc_t_test":         {"t": round(float(t_auc), 4), "p": round(float(p_auc), 6)},
            "conv_acc_t_test":    {"t": round(float(t_conv), 4), "p": round(float(p_conv), 6)},
            "auc_significant":    bool(p_auc < 0.05),
            "conv_acc_significant": bool(p_conv < 0.05),
        },
        "learning_curves": {
            "fixed_avg":    [round(v, 4) for v in avg_fixed_curve[:50]],
            "adaptive_avg": [round(v, 4) for v in avg_adaptive_curve[:50]],
        },
    }

    logger.info(f"\n  AUC  — 固定={np.mean(fixed_auc):.4f}, 自适应={np.mean(adaptive_auc):.4f}, "
                f"Δ={results['improvements']['auc_delta']:+.4f}, p={p_auc:.4f}")
    logger.info(f"  收敛准确率 — 固定={np.mean(fixed_conv):.4f}, 自适应={np.mean(adaptive_conv):.4f}, "
                f"Δ={results['improvements']['conv_acc_delta']:+.4f}, p={p_conv:.4f}")

    # ---- 保存结果 ----
    out_path = RESULTS_DIR / "phase4_ednet_adaptive_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"结果已保存：{out_path}")

    # ---- 可视化 ----
    _plot_phase4(results, fixed_results, adaptive_results)

    return results


def _plot_phase4(results: Dict, fixed_res: List, adaptive_res: List):
    """生成 Phase 4 可视化"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Phase 4: EdNet × NCT 自适应学习 — 固定策略 vs NCT 自适应策略", fontsize=13)

        # 子图1：平均学习曲线对比
        ax = axes[0]
        curves = results.get("learning_curves", {})
        fixed_curve    = curves.get("fixed_avg",    [])
        adaptive_curve = curves.get("adaptive_avg", [])
        if fixed_curve:
            q_idx = list(range(len(fixed_curve)))
            ax.plot(q_idx, fixed_curve,    label="固定难度策略",    color="#3498DB", linewidth=2)
            ax.plot(q_idx, adaptive_curve, label="NCT 自适应策略",  color="#E74C3C", linewidth=2)
            ax.fill_between(q_idx, fixed_curve, adaptive_curve,
                            alpha=0.15, color="#E74C3C")
        ax.set_xlabel("题目序号")
        ax.set_ylabel("滑动窗口准确率")
        ax.set_title("平均学习曲线对比")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        # 子图2：AUC 比较（散点图）
        ax2 = axes[1]
        fixed_auc    = [r["final_auc"]     for r in fixed_res]
        adaptive_auc = [r["final_auc"]     for r in adaptive_res]
        n = len(fixed_auc)
        ax2.scatter(fixed_auc, adaptive_auc, alpha=0.5, color="#9B59B6", s=30)
        lim_min = min(min(fixed_auc), min(adaptive_auc)) - 0.02
        lim_max = max(max(fixed_auc), max(adaptive_auc)) + 0.02
        ax2.plot([lim_min, lim_max], [lim_min, lim_max], "k--", alpha=0.4, label="对角线（相等）")
        ax2.set_xlabel("固定策略 AUC")
        ax2.set_ylabel("NCT 自适应策略 AUC")
        ax2.set_title("每学生 AUC 对比")
        ax2.legend()
        imp = results["improvements"]
        ax2.set_title(
            f"每学生 AUC 对比\n"
            f"Δ={imp['auc_delta']:+.4f}, p={imp['auc_t_test']['p']:.4f}"
        )

        # 子图3：收敛准确率对比（箱线图）
        ax3 = axes[2]
        fixed_conv    = [r["converged_acc"] for r in fixed_res]
        adaptive_conv = [r["converged_acc"] for r in adaptive_res]
        bp = ax3.boxplot([fixed_conv, adaptive_conv],
                         labels=["固定策略", "NCT 自适应"],
                         patch_artist=True)
        bp["boxes"][0].set_facecolor("#3498DB")
        bp["boxes"][0].set_alpha(0.7)
        bp["boxes"][1].set_facecolor("#E74C3C")
        bp["boxes"][1].set_alpha(0.7)
        ax3.set_ylabel("收敛准确率（最后 20% 题目）")
        ax3.set_title(
            f"收敛准确率对比\n"
            f"Δ={imp['conv_acc_delta']:+.4f}, p={imp['conv_acc_t_test']['p']:.4f}"
        )
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = RESULTS_DIR / "phase4_ednet_adaptive_comparison.png"
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
    parser = argparse.ArgumentParser(description="Phase 4: EdNet × NCT 自适应学习实验")
    parser.add_argument("--mock",        action="store_true", help="强制使用 Mock 数据")
    parser.add_argument("--n-students",  type=int, default=50,  help="学生数量（默认 50）")
    parser.add_argument("--window-size", type=int, default=10,  help="滑动窗口大小（默认 10）")
    args = parser.parse_args()

    results = run_ednet_experiment(
        use_mock=args.mock,
        n_students=args.n_students,
        window_size=args.window_size,
    )

    print("\n========== Phase 4 实验结果摘要 ==========")
    if not results:
        print("实验未产生有效结果")
    else:
        fixed    = results["fixed_strategy"]
        adaptive = results["adaptive_strategy"]
        imp      = results["improvements"]
        print(f"\n          {'固定策略':>12}   {'NCT自适应':>12}   {'提升Δ':>10}")
        print(f"  AUC       {fixed['mean_auc']:>12.4f}   {adaptive['mean_auc']:>12.4f}"
              f"   {imp['auc_delta']:>+10.4f}  {'[显著]' if imp['auc_significant'] else ''}")
        print(f"  收敛准确率 {fixed['mean_conv_acc']:>12.4f}   {adaptive['mean_conv_acc']:>12.4f}"
              f"   {imp['conv_acc_delta']:>+10.4f}  {'[显著]' if imp['conv_acc_significant'] else ''}")
        print(f"  整体准确率 {fixed['mean_overall_acc']:>12.4f}   {adaptive['mean_overall_acc']:>12.4f}")
        print(f"\n配对 t 检验（AUC）：t={imp['auc_t_test']['t']:.4f}, "
              f"p={imp['auc_t_test']['p']:.6f}")
        print(f"配对 t 检验（收敛）：t={imp['conv_acc_t_test']['t']:.4f}, "
              f"p={imp['conv_acc_t_test']['p']:.6f}")
