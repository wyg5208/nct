"""
教育领域 NCT 全量实验总控运行器
=================================
按顺序执行全部四个 Phase 的实验，收集跨 Phase 结果，
生成综合对比可视化和最终 JSON 报告。

执行顺序：
  Phase 2  → FER 面部表情预训练      (训练 CheckPoint 供 Phase 1 使用)
  Phase 1  → DAiSEE 视觉验证          (使用 FER 模型增强)
  Phase 3  → MEMA EEG 神经调质映射   (核心实验)
  Phase 4  → EdNet 自适应学习         (知识追踪)

用法：
  python run_all_education_experiments.py --mock
  python run_all_education_experiments.py --mock --epochs 5 --clips 50

参数：
  --mock      所有 Phase 均使用 Mock 数据（无真实数据集时）
  --epochs N  Phase 2 FER 预训练轮数（默认 5，Mock 下推荐 3）
  --clips N   Phase 1 DAiSEE 处理片段数（默认 100）
  --students N Phase 4 EdNet 学生数（默认 30）
  --skip-2    跳过 Phase 2（使用已有 checkpoint）

作者：NCT LAB Team
日期：2026-03-15
"""

import sys
import json
import logging
import time
import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("run_all")

RESULTS_DIR = PROJECT_ROOT / "results" / "education"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 1. Phase 执行器
# ==============================================================================

def run_phase2_fer(use_mock: bool, n_epochs: int = 5) -> Optional[Dict]:
    """Phase 2: FER 面部表情预训练"""
    logger.info("\n" + "=" * 70)
    logger.info("Phase 2 ▶  FER 面部表情预训练（生成 NCT 神经调质映射模型）")
    logger.info("=" * 70)
    try:
        from experiments.fer_pretrain import run_fer_pretrain_experiment
        results = run_fer_pretrain_experiment(use_mock=use_mock, n_epochs=n_epochs)
        logger.info(f"  Phase 2 完成 ✓  最优验证准确率 = {results.get('best_val_acc', '?')}")
        return results
    except Exception as e:
        logger.error(f"Phase 2 失败：{e}")
        import traceback
        traceback.print_exc()
        return None


def run_phase1_daisee(use_mock: bool, max_clips: int = 100,
                      frames: int = 3, use_fer: bool = True) -> Optional[Dict]:
    """Phase 1: DAiSEE 视觉验证（FER 增强）"""
    logger.info("\n" + "=" * 70)
    logger.info("Phase 1 ▶  DAiSEE × NCT 视觉验证实验（FER 增强）")
    logger.info("=" * 70)
    try:
        from experiments.daisee_nct_experiment import run_daisee_experiment
        results = run_daisee_experiment(
            max_clips=max_clips,
            frames_per_clip=frames,
            use_mock=use_mock,
            use_fer=use_fer,
        )
        fer_flag = results.get("fer_enhanced", False)
        r_corr   = results.get("correlations", {}).get("engagement_level_vs_phi", {})
        logger.info(f"  Phase 1 完成 ✓  FER增强={fer_flag} | "
                    f"Engagement×Φ r={r_corr.get('r', '?')}")
        return results
    except Exception as e:
        logger.error(f"Phase 1 失败：{e}")
        import traceback
        traceback.print_exc()
        return None


def run_phase3_mema(use_mock: bool, max_per_class: int = 30) -> Optional[Dict]:
    """Phase 3: MEMA EEG 神经调质映射实验"""
    logger.info("\n" + "=" * 70)
    logger.info("Phase 3 ▶  MEMA EEG 神经调质映射实验（核心实验）")
    logger.info("=" * 70)
    try:
        from experiments.mema_nct_experiment import run_mema_experiment
        results = run_mema_experiment(use_mock=use_mock, max_per_class=max_per_class)
        clf   = results.get("experiment_A_classification", {})
        svm_f1  = clf.get("SVM",  {}).get("mean_f1", "?")
        lstm_f1 = clf.get("LSTM", {}).get("mean_f1", "?")
        phi_b = results.get("experiment_B_phi_analysis", {})
        sig   = phi_b.get("t_test_relax_vs_conc", {})
        is_sig = sig.get("significant", False) if sig else False
        logger.info(f"  Phase 3 完成 ✓  SVM F1={svm_f1}  LSTM F1={lstm_f1} | "
                    f"Φ t检验显著={is_sig}")
        return results
    except Exception as e:
        logger.error(f"Phase 3 失败：{e}")
        import traceback
        traceback.print_exc()
        return None


def run_phase4_ednet(use_mock: bool, n_students: int = 30) -> Optional[Dict]:
    """Phase 4: EdNet 自适应学习实验"""
    logger.info("\n" + "=" * 70)
    logger.info("Phase 4 ▶  EdNet × NCT 自适应学习实验")
    logger.info("=" * 70)
    try:
        from experiments.ednet_adaptive_experiment import run_ednet_experiment
        results = run_ednet_experiment(use_mock=use_mock, n_students=n_students)
        delta_auc = results.get("improvements", {}).get("auc_delta", "?")
        p_val     = results.get("improvements", {}).get("auc_t_test", {}).get("p", "?")
        logger.info(f"  Phase 4 完成 ✓  ΔAUC={delta_auc}  p={p_val}")
        return results
    except Exception as e:
        logger.error(f"Phase 4 失败：{e}")
        import traceback
        traceback.print_exc()
        return None


# ==============================================================================
# 2. 综合可视化
# ==============================================================================

def _plot_combined_summary(all_results: Dict):
    """生成跨 Phase 4 宫格综合可视化"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            "NCT 教育领域实验综合报告\n"
            "NeuroConscious Transformer × Education AI",
            fontsize=15, fontweight="bold"
        )

        # ── 子图 1：Phase 1 Engagement 强度 vs Φ 均值 ──
        ax1 = axes[0][0]
        p1  = all_results.get("phase1")
        if p1:
            phi_data = p1.get("phi_by_label", {}).get("Engagement", {})
            levels = [0, 1, 2, 3]
            means  = [phi_data.get(str(lv), {}).get("mean", 0) for lv in levels]
            ax1.bar(levels, means, color=["#AED6F1", "#5DADE2", "#2E86C1", "#1A5276"],
                    width=0.6)
            ax1.set_xlabel("Engagement 强度 (0-3)")
            ax1.set_ylabel("平均 Φ 值")
            ax1.set_title("Phase 1 – DAiSEE\nEngagement 强度 × NCT Φ 值")
            ax1.set_xticks(levels)
            ax1.set_xticklabels(["Very Low", "Low", "Medium", "High"], fontsize=8)
            r  = p1.get("correlations", {}).get("engagement_level_vs_phi", {}).get("r", "?")
            p_ = p1.get("correlations", {}).get("engagement_level_vs_phi", {}).get("p", "?")
            ax1.text(0.05, 0.93, f"r={r}, p={p_}", transform=ax1.transAxes,
                     fontsize=9, color="darkblue", va="top")
            fer_flag = "FER增强" if p1.get("fer_enhanced") else "规则模式"
            ax1.text(0.05, 0.84, fer_flag, transform=ax1.transAxes,
                     fontsize=8, color="green" if p1.get("fer_enhanced") else "gray")
        else:
            ax1.text(0.5, 0.5, "Phase 1 未运行", ha="center", transform=ax1.transAxes)
        ax1.grid(True, alpha=0.3)

        # ── 子图 2：Phase 2 FER 训练曲线 ──
        ax2 = axes[0][1]
        p2 = all_results.get("phase2")
        if p2 and "history" in p2:
            hist = p2["history"]
            epochs = range(1, len(hist.get("val_acc", [])) + 1)
            if epochs:
                ax2.plot(epochs, hist.get("train_acc", []), label="训练准确率",
                         color="#E74C3C", linewidth=2)
                ax2.plot(epochs, hist.get("val_acc", []),   label="验证准确率",
                         color="#27AE60", linewidth=2, linestyle="--")
                ax2.set_xlabel("训练轮次")
                ax2.set_ylabel("准确率")
                ax2.set_title("Phase 2 – FER2013\n面部表情分类训练曲线")
                ax2.legend(fontsize=9)
                ax2.grid(True, alpha=0.3)
                best = p2.get("best_val_acc", "?")
                ax2.axhline(y=float(best) if isinstance(best, (int, float)) else 0,
                            color="orange", linestyle=":", label=f"最优={best}")
        else:
            ax2.text(0.5, 0.5, "Phase 2 未运行 / 无历史数据",
                     ha="center", transform=ax2.transAxes)
        ax2.set_title("Phase 2 – FER2013\n面部表情分类训练曲线")

        # ── 子图 3：Phase 3 三方对比柱状图 ──
        ax3 = axes[1][0]
        p3  = all_results.get("phase3")
        if p3:
            clf   = p3.get("experiment_A_classification", {})
            svm_f1  = clf.get("SVM",  {}).get("mean_f1")
            lstm_f1 = clf.get("LSTM", {}).get("mean_f1")
            phi_b   = p3.get("experiment_B_phi_analysis", {})
            phi_c   = phi_b.get("phi_stats", {}).get("concentrating", {}).get("mean")

            methods = ["SVM", "LSTM", "NCT (Φ-max)"]
            scores  = []
            for v in [svm_f1, lstm_f1, phi_c]:
                scores.append(float(v) if v is not None else 0.0)

            # 归一化 Phi 到 [0,1] 以便可视化（Phi 值已归一化为分类置信度代理）
            if phi_c is not None:
                conc_phi   = phi_b.get("phi_stats", {}).get("concentrating", {}).get("mean", 0)
                relax_phi  = phi_b.get("phi_stats", {}).get("relaxing",      {}).get("mean", 0)
                neutral_phi = phi_b.get("phi_stats", {}).get("neutral",      {}).get("mean", 0)
                phi_range   = max(conc_phi, relax_phi, neutral_phi) - \
                              min(conc_phi, relax_phi, neutral_phi) + 1e-8
                nct_score = (conc_phi - neutral_phi) / phi_range  # 判别力归一化
                nct_score = float(np.clip(nct_score, 0, 1))
                scores[2] = nct_score

            colors = ["#3498DB", "#E67E22", "#E74C3C"]
            bars   = ax3.bar(methods, scores, color=colors, width=0.5)
            ax3.set_ylabel("宏平均 F1 / Φ 判别力")
            ax3.set_title("Phase 3 – MEMA EEG\n三方对比（SVM / LSTM / NCT）")
            ax3.set_ylim(0, 1.1)
            ax3.grid(True, alpha=0.3)
            for bar, val in zip(bars, scores):
                ax3.text(bar.get_x() + bar.get_width() / 2,
                         val + 0.02, f"{val:.3f}",
                         ha="center", va="bottom", fontsize=9)
            # 显著性标注
            t_res = phi_b.get("t_test_relax_vs_conc", {}) or {}
            if t_res.get("significant"):
                ax3.text(0.98, 0.95, f"Φ t检验 p={t_res.get('p_value','?'):.4f} *",
                         transform=ax3.transAxes, ha="right", fontsize=8,
                         color="red", va="top")
        else:
            ax3.text(0.5, 0.5, "Phase 3 未运行", ha="center", transform=ax3.transAxes)

        # ── 子图 4：Phase 4 学习曲线对比 ──
        ax4 = axes[1][1]
        p4  = all_results.get("phase4")
        if p4:
            fixed_curve    = p4.get("learning_curves", {}).get("fixed_avg",    [])
            adaptive_curve = p4.get("learning_curves", {}).get("adaptive_avg", [])
            if fixed_curve and adaptive_curve:
                n = min(len(fixed_curve), len(adaptive_curve))
                xs = range(1, n + 1)
                ax4.plot(xs, fixed_curve[:n],    label="固定难度",
                         color="#7F8C8D", linewidth=2, linestyle="--")
                ax4.plot(xs, adaptive_curve[:n], label="NCT 自适应",
                         color="#E74C3C", linewidth=2)
                ax4.fill_between(xs, fixed_curve[:n], adaptive_curve[:n],
                                 alpha=0.15, color="#E74C3C", label="提升区域")
                ax4.set_xlabel("答题序列步骤")
                ax4.set_ylabel("滑动窗口准确率")
                ax4.set_title("Phase 4 – EdNet\nNCT 自适应 vs 固定难度学习曲线")
                ax4.legend(fontsize=9)
                ax4.grid(True, alpha=0.3)
                delta_auc = p4.get("improvements", {}).get("auc_delta", "?")
                p_val     = p4.get("improvements", {}).get("auc_t_test", {}).get("p", "?")
                ax4.text(0.05, 0.93,
                         f"ΔAUC={delta_auc}  p={p_val}",
                         transform=ax4.transAxes, fontsize=9, color="darkblue", va="top")
            else:
                ax4.text(0.5, 0.5, "学习曲线数据不足",
                         ha="center", transform=ax4.transAxes)
        else:
            ax4.text(0.5, 0.5, "Phase 4 未运行", ha="center", transform=ax4.transAxes)

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        out_path = RESULTS_DIR / "education_nct_combined_report.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"综合报告图已保存：{out_path}")
    except Exception as e:
        logger.warning(f"综合可视化失败：{e}")
        import traceback
        traceback.print_exc()


# ==============================================================================
# 3. 打印最终摘要
# ==============================================================================

def _print_summary(all_results: Dict, elapsed: float):
    """打印跨 Phase 综合摘要表"""
    print("\n" + "=" * 72)
    print("  NCT 教育实验 综合摘要")
    print("=" * 72)
    print(f"  总用时：{elapsed:.1f} 秒\n")

    # Phase 2
    p2 = all_results.get("phase2")
    if p2:
        print(f"  [Phase 2] FER 预训练")
        print(f"      最优验证准确率 : {p2.get('best_val_acc', '?')}")
        print(f"      情绪类别数     : {p2.get('n_classes', 7)}")
        print(f"      Checkpoint     : checkpoints/fer_pretrain/fer_best.pt")

    # Phase 1
    p1 = all_results.get("phase1")
    if p1:
        corr = p1.get("correlations", {}).get("engagement_level_vs_phi", {})
        print(f"\n  [Phase 1] DAiSEE 视觉验证")
        print(f"      样本数         : {p1.get('n_samples', '?')}")
        print(f"      FER 增强       : {p1.get('fer_enhanced', False)}")
        print(f"      Engagement×Φ r : {corr.get('r', '?')}  p={corr.get('p', '?')}")

    # Phase 3
    p3 = all_results.get("phase3")
    if p3:
        clf     = p3.get("experiment_A_classification", {})
        phi_b   = p3.get("experiment_B_phi_analysis", {})
        t_res   = phi_b.get("t_test_relax_vs_conc", {}) or {}
        print(f"\n  [Phase 3] MEMA EEG 神经调质映射（核心实验）")
        print(f"      SVM 宏F1       : {clf.get('SVM',{}).get('mean_f1','?')}")
        print(f"      LSTM 宏F1      : {clf.get('LSTM',{}).get('mean_f1','?')}")
        print(f"      Φ t检验显著    : {t_res.get('significant','?')}  "
              f"p={t_res.get('p_value','?')}")
        conc_phi = phi_b.get("phi_stats", {}).get("concentrating", {})
        rel_phi  = phi_b.get("phi_stats", {}).get("relaxing",      {})
        print(f"      Concentrating Φ: {conc_phi.get('mean','?'):.4f}"
              if isinstance(conc_phi.get("mean"), float) else
              f"      Concentrating Φ: {conc_phi.get('mean','?')}")
        print(f"      Relaxing Φ     : {rel_phi.get('mean','?'):.4f}"
              if isinstance(rel_phi.get("mean"), float) else
              f"      Relaxing Φ     : {rel_phi.get('mean','?')}")

    # Phase 4
    p4 = all_results.get("phase4")
    if p4:
        impr = p4.get("improvements", {})
        fix  = p4.get("fixed_strategy", {})
        adp  = p4.get("adaptive_strategy", {})
        print(f"\n  [Phase 4] EdNet 自适应学习")
        print(f"      ΔAUC (自适应-固定): {impr.get('auc_delta', '?')}")
        print(f"      t 检验 p 值        : {impr.get('auc_t_test', {}).get('p', '?')}")
        print(f"      收敛准确率(自适应) : {adp.get('mean_conv_acc','?')}")
        print(f"      收敛准确率(固定)   : {fix.get('mean_conv_acc','?')}")

    print("\n  输出文件：")
    for fname in sorted(RESULTS_DIR.glob("*.json")):
        print(f"      {fname.name}")
    for fname in sorted(RESULTS_DIR.glob("*.png")):
        print(f"      {fname.name}")
    print("=" * 72)


# ==============================================================================
# 4. 主入口
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="NCT 教育实验全量运行器（Phase 1-4）"
    )
    parser.add_argument("--mock",       action="store_true",
                        help="所有 Phase 使用 Mock 数据（无真实数据集时）")
    parser.add_argument("--epochs",     type=int, default=5,
                        help="Phase 2 FER 训练轮数（默认 5）")
    parser.add_argument("--clips",      type=int, default=100,
                        help="Phase 1 DAiSEE 处理片段数（默认 100）")
    parser.add_argument("--students",   type=int, default=30,
                        help="Phase 4 EdNet 学生数（默认 30）")
    parser.add_argument("--skip-2",     action="store_true",
                        help="跳过 Phase 2（使用已有 checkpoint）")
    parser.add_argument("--no-fer",     action="store_true",
                        help="Phase 1 不使用 FER 模型（纯规则视觉特征）")
    args = parser.parse_args()

    t0 = time.time()
    all_results = {}

    logger.info("\n" + "=" * 70)
    logger.info("   NCT 教育领域实验 — 全量运行器  START")
    logger.info(f"   Mock 模式: {args.mock}")
    logger.info("=" * 70)

    # ---- Phase 2: FER 预训练 ----
    if not args.skip_2:
        r2 = run_phase2_fer(use_mock=args.mock, n_epochs=args.epochs)
        all_results["phase2"] = r2
    else:
        logger.info("\n[跳过 Phase 2，使用已有 checkpoint]")
        all_results["phase2"] = None

    # ---- Phase 1: DAiSEE ----
    r1 = run_phase1_daisee(
        use_mock=args.mock,
        max_clips=args.clips,
        use_fer=(not args.no_fer),
    )
    all_results["phase1"] = r1

    # ---- Phase 3: MEMA EEG ----
    r3 = run_phase3_mema(use_mock=args.mock)
    all_results["phase3"] = r3

    # ---- Phase 4: EdNet ----
    r4 = run_phase4_ednet(use_mock=args.mock, n_students=args.students)
    all_results["phase4"] = r4

    # ---- 综合可视化 ----
    logger.info("\n生成跨 Phase 综合可视化...")
    _plot_combined_summary(all_results)

    # ---- 保存总报告 ----
    summary_path = RESULTS_DIR / "education_nct_combined_results.json"
    # 只保存可序列化的部分（忽略 numpy 数组）
    serializable = {}
    for k, v in all_results.items():
        if v is not None:
            try:
                json.dumps(v)  # 测试是否可序列化
                serializable[k] = v
            except (TypeError, ValueError):
                serializable[k] = str(v)
        else:
            serializable[k] = None
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"综合结果已保存：{summary_path}")

    # ---- 摘要输出 ----
    elapsed = time.time() - t0
    _print_summary(all_results, elapsed)


if __name__ == "__main__":
    main()
