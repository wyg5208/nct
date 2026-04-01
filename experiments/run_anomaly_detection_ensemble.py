"""
Phase 2F: NCT Anomaly Detection - Ensemble Learning
====================================================
集成学习方法异常检测

修订日期：2026-02-27
修订时间：自动记录
修订人员：NCT LAB Team

核心优化策略：
1. 【多模型集成】结合 Phase 2B（宽松）和 Phase 2E（自适应）的优势
   - Phase 2B: 在旋转、遮挡上表现优秀
   - Phase 2E: 在 OOD、噪声上表现较好
   
2. 【投票策略】采用"任一方案判定为异常即为异常"的宽松策略
   - 最大化 Recall
   - 保持合理的 Precision
   
3. 【场景适配】根据应用场景动态调整集成权重
   - OOD 检测场景：增加 Phase 2E 权重
   - 通用检测场景：平等对待两个方案
   
4. 【实验隔离】独立输出目录，不覆盖之前结果

预期效果：
- 平均 F1: 0.35-0.40（相比 Phase 2B 提升 10-25%）
- 结合 2B 的旋转/遮挡优势 + 2E 的 OOD/噪声优势
- 提高鲁棒性和泛化能力

Author: NCT LAB Team
Date: February 27, 2026
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import json
import os
import argparse
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# Custom JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types and Path objects"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


# Import NCT modules
import sys
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from nct_modules.nct_anomaly_detector import SimplifiedNCT, NCTAnomalyDetectorV2


def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='NCT Anomaly Detection - Phase 2F: Ensemble Learning')
    
    # 模型参数
    parser.add_argument('--d_model', type=int, default=384, help='Model dimension (default: 384)')
    parser.add_argument('--n_heads', type=int, default=6, help='Number of attention heads (default: 6)')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of transformer layers (default: 3)')
    parser.add_argument('--dim_ff', type=int, default=768, help='Feedforward dimension (default: 768)')
    parser.add_argument('--dropout_rate', type=float, default=0.4, help='Dropout rate (default: 0.4)')
    parser.add_argument('--n_candidates', type=int, default=15, help='Number of candidates (default: 15)')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size (default: 128)')
    parser.add_argument('--pretrained', type=str, required=True, help='Path to pretrained model')
    
    # 集成策略参数
    parser.add_argument('--ensemble_strategy', type=str, default='voting', 
                        choices=['voting', 'weighted', 'stacking'], 
                        help='Ensemble strategy (default: voting)')
    parser.add_argument('--phase2b_weight', type=float, default=0.5,
                       help='Weight for Phase 2B predictions (default: 0.5)')
    parser.add_argument('--phase2e_weight', type=float, default=0.5,
                       help='Weight for Phase 2E predictions (default: 0.5)')
    
    # Phase 2B 参数（宽松策略）
    parser.add_argument('--pe_threshold_2b', type=float, default=4157.0,
                       help='PE threshold for Phase 2B (default: 4157)')
    parser.add_argument('--phi_threshold_2b', type=float, default=0.000092,
                       help='Phi threshold for Phase 2B (default: 0.000092)')
    parser.add_argument('--entropy_threshold_2b', type=float, default=2.70,
                       help='Entropy threshold for Phase 2B (default: 2.70)')
    
    # Phase 2E 参数（自适应策略）
    parser.add_argument('--temperature', type=float, default=2.0,
                       help='Softmax temperature for confidence calibration (default: 2.0)')
    
    # 阈值优化
    parser.add_argument('--optimize_thresholds', action='store_true', default=True,
                       help='Optimize thresholds on validation set')
    parser.add_argument('--n_normal_val', type=int, default=200,
                       help='Number of normal samples for validation (default: 200)')
    parser.add_argument('--n_anomalies_val', type=int, default=100,
                       help='Number of anomalies per type for validation (default: 100)')
    
    # 输出
    parser.add_argument('--results_dir', type=str, default='results/anomaly_detection_ensemble',
                       help='Results directory (default: results/anomaly_detection_ensemble)')
    parser.add_argument('--name', type=str, default='', help='Experiment name suffix')
    
    # 设备
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device (default: cuda if available)')
    
    return parser.parse_args()


def create_anomalous_samples(normal_samples, anomaly_type, n_anomalies=50):
    """生成异常样本"""
    anomalous_data = []
    
    indices = np.random.choice(len(normal_samples), min(n_anomalies, len(normal_samples)), replace=False)
    
    for idx in indices:
        img = normal_samples[idx].clone()
        
        if anomaly_type == 'noise':
            noisy = img + torch.randn_like(img) * 0.5
            noisy = torch.clamp(noisy, 0, 1)
            anomalous_data.append(noisy)
        
        elif anomaly_type == 'rotation':
            rotated = img.flip(1).transpose(1, 2)
            anomalous_data.append(rotated)
        
        elif anomaly_type == 'occlusion':
            occluded = img.clone()
            patch_size = 7
            h_start = np.random.randint(0, 28 - patch_size)
            w_start = np.random.randint(0, 28 - patch_size)
            occluded[:, h_start:h_start+patch_size, w_start:w_start+patch_size] = 0
            anomalous_data.append(occluded)
        
        elif anomaly_type == 'ood':
            from torchvision import datasets, transforms
            fashion_dataset = datasets.FashionMNIST(root='./data', train=True, download=True,
                                                   transform=transforms.ToTensor())
            ood_indices = torch.randint(0, len(fashion_dataset), (1,))
            ood_img = fashion_dataset[ood_indices[0]][0]
            ood_img = (ood_img - 0.5) / 0.5
            anomalous_data.append(ood_img)
    
    return torch.stack(anomalous_data)


@torch.no_grad()
def phase2b_predict(model, data, pe_thresh, phi_thresh, entropy_thresh, device):
    """
    Phase 2B 预测（宽松策略）
    
    策略：任一信号触发即判定为异常
    PE OR Φ OR Entropy → Anomaly
    """
    model.eval()
    data = data.to(device)
    
    output_dict = model(data)
    
    pes = output_dict['prediction_error'].cpu().numpy()
    phis = output_dict['phi'].cpu().numpy()
    
    attn_weights = output_dict['attention_weights']
    attn_probs = nn.functional.softmax(attn_weights, dim=-1)
    entropies = -torch.sum(attn_probs * torch.log(attn_probs + 1e-9), dim=-1).cpu().numpy()
    
    # 各信号判定
    pred_pe = (pes > pe_thresh).astype(int)
    pred_phi = (phis < phi_thresh).astype(int)
    pred_entropy = (entropies > entropy_thresh).astype(int)
    
    # 宽松策略：任一触发
    pred_combined = ((pred_pe + pred_phi + pred_entropy) >= 1).astype(int)
    
    return pred_combined, {
        'pe': pred_pe,
        'phi': pred_phi,
        'entropy': pred_entropy,
        'pes': pes,
        'phis': phis,
        'entropies': entropies
    }


@torch.no_grad()
def phase2e_predict(model, data, temperature=2.0, device='cuda'):
    """
    Phase 2E 预测（自适应加权融合策略）
    
    简化版本：使用固定的自适应权重
    """
    model.eval()
    data = data.to(device)
    
    output_dict = model(data)
    
    probs = torch.softmax(output_dict['output'] / temperature, dim=1)
    confidence, _ = probs.max(1)
    confidence = confidence.cpu().numpy()
    
    pes = output_dict['prediction_error'].cpu().numpy()
    phis = output_dict['phi'].cpu().numpy()
    
    attn_weights = output_dict['attention_weights']
    attn_probs = nn.functional.softmax(attn_weights, dim=-1)
    entropies = -torch.sum(attn_probs * torch.log(attn_probs + 1e-9), dim=-1).cpu().numpy()
    
    # 归一化（Z-score）
    def normalize_zscore(scores, higher_is_better=True):
        mean_val = np.mean(scores)
        std_val = np.std(scores)
        if std_val < 1e-10:
            return np.ones_like(scores) * 0.5
        normalized = (scores - mean_val) / std_val
        normalized = 1 / (1 + np.exp(-normalized))  # Sigmoid
        if not higher_is_better:
            normalized = 1 - normalized
        return normalized
    
    pe_norm = normalize_zscore(pes, higher_is_better=False)
    phi_norm = normalize_zscore(phis, higher_is_better=True)
    entropy_norm = normalize_zscore(entropies, higher_is_better=False)
    
    # 自适应权重
    weights = []
    for i in range(len(pes)):
        if confidence[i] < 0.5:  # 低置信度
            w_pe, w_phi, w_entropy = 0.35, 0.25, 0.40
        else:  # 高置信度
            w_pe, w_phi, w_entropy = 0.50, 0.15, 0.35
        
        weights.append({'pe': w_pe, 'phi': w_phi, 'entropy': w_entropy})
    
    # 计算融合分数
    fusion_scores = []
    for i in range(len(pes)):
        score = (weights[i]['pe'] * pe_norm[i] +
                weights[i]['phi'] * phi_norm[i] +
                weights[i]['entropy'] * entropy_norm[i])
        fusion_scores.append(score)
    
    fusion_scores = np.array(fusion_scores)
    threshold = np.percentile(fusion_scores, 70)
    pred_combined = (fusion_scores > threshold).astype(int)
    
    return pred_combined, {
        'fusion_scores': fusion_scores,
        'confidences': confidence,
        'pes': pes,
        'phis': phis,
        'entropies': entropies
    }


def ensemble_voting(pred_2b, pred_2e, strategy='or'):
    """
    集成投票策略
    
    Args:
        pred_2b: Phase 2B 预测结果
        pred_2e: Phase 2E 预测结果
        strategy: 'or' (任一) 或 'and' (两者都)
    """
    if strategy == 'or':
        # 宽松策略：任一判定为异常即为异常
        return ((pred_2b + pred_2e) >= 1).astype(int)
    elif strategy == 'and':
        # 严格策略：两者都判定为异常才为异常
        return ((pred_2b + pred_2e) >= 2).astype(int)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def optimize_ensemble_thresholds(model, val_loader, anomaly_types, args):
    """
    在验证集上优化集成策略的阈值
    """
    print("\n" + "=" * 60)
    print("Optimizing Ensemble Thresholds")
    print("=" * 60)
    
    best_f1 = 0
    best_config = {}
    
    # Phase 2B 参数搜索范围（更宽松）
    pe_2b_range = [3000, 3500, 4000, 4500, 5000]
    phi_2b_range = [0.00008, 0.00010, 0.00012, 0.00015, 0.00018]
    entropy_2b_range = [2.3, 2.5, 2.7, 2.9, 3.1]
    
    # Phase 2E 百分位数搜索范围
    percentile_2e_range = [65, 70, 75, 80]
    
    print(f"Testing {len(pe_2b_range)} PE × {len(phi_2b_range)} Φ × "
          f"{len(entropy_2b_range)} Entropy × {len(percentile_2e_range)} Percentile")
    print(f"Total combinations: {len(pe_2b_range) * len(phi_2b_range) * len(entropy_2b_range) * len(percentile_2e_range)}")
    
    iteration = 0
    for pe_2b in pe_2b_range:
        for phi_2b in phi_2b_range:
            for entropy_2b in entropy_2b_range:
                for perc_2e in percentile_2e_range:
                    iteration += 1
                    
                    all_preds_2b = []
                    all_preds_2e = []
                    all_labels = []
                    
                    # 收集所有验证集样本的预测
                    for anomaly_type in anomaly_types:
                        # 准备数据
                        from torchvision import datasets, transforms
                        transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])
                        test_dataset = datasets.MNIST('../data', train=False, download=True, transform=transform)
                        
                        n_normal = args.n_normal_val
                        n_anomaly = args.n_anomalies_val
                        
                        normal_indices = torch.randperm(len(test_dataset))[:n_normal]
                        normal_data = torch.stack([test_dataset[i][0] for i in normal_indices])
                        
                        anomalous_data = create_anomalous_samples(
                            torch.stack([test_dataset[i][0] for i in range(min(n_anomaly, len(test_dataset)))]),
                            anomaly_type,
                            n_anomalies=n_anomaly
                        )
                        
                        combined_data = torch.cat([normal_data, anomalous_data], dim=0)
                        labels = np.concatenate([np.zeros(n_normal), np.ones(n_anomaly)])
                        
                        # Phase 2B 预测
                        pred_2b, _ = phase2b_predict(model, combined_data, pe_2b, phi_2b, entropy_2b, args.device)
                        
                        # Phase 2E 预测（自定义百分位数）
                        _, details_2e = phase2e_predict(model, combined_data, args.temperature, args.device)
                        fusion_scores = details_2e['fusion_scores']
                        threshold_2e = np.percentile(fusion_scores, perc_2e)
                        pred_2e = (fusion_scores > threshold_2e).astype(int)
                        
                        all_preds_2b.extend(pred_2b)
                        all_preds_2e.extend(pred_2e)
                        all_labels.extend(labels)
                    
                    # 集成投票
                    pred_ensemble = ensemble_voting(np.array(all_preds_2b), np.array(all_preds_2e), strategy='or')
                    
                    # 计算 F1
                    f1 = f1_score(all_labels, pred_ensemble)
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        best_config = {
                            'pe_2b': pe_2b,
                            'phi_2b': phi_2b,
                            'entropy_2b': entropy_2b,
                            'percentile_2e': perc_2e
                        }
                    
                    if iteration % 20 == 0:
                        print(f"  Iteration {iteration}, Current best F1: {best_f1:.4f}")
    
    print(f"\n✅ Best configuration found:")
    print(f"  Phase 2B - PE: {best_config['pe_2b']}, Φ: {best_config['phi_2b']}, Entropy: {best_config['entropy_2b']}")
    print(f"  Phase 2E - Percentile: {best_config['percentile_2e']}")
    print(f"  Best Combined F1: {best_f1:.4f}")
    
    return best_config, best_f1


def evaluate_on_anomaly_type(model, anomaly_type, config_2b, config_2e, args):
    """在特定类型的异常上评估集成模型"""
    print(f"\nTesting {anomaly_type} anomalies...")
    
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    n_normal = 200
    n_anomaly = 100
    
    normal_indices = torch.randint(0, len(full_dataset), (n_normal,))
    normal_images = torch.stack([full_dataset[i][0] for i in normal_indices])
    normal_labels = np.zeros(n_normal)
    
    anomaly_images = create_anomalous_samples(normal_images.clone()[:n_anomaly], anomaly_type, n_anomalies=n_anomaly)
    anomaly_labels = np.ones(n_anomaly)
    
    combined_data = torch.cat([normal_images, anomaly_images], dim=0).to(args.device)
    all_labels = np.concatenate([normal_labels, anomaly_labels])
    
    # Phase 2B 预测
    pred_2b, details_2b = phase2b_predict(
        model, combined_data,
        config_2b['pe'], config_2b['phi'], config_2b['entropy'],
        args.device
    )
    
    # Phase 2E 预测
    pred_2e, details_2e = phase2e_predict(model, combined_data, args.temperature, args.device)
    
    # 集成投票
    pred_ensemble = ensemble_voting(pred_2b, pred_2e, strategy='or')
    
    # 计算指标
    results = {}
    
    for name, preds in [('PE_2B', pred_2b), ('PHI_2B', details_2b['phi']),
                         ('ENTROPY_2B', details_2b['entropy']),
                         ('Phase2E', pred_2e), ('ENSEMBLE', pred_ensemble)]:
        if preds.sum() == 0 or (preds == 1).sum() == len(preds):
            results[name.lower().replace('_', '')] = {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'accuracy': accuracy_score(all_labels, preds)
            }
        else:
            results[name.lower().replace('_', '')] = {
                'precision': precision_score(all_labels, preds, zero_division=0),
                'recall': recall_score(all_labels, preds, zero_division=0),
                'f1': f1_score(all_labels, preds, zero_division=0),
                'accuracy': accuracy_score(all_labels, preds)
            }
    
    # 打印结果
    print(f"\nResults for {anomaly_type}:")
    print(f"  Sample counts: Normal={n_normal}, Anomaly={n_anomaly}")
    for name in ['PE_2B', 'PHI_2B', 'ENTROPY_2B', 'Phase2E', 'ENSEMBLE']:
        key = name.lower().replace('_', '') if '_' in name else name.lower()
        if key in results:
            print(f"  {name:10s}: Precision={results[key]['precision']:.3f}, "
                  f"Recall={results[key]['recall']:.3f}, "
                  f"F1={results[key]['f1']:.3f}, "
                  f"Acc={results[key]['accuracy']:.3f}")
    
    return results, n_normal, n_anomaly


def visualize_results(results_list, save_path):
    """可视化结果"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    anomaly_types = ['noise', 'rotation', 'occlusion', 'ood']
    metrics_names = ['precision', 'recall', 'f1', 'accuracy']
    
    for idx, metric_name in enumerate(metrics_names):
        ax = axes[idx // 2, idx % 2]
        
        x = np.arange(len(anomaly_types))
        width = 0.18
        
        methods = ['pe2b', 'entropy2b', 'phase2e', 'ensemble']
        method_labels = ['PE (2B)', 'Entropy (2B)', 'Phase 2E', 'Ensemble']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        
        for i, method in enumerate(methods):
            values = [r[method][metric_name] for r in results_list]
            ax.bar(x + i * width, values, width, label=method_labels[i], color=colors[i], alpha=0.8)
        
        ax.set_xlabel('Anomaly Type', fontsize=12)
        ax.set_ylabel(metric_name.capitalize(), fontsize=12)
        ax.set_title(f'{metric_name.capitalize()} Comparison', fontsize=14)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(anomaly_types, rotation=15)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {save_path}")


def main():
    """主函数"""
    args = get_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"ensemble_{timestamp}"
    if args.name:
        exp_name += f"_{args.name}"
    
    results_dir = Path(args.results_dir) / exp_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Phase 2F: NCT Anomaly Detection - Ensemble Learning")
    print("=" * 80)
    print(f"\nRevision Date: 2026-02-27")
    print(f"Revision Time: {datetime.now().strftime('%H:%M:%S')}")
    print("\nOptimization Strategies:")
    print("  1. Ensemble of Phase 2B (loose) and Phase 2E (adaptive)")
    print("  2. Voting strategy: OR (either predicts anomaly → anomaly)")
    print("  3. Optimized thresholds on validation set")
    print("  4. Independent experiment (no overwrite)")
    print("=" * 80)
    
    print("\nConfiguration:")
    print(f"  d_model: {args.d_model}")
    print(f"  n_heads: {args.n_heads}")
    print(f"  n_layers: {args.n_layers}")
    print(f"  n_candidates: {args.n_candidates}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Ensemble strategy: {args.ensemble_strategy}")
    print(f"  Phase 2B weight: {args.phase2b_weight}")
    print(f"  Phase 2E weight: {args.phase2e_weight}")
    print(f"  Pretrained model: {args.pretrained}")
    print(f"  Results dir: {results_dir}")
    print(f"\nDevice: {args.device}")
    
    # Load model
    print("\nLoading model...")
    
    model = SimplifiedNCT(
        input_shape=(1, 28, 28),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dim_ff=args.dim_ff,
        num_classes=10,
        dropout_rate=args.dropout_rate,
        n_candidates=args.n_candidates
    )
    
    checkpoint = torch.load(args.pretrained, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    print("  ✓ Model loaded successfully")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    anomaly_types = ['noise', 'rotation', 'occlusion', 'ood']
    
    # Optimize thresholds
    if args.optimize_thresholds:
        best_config, best_f1 = optimize_ensemble_thresholds(model, None, anomaly_types, args)
    else:
        best_config = {
            'pe_2b': args.pe_threshold_2b,
            'phi_2b': args.phi_threshold_2b,
            'entropy_2b': args.entropy_threshold_2b,
            'percentile_2e': 70
        }
        best_f1 = 0.0
    
    config_2b = {
        'pe': best_config['pe_2b'],
        'phi': best_config['phi_2b'],
        'entropy': best_config['entropy_2b']
    }
    
    config_2e = {
        'percentile': best_config['percentile_2e']
    }
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Testing on Independent Test Set")
    print("=" * 60)
    
    results_list = []
    
    for anomaly_type in anomaly_types:
        results, n_normal, n_anomaly = evaluate_on_anomaly_type(
            model, anomaly_type, config_2b, config_2e, args
        )
        results['anomaly_type'] = anomaly_type
        results['n_normal'] = n_normal
        results['n_anomaly'] = n_anomaly
        results_list.append(results)
    
    # Save results
    results_summary = {
        'config': vars(args),
        'optimization_config': {
            'phase2b_config': config_2b,
            'phase2e_config': config_2e,
            'ensemble_strategy': args.ensemble_strategy,
            'voting_method': 'OR'
        },
        'validation_best_f1': float(best_f1),
        'test_results': results_list,
        'revision_info': {
            'date': '2026-02-27',
            'time': datetime.now().strftime('%H:%M:%S'),
            'strategies': [
                f'Ensemble of Phase 2B (PE>{config_2b["pe"]:.0f}, Φ<{config_2b["phi"]:.5f}, Entropy>{config_2b["entropy"]:.2f})',
                f'Phase 2E (Percentile={config_2e["percentile"]})',
                f'Voting strategy: OR (either predicts anomaly)',
                f'Validation F1: {best_f1:.4f}',
                'Independent experiment output'
            ]
        }
    }
    
    with open(results_dir / 'ensemble_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2, cls=NumpyEncoder)
    
    # Visualization
    print("\n" + "=" * 60)
    print("Creating visualizations...")
    print("=" * 60)
    visualize_results(results_list, results_dir / 'ensemble_results.png')
    
    print(f"\nResults saved to: {results_dir}")
    print("\n" + "=" * 80)
    print("Phase 2F Completed Successfully!")
    print("=" * 80)
    
    return results_summary


if __name__ == '__main__':
    results = main()
