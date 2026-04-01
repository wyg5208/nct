"""
Phase 2: FER2013 / RAF-DB 面部表情预训   
==========================================
训练轻量 CNN 表情分类器，   7 类情绪映射到 NCT 神经调质参数   
输出模型   Phase 1 视觉特征提取模块调用   

FER2013 数据格式 (Kaggle 下载，CSV 格式)：
  - emotion: 0-6 (Angry/Disgust/Fear/Happy/Sad/Surprise/Neutral)
  - pixels:  48×48 空格分隔灰度   
  - Usage:   Training / PublicTest / PrivateTest

下载地址 (Kaggle)：https://www.kaggle.com/datasets/msambare/fer2013
数据放置：data/fer2013/fer2013.csv

情绪 -> 神经调质映射 (基于神经科学文献)
  Happy       DA    5-HT           # 奖励/满足   
  Surprise    DA    NE             # 新奇/警觉
  Fear        NE↑↑, 5-HT         # 应激/威胁
  Angry       NE    DA             # 攻击性唤   
  Sad         5-HT    DA           # 抑郁/动机   
  Disgust     ACh    5-HT         # 厌恶/回避
  Neutral     基准 0.5            # 稳   

作者：NCT LAB Team
日期   026-03-15
"""

import sys
import json
import logging
import time
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR    = PROJECT_ROOT / "data" / "fer2013"
RESULTS_DIR = PROJECT_ROOT / "results" / "education"
CKPT_DIR    = PROJECT_ROOT / "checkpoints" / "fer_pretrain"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---- FER2013 情绪类别 ----
EMOTION_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# ---- 情绪 -> 神经调质映射表 (偏差量，基准 0.5) ---
EMOTION_NM_MAP: Dict[str, Dict[str, float]] = {
    "Happy":    {"DA":  0.3,  "5-HT":  0.2, "NE":  0.0, "ACh":  0.1},
    "Surprise": {"DA":  0.2,  "5-HT":  0.0, "NE":  0.3, "ACh":  0.0},
    "Fear":     {"DA": -0.1,  "5-HT": -0.3, "NE":  0.4, "ACh":  0.0},
    "Angry":    {"DA": -0.2,  "5-HT": -0.1, "NE":  0.3, "ACh":  0.0},
    "Sad":      {"DA": -0.25, "5-HT": -0.2, "NE": -0.1, "ACh": -0.1},
    "Disgust":  {"DA": -0.15, "5-HT": -0.2, "NE":  0.1, "ACh":  0.2},
    "Neutral":  {"DA":  0.0,  "5-HT":  0.0, "NE":  0.0, "ACh":  0.0},
}


def emotion_to_neuromodulator(emotion_idx: int) -> Dict[str, float]:
    """
    将情绪索引映射到神经调质字典   

    Args:
        emotion_idx: 0-6 (对应 EMOTION_NAMES)

    Returns:
        nm_state: {'DA': float, '5-HT': float, 'NE': float, 'ACh': float}
    """
    name   = EMOTION_NAMES[emotion_idx]
    deltas = EMOTION_NM_MAP.get(name, {})
    baseline = 0.5
    return {
        k: float(np.clip(baseline + deltas.get(k, 0.0), 0.0, 1.0))
        for k in ["DA", "5-HT", "NE", "ACh"]
    }


# ==============================================================================
# 1. 数据集加   
# ==============================================================================

class FERDataset:
    """FER2013 数据集加载器 (CSV 格式 / HuggingFace 自动回退) """

    HF_REPO = "3una/Fer2013"   # HuggingFace 镜像 (无需认证)

    def __init__(self, csv_path: Path, usage: str = "Training", max_samples: int = 10000):
        self.csv_path    = csv_path
        self.usage       = usage
        self.max_samples = max_samples
        self.images: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None

    def is_available(self) -> bool:
        return self.csv_path.exists()

    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        """加载图像 [N, 1, 48, 48] 和标   [N]"""
        if self.images is not None:
            return self.images, self.labels

        logger.info(f"加载 FER2013：{self.csv_path} (usage={self.usage})")
        images, labels = [], []
        import csv
        with open(self.csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("Usage", "") != self.usage:
                    continue
                label = int(row["emotion"])
                pixels = np.array(row["pixels"].split(), dtype=np.float32) / 255.0
                img = pixels.reshape(1, 48, 48)  # [C, H, W]
                images.append(img)
                labels.append(label)
                if len(images) >= self.max_samples:
                    break

        self.images = np.array(images, dtype=np.float32)
        self.labels = np.array(labels, dtype=np.int64)
        logger.info(f"  加载完成：{len(self.images)} 张，7 类)")
        return self.images, self.labels

    # ------------------------------------------------------------------
    # HuggingFace 自动下载 (无需 Kaggle 账号)
    # ------------------------------------------------------------------
    def download_from_huggingface(
        self,
        max_samples: Optional[int] = None,
        save_csv: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        从 HuggingFace 镜像 3una/Fer2013 下载数据

        会自动将下载结果保存为 CSV (下次可直接加载)，无需 Kaggle 认证

        Args:
            max_samples: 限制样本数 (None = 全量 ~28,709 train)
            save_csv:    是否缓存为 fer2013.csv

        Returns:
            images [N,1,48,48], labels [N]
        """
        import os
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

        try:
            from datasets import load_dataset
        except ImportError:
            raise RuntimeError("请先安装 datasets 库：pip install datasets")

        limit = max_samples or self.max_samples
        hf_split = "train" if self.usage == "Training" else "test"

        logger.info(f"  从 HuggingFace 下载 FER2013 ({self.HF_REPO}, split={hf_split})")
        ds = load_dataset(self.HF_REPO, split=hf_split, streaming=False)
        logger.info(f"  数据集大小：{len(ds)} 张，开始转换")

        images, labels_list = [], []
        for i, item in enumerate(ds):
            if i >= limit:
                break
            img_pil = item["image"]
            # 确保灰度 48x48
            if img_pil.mode != "L":
                img_pil = img_pil.convert("L")
            arr = np.array(img_pil, dtype=np.float32) / 255.0  # [48,48]
            images.append(arr.reshape(1, 48, 48))              # [1,48,48]
            labels_list.append(int(item["label"]))

            if (i + 1) % 5000 == 0:
                logger.info(f"  已处理 {i+1}/{min(limit, len(ds))} 张")

        images_arr = np.array(images, dtype=np.float32)
        labels_arr = np.array(labels_list, dtype=np.int64)
        logger.info(f"  转换完成：{len(images_arr)} 张")

        # 缓存为 CSV (使得下次可无网络加载)
        if save_csv:
            self._save_as_csv(images_arr, labels_arr, hf_split)

        self.images = images_arr
        self.labels = labels_arr
        return self.images, self.labels

    def _save_as_csv(self, images: np.ndarray, labels: np.ndarray, usage_tag: str):
        """将 numpy 数组保存为 fer2013.csv (或追加到已有 CSV) """
        import csv
        usage_map = {"train": "Training", "test": "PublicTest"}
        usage_str = usage_map.get(usage_tag, usage_tag)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if self.csv_path.exists() else "w"
        with open(self.csv_path, mode, newline="") as f:
            writer = csv.writer(f)
            if mode == "w":
                writer.writerow(["emotion", "pixels", "Usage"])
            for img, lbl in zip(images, labels):
                pixels_str = " ".join(f"{int(round(v * 255))}" for v in img.flatten())
                writer.writerow([int(lbl), pixels_str, usage_str])
        logger.info(f"  CSV 已缓存：{self.csv_path}")

    def generate_mock(self, n: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
        """生成 Mock 数据 (无真实数据时使用)"""
        np.random.seed(42)
        images = np.random.rand(n, 1, 48, 48).astype(np.float32)
        labels = np.random.randint(0, 7, n).astype(np.int64)
        # 为不同情绪注入亮度信号 (让模型有东西可学)
        for i in range(n):
            bias = labels[i] / 7.0
            images[i] += bias * 0.3
        images = np.clip(images, 0, 1)
        return images, labels


# ==============================================================================
# 2. 轻量 CNN 模型（PyTorch   
# ==============================================================================

def build_fer_model(num_classes: int = 7):
    """构建轻量面部表情识别 CNN"""
    import torch.nn as nn

    class FERNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                # Block 1: 48   4
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.2),
                # Block 2: 24   2
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.2),
                # Block 3: 12   
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.3),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 6 * 6, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(256, num_classes),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    return FERNet()


# ==============================================================================
# 3. 训练逻辑
# ==============================================================================

def train_fer_model(
    images: np.ndarray,
    labels: np.ndarray,
    n_epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> Dict:
    """
    训练 FER 模型并保   checkpoint   

    Returns:
        metrics: 包含训练/验证精度、每类精度、损失曲   
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader, random_split

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"训练设备：{device}")

    # ---- 数据集划   8:2 ----
    X = torch.from_numpy(images)
    y = torch.from_numpy(labels)
    dataset = TensorDataset(X, y)
    n_train = int(len(dataset) * 0.8)
    n_val   = len(dataset) - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val],
                                      generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=0)

    # ---- 模型、损失、优化器 ----
    model     = build_fer_model(num_classes=7).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
    best_val_acc = 0.0

    for epoch in range(1, n_epochs + 1):
        # -- 训练 --
        model.train()
        train_correct, train_total, train_loss = 0, 0, 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss    += loss.item() * len(yb)
            train_correct += (logits.argmax(1) == yb).sum().item()
            train_total   += len(yb)
        scheduler.step()
        t_acc  = train_correct / train_total
        t_loss = train_loss    / train_total

        # -- 验证 --
        model.eval()
        val_correct, val_total, val_loss = 0, 0, 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits  = model(xb)
                loss    = criterion(logits, yb)
                val_loss    += loss.item() * len(yb)
                val_correct += (logits.argmax(1) == yb).sum().item()
                val_total   += len(yb)
        v_acc  = val_correct / val_total
        v_loss = val_loss    / val_total

        history["train_acc"].append(round(t_acc, 4))
        history["val_acc"].append(round(v_acc, 4))
        history["train_loss"].append(round(t_loss, 4))
        history["val_loss"].append(round(v_loss, 4))

        logger.info(f"  Epoch {epoch:3d}/{n_epochs} | "
                    f"train_acc={t_acc:.4f} loss={t_loss:.4f} | "
                    f"val_acc={v_acc:.4f} loss={v_loss:.4f}")

        # 保存最优模   
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            ckpt_path = CKPT_DIR / "fer_best.pt"
            torch.save(model.state_dict(), ckpt_path)

    # ---- 每类精度（在验证集上   ---
    model.eval()
    class_correct = {i: 0 for i in range(7)}
    class_total   = {i: 0 for i in range(7)}
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(yb.cpu().numpy().tolist())
            for true, pred in zip(yb, preds):
                class_total[true.item()] += 1
                if true == pred:
                    class_correct[true.item()] += 1

    per_class_acc = {
        EMOTION_NAMES[i]: round(class_correct[i] / max(class_total[i], 1), 4)
        for i in range(7)
    }

    metrics = {
        "best_val_acc":    round(best_val_acc, 4),
        "per_class_acc":   per_class_acc,
        "history":         history,
        "checkpoint":      str(CKPT_DIR / "fer_best.pt"),
        "emotion_nm_map":  EMOTION_NM_MAP,
    }
    return metrics, model


# ==============================================================================
# 4. 推理接口（供 Phase 1 调用   
# ==============================================================================

def load_fer_model(ckpt_path: Optional[Path] = None):
    """
    加载已训练的 FER 模型   

    Returns:
        model (FERNet)    None（如   checkpoint 不存在）
    """
    import torch
    path = ckpt_path or (CKPT_DIR / "fer_best.pt")
    if not path.exists():
        logger.warning(f"FER checkpoint 不存在：{path}，将返回未训练模型")
        return build_fer_model()
    model = build_fer_model()
    model.load_state_dict(torch.load(str(path), map_location="cpu"))
    model.eval()
    logger.info(f"FER 模型已加载：{path}")
    return model


def predict_emotion(model, face_crop: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    使用 FER 模型预测表情

    Args:
        model:     FERNet
        face_crop: BGR 图像 [H, W, 3] 或灰度 [H, W]

    Returns:
        (emotion_idx, prob_array)
    """
    import torch
    import torch.nn.functional as F
    import cv2

    if face_crop.ndim == 3:
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_crop
    resized = cv2.resize(gray, (48, 48)).astype(np.float32) / 255.0
    tensor  = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0)  # [1, 1, 48, 48]
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).squeeze().numpy()
    return int(probs.argmax()), probs


# ==============================================================================
# 5. 主实验流   
# ==============================================================================

def run_fer_pretrain_experiment(
    n_epochs: int = 20,
    max_samples: int = 10000,
    use_mock: bool = False,
) -> Dict:
    """
    运行 FER 预训练实验   

    Args:
        n_epochs:    训练轮数
        max_samples: 最大样本数
        use_mock:    强制使用 Mock 数据

    Returns:
        metrics
    """
    logger.info("=" * 60)
    logger.info("Phase 2 - FER 面部表情预训练实验")
    logger.info("=" * 60)

    csv_path = DATA_DIR / "fer2013.csv"
    dataset  = FERDataset(csv_path, usage="Training", max_samples=max_samples)

    if use_mock:
        logger.info("强制使用 Mock 数据")
        images, labels = dataset.generate_mock(n=min(max_samples, 3000))
        mock_mode = True
    elif dataset.is_available():
        logger.info(f"发现本地 CSV：{csv_path}")
        images, labels = dataset.load()
        mock_mode = False
    else:
        # 尝试从 HuggingFace 自动下载 (无需 Kaggle 认证)
        logger.info("本地 CSV 未找到，尝试从 HuggingFace 下载 FER2013")
        try:
            images, labels = dataset.download_from_huggingface(
                max_samples=max_samples, save_csv=True
            )
            mock_mode = False
            logger.info("HuggingFace 下载成功")
        except Exception as e:
            logger.warning(f"HuggingFace 下载失败 ({e}) → 回退到 Mock 数据")
            images, labels = dataset.generate_mock(n=min(max_samples, 3000))
            mock_mode = True

    logger.info(f"数据规模：{len(images)} 张，类分布：{dict(zip(*np.unique(labels, return_counts=True)))}")

    # ---- 训练 ----
    metrics, model = train_fer_model(images, labels, n_epochs=n_epochs)
    metrics["mock_mode"] = mock_mode
    metrics["n_samples"] = len(images)

    # ---- 保存结果 ----
    out_path = RESULTS_DIR / "phase2_fer_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        serializable = {k: v for k, v in metrics.items()
                        if isinstance(v, (int, float, str, dict, list))}
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    logger.info(f"结果已保存：{out_path}")

    # ---- 可视   ----
    _plot_phase2(metrics)

    return metrics


def _plot_phase2(metrics: Dict):
    """生成 Phase 2 可视化"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Phase 2: FER 表情预训练 - 训练曲线与每类精度", fontsize=14)

        # 子图1：训练曲线
        ax = axes[0]
        history = metrics.get("history", {})
        epochs  = list(range(1, len(history.get("train_acc", [])) + 1))
        if epochs:
            ax.plot(epochs, history["train_acc"], label="Train Acc", color="#2ECC71")
            ax.plot(epochs, history["val_acc"],   label="Val Acc",   color="#E74C3C")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy")
            ax.set_title("训练 / 验证精度曲线")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)

        # 子图2：每类精度柱状图
        ax2 = axes[1]
        per_cls = metrics.get("per_class_acc", {})
        if per_cls:
            names = list(per_cls.keys())
            vals  = [per_cls[n] for n in names]
            colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
            bars = ax2.bar(names, vals, color=colors)
            ax2.set_ylabel("Accuracy")
            ax2.set_title("每类情绪分类精度")
            ax2.set_ylim(0, 1)
            ax2.tick_params(axis="x", rotation=20)
            for bar, val in zip(bars, vals):
                ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                         f"{val:.3f}", ha="center", va="bottom", fontsize=8)

        # 神经调质映射标注
        fig.text(0.5, 0.01,
                 "情绪→神经调质映射：Happy→DA+5-HT | Fear→NE+5-HT | Sad→DA+5-HT | Neutral→基准",
                 ha="center", fontsize=9, color="dimgray")

        plt.tight_layout(rect=[0, 0.04, 1, 1])
        out_path = RESULTS_DIR / "phase2_fer_training.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"可视化已保存：{out_path}")
    except Exception as e:
        logger.warning(f"可视化生成失败：{e}")


# ==============================================================================
# 入口
# ==============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 2: FER 表情预训练")
    parser.add_argument("--epochs",      type=int,  default=20,    help="训练轮数")
    parser.add_argument("--max-samples", type=int,  default=10000, help="最大样本数")
    parser.add_argument("--mock",        action="store_true",      help="强制使用 Mock 数据")
    args = parser.parse_args()

    metrics = run_fer_pretrain_experiment(
        n_epochs=args.epochs,
        max_samples=args.max_samples,
        use_mock=args.mock,
    )

    print("\n========== Phase 2 实验结果摘要 ==========")
    print(f"最优验证精度：{metrics['best_val_acc']:.4f}")
    print("每类精度:")
    for emo, acc in metrics.get("per_class_acc", {}).items():
        print(f"  {emo:10s}: {acc:.4f}")
    print(f"\n情绪→神经调质映射示例：")
    for emo, deltas in list(EMOTION_NM_MAP.items())[:3]:
        nm = emotion_to_neuromodulator(EMOTION_NAMES.index(emo))
        print(f"  {emo:10s}: {nm}")
    print(f"\nCheckpoint：{metrics.get('checkpoint')}")
