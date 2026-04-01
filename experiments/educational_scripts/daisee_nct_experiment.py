"""
Phase 1: DAiSEE 视觉验证实验
==============================
使用 DAiSEE 数据集验证 NCT 视觉编码器对学生情感状态的识别能力。

DAiSEE 数据集：
  - 9,068 段视频，112 名学生
  - 4 类状态：Boredom / Confusion / Engagement / Frustration
  - 每类 4 级强度：0=very low, 1=low, 2=high, 3=very high

实验流程：
  视频帧 → 视觉特征 → StudentState → NeuromodulatorState → NCT → Φ值

下载地址：https://people.iith.ac.in/vineethnb/resources/daisee/index.html
数据放置：data/daisee/DataSet/ 和 data/daisee/Labels/

作者：NCT LAB Team
日期：2026-03-15
"""

import sys
import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
from scipy.stats import pearsonr

# ---- 项目路径 ----
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nct_modules.nct_manager import NCTManager
from nct_modules.nct_core import NCTConfig
from experiments.education_state_detection import (
    StudentState,
    NeuromodulatorState,
    StudentStateRecognizer,
    NeuromodulatorMapper,
    MultiModalSensor,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---- 路径常规 ----
DATA_DIR    = PROJECT_ROOT / "data" / "daisee"
RESULTS_DIR = PROJECT_ROOT / "results" / "education"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---- DAiSEE 标签列名 ----
LABEL_COLS = ["Boredom", "Confusion", "Engagement", "Frustration"]
# CSV 中可能有尾随空格，需要 strip
LABEL_COLS_ALT = {"Boredom": "Boredom", "Confusion": "Confusion", "Engagement": "Engagement", "Frustration": "Frustration ", "Frustration ": "Frustration"}

# ---- 小尺寸 NCT 配置（加速实验）----
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
# 1. 数据加载器
# ==============================================================================

class DAiSEELoader:
    """DAiSEE 数据集加载器（支持真实数据与 Mock 模式）"""

    def __init__(self, data_dir: Path, split: str = "Test", max_clips: int = 200):
        self.data_dir   = data_dir
        self.split      = split
        self.max_clips  = max_clips
        # 真实 DAiSEE 路径结构：data/daisee/DAiSEE/DAiSEE/DataSet/{split}/
        self.video_dir  = data_dir / "DAiSEE" / "DAiSEE" / "DataSet" / split
        self.label_file = data_dir / "DAiSEE" / "DAiSEE" / "Labels" / f"{split}Labels.csv"

    def is_available(self) -> bool:
        # 检查标签文件和视频目录是否存在
        return self.label_file.exists() or self.video_dir.exists()

    def load_labels(self) -> List[Dict]:
        """加载 CSV 标签，匹配视频路径，返回 {video_path, labels} 列表"""
        samples = []
        
        # 尝试真实 DAiSEE 结构
        if self.label_file.exists():
            import csv
            with open(self.label_file, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                # strip 列名中的空格
                for row in reader:
                    # 清理列名
                    clean_row = {k.strip(): v for k, v in row.items()}
                    clip_id = clean_row.get("ClipID", "").replace(".avi", "")
                    # 视频路径：DataSet/{split}/{subject_id}/{clip_id}/{clip_id}.avi
                    # ClipID 格式：100011002 -> subject=110001, clip=1100011002
                    if len(clip_id) >= 7:
                        subject_id = clip_id[:6]
                        video_path = self.video_dir / subject_id / clip_id / f"{clip_id}.avi"
                    else:
                        video_path = self.video_dir / f"{clip_id}.avi"
                    
                    samples.append({
                        "video_path": str(video_path),
                        "labels": {
                            col: int(clean_row[col]) for col in LABEL_COLS if col in clean_row
                        },
                    })
                    if len(samples) >= self.max_clips:
                        break
            logger.info(f"加载 {len(samples)} 个样本自 {self.label_file}")
            return samples
        
        # 回退：扫描视频目录
        if self.video_dir.exists():
            for avi_file in sorted(self.video_dir.rglob("*.avi"))[:self.max_clips]:
                samples.append({
                    "video_path": str(avi_file),
                    "labels": {col: 0 for col in LABEL_COLS},  # 无标签
                })
            logger.info(f"扫描到 {len(samples)} 个视频文件")
            return samples
        
        return []

    def extract_frames(self, video_path: str, n_frames: int = 5) -> List[np.ndarray]:
        """均匀抽取 n_frames 帧"""
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        indices = np.linspace(0, total - 1, n_frames, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        cap.release()
        return frames if frames else [np.zeros((480, 640, 3), dtype=np.uint8)]

    def generate_mock_samples(self, n: int = 100) -> List[Dict]:
        """生成 Mock 样本（无真实数据时使用）"""
        np.random.seed(42)
        samples = []
        for i in range(n):
            # 随机生成各维度 0-3 标签
            labels = {col: np.random.randint(0, 4) for col in LABEL_COLS}
            samples.append({"video_path": None, "labels": labels, "mock": True})
        return samples


# ==============================================================================
# 2. 视觉特征提取（不依赖 dlib）
# ==============================================================================

def extract_visual_features_opencv(frame: np.ndarray) -> Dict:
    """
    使用 OpenCV Haar 级联提取轻量视觉特征。

    返回字段：
      - face_detected: bool
      - eye_aspect_ratio: float  (EAR 近似值)
      - head_pose_yaw: float     (归一化偏航角估计)
      - face_brightness: float   (归一化面部亮度)
      - edge_density: float      (表示细节/运动)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    features = {
        "face_detected":    False,
        "eye_aspect_ratio": 0.3,     # 正常值 ~0.3，疲劳时 <0.25
        "head_pose_yaw":    0.0,
        "face_brightness":  0.5,
        "edge_density":     0.0,
    }

    # ---- 人脸检测 ----
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    eye_cascade  = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_roi = gray[y:y + h, x:x + w]
        features["face_detected"]   = True
        features["face_brightness"] = float(face_roi.mean()) / 255.0

        # ---- EAR 近似（基于眼睛检测）----
        eyes = eye_cascade.detectMultiScale(face_roi)
        if len(eyes) >= 2:
            # 两眼高宽比均值
            ear = np.mean([ey[3] / (ey[2] + 1e-6) for ey in eyes[:2]])
            features["eye_aspect_ratio"] = float(np.clip(ear, 0.1, 0.6))
        elif len(eyes) == 1:
            # 单眼可见 → 侧脸
            features["head_pose_yaw"] = 0.3

        # ---- 边缘密度（局部运动/表情细节）----
        edges = cv2.Canny(face_roi, 50, 150)
        features["edge_density"] = float(edges.mean()) / 255.0

        # ---- 头部偏航近似（人脸中心相对图像中心）----
        face_cx = x + w / 2
        img_cx  = frame.shape[1] / 2
        features["head_pose_yaw"] = float((face_cx - img_cx) / (img_cx + 1e-6))

    return features


def visual_features_to_student_state(
    vis: Dict, labels: Optional[Dict] = None
) -> StudentState:
    """
    将视觉特征映射为 StudentState。
    如果提供真实 labels，辅助校准；否则纯规则推断。
    """
    # -- 疲劳度: EAR 低 → 疲劳
    ear = vis.get("eye_aspect_ratio", 0.3)
    fatigue = float(np.clip((0.3 - ear) / 0.15, 0, 1))

    # -- 专注度: 亮度适中 + 低边缘密度 + 低偏航
    brightness = vis.get("face_brightness", 0.5)
    yaw        = abs(vis.get("head_pose_yaw", 0.0))
    focus = float(np.clip(brightness * (1 - yaw) * (1 - fatigue), 0, 1))

    # -- 困惑度: 高边缘密度（皱眉）+ 低 EAR 变化
    edge_density = vis.get("edge_density", 0.0)
    confusion = float(np.clip(edge_density * 1.5, 0, 1))

    # -- 参与度: 面部检测 + 专注
    face_ok   = float(vis.get("face_detected", False))
    engagement = float(np.clip(face_ok * focus, 0, 1))

    # -- 压力: 困惑 + 疲劳组合
    stress = float(np.clip(0.5 * confusion + 0.5 * fatigue, 0, 1))
    confidence = float(np.clip(1 - stress, 0, 1))

    return StudentState(
        timestamp=time.time(),
        focus_level=focus,
        engagement=engagement,
        confusion=confusion,
        fatigue=fatigue,
        stress_level=stress,
        confidence=confidence,
    )


def build_nct_visual_input(frame: np.ndarray, target_size: int = 28) -> np.ndarray:
    """将帧缩放为 NCT 视觉输入 [H, W]（灰度 28×28）"""
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (target_size, target_size))
    return resized.astype(np.float32) / 255.0


# ==============================================================================
# 2.5  FER 模型集成（Phase 2 → Phase 1 桥接）
# ==============================================================================

class FERVisualExtractor:
    """
    使用 Phase 2 训练的 FER 模型从面部区域预测情绪，
    并将情绪映射为 StudentState，替代纯规则方案。
    """
    # 情绪 → StudentState 映射（基于情绪/教育状态对应关系）
    EMOTION_TO_STATE = {
        "Happy":    dict(focus_level=0.7, engagement=0.8, confusion=0.1, fatigue=0.1, stress_level=0.1, confidence=0.85),
        "Surprise": dict(focus_level=0.75, engagement=0.9, confusion=0.35, fatigue=0.1, stress_level=0.25, confidence=0.70),
        "Fear":     dict(focus_level=0.50, engagement=0.6, confusion=0.55, fatigue=0.3, stress_level=0.75, confidence=0.35),
        "Angry":    dict(focus_level=0.45, engagement=0.5, confusion=0.40, fatigue=0.4, stress_level=0.70, confidence=0.40),
        "Sad":      dict(focus_level=0.30, engagement=0.3, confusion=0.30, fatigue=0.6, stress_level=0.55, confidence=0.40),
        "Disgust":  dict(focus_level=0.35, engagement=0.4, confusion=0.50, fatigue=0.3, stress_level=0.60, confidence=0.35),
        "Neutral":  dict(focus_level=0.55, engagement=0.55, confusion=0.20, fatigue=0.2, stress_level=0.20, confidence=0.65),
    }
    EMOTION_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    CKPT_PATH = PROJECT_ROOT / "checkpoints" / "fer_pretrain" / "fer_best.pt"

    def __init__(self):
        self._model = None
        self._device = None
        self._available = False
        self._try_load()

    def _try_load(self):
        """尝试加载 FER 模型（如果 checkpoint 存在）"""
        if not self.CKPT_PATH.exists():
            logger.info("FER checkpoint 未找到，将使用规则方案")
            return
        try:
            import torch
            import torch.nn as nn
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            class _FERNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.features = nn.Sequential(
                        nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                        nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                        nn.MaxPool2d(2), nn.Dropout2d(0.2),
                        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                        nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                        nn.MaxPool2d(2), nn.Dropout2d(0.2),
                        nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                        nn.MaxPool2d(2), nn.Dropout2d(0.3),
                    )
                    self.classifier = nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(128 * 6 * 6, 256), nn.ReLU(inplace=True),
                        nn.Dropout(0.4), nn.Linear(256, 7),
                    )
                def forward(self, x):
                    return self.classifier(self.features(x))

            model = _FERNet().to(self._device)
            model.load_state_dict(torch.load(str(self.CKPT_PATH),
                                             map_location=self._device))
            model.eval()
            self._model  = model
            self._available = True
            logger.info(f"FER 模型已加载：{self.CKPT_PATH}")
        except Exception as e:
            logger.warning(f"FER 模型加载失败（{e}），使用规则方案")

    @property
    def available(self) -> bool:
        return self._available

    def predict_from_frame(self, frame: np.ndarray) -> Tuple[str, float]:
        """
        从视频帧预测主要情绪。

        Args:
            frame: BGR 帧 [H, W, 3]

        Returns:
            (emotion_name, confidence)
        """
        if not self._available:
            return "Neutral", 0.5
        import torch
        try:
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 尝试检测人脸区域
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            if len(faces) > 0:
                x, y, w, h = faces[0]
                roi = gray[y:y + h, x:x + w]
            else:
                roi = gray  # 无人脸时用全图
            roi_resized = cv2.resize(roi, (48, 48)).astype(np.float32) / 255.0
            inp = torch.from_numpy(roi_resized[np.newaxis, np.newaxis]).to(self._device)
            with torch.no_grad():
                logits = self._model(inp)[0]
                probs  = torch.softmax(logits, dim=0).cpu().numpy()
            idx   = int(np.argmax(probs))
            conf  = float(probs[idx])
            return self.EMOTION_NAMES[idx], conf
        except Exception as e:
            logger.debug(f"FER 预测异常：{e}")
            return "Neutral", 0.5

    def emotion_to_student_state(self, emotion: str) -> StudentState:
        """将情绪名称转换为 StudentState"""
        params = self.EMOTION_TO_STATE.get(emotion, self.EMOTION_TO_STATE["Neutral"])
        return StudentState(timestamp=time.time(), **params)


# ==============================================================================
# 3. 核心实验逻辑
# ==============================================================================

def run_daisee_experiment(
    max_clips: int = 200,
    frames_per_clip: int = 3,
    use_mock: bool = False,
    use_fer: bool = False,
    split: str = "Test",
) -> Dict:
    """
    运行 DAiSEE NCT 实验。

    Args:
        max_clips:       处理的最大片段数
        frames_per_clip: 每段视频抽帧数
        use_mock:        强制使用 Mock 数据（不论真实数据是否存在）
        use_fer:         启用 FER 模型增强视觉特征提取（Phase 2 集成）

    Returns:
        results: 包含 phi 值统计、分类准确率、相关系数的字典
    """
    logger.info("=" * 60)
    logger.info("Phase 1 - DAiSEE NCT 视觉验证实验")
    logger.info("=" * 60)

    # ---- 初始化 NCT ----
    nct = NCTManager(LIGHT_NCT_CONFIG)
    nct.start()
    nm_mapper = NeuromodulatorMapper()

    # ---- FER 模型初始化（Phase 2 集成）----
    fer_extractor = None
    if use_fer:
        fer_extractor = FERVisualExtractor()
        if fer_extractor.available:
            logger.info("[FER 集成] 已启用 Phase 2 FER 模型进行情绪增强")
        else:
            logger.info("[FER 集成] FER checkpoint 不存在，降级为规则方案")

    # ---- 加载数据 ----
    loader = DAiSEELoader(DATA_DIR, split=split, max_clips=max_clips)
    if use_mock or not loader.is_available():
        logger.info("使用 Mock 数据（真实 DAiSEE 数据未检测到）")
        samples = loader.generate_mock_samples(max_clips)
        mock_mode = True
    else:
        logger.info(f"加载真实 DAiSEE 数据：{loader.video_dir}")
        samples = loader.load_labels()
        mock_mode = False

    logger.info(f"共 {len(samples)} 条样本")

    # ---- 按标签类别聚合 Phi 值 ----
    category_phi: Dict[str, List[Tuple[int, float]]] = {col: [] for col in LABEL_COLS}
    nm_values: Dict[str, List[float]] = {"DA": [], "5-HT": [], "NE": [], "ACh": []}
    engagement_labels: List[int]   = []
    engagement_da:     List[float] = []
    fer_emotion_counts: Dict[str, int] = {}

    for idx, sample in enumerate(samples):
        labels = sample["labels"]

        # ---- 帧处理 ----
        if mock_mode or sample["video_path"] is None:
            eng_norm = labels.get("Engagement", 1) / 3.0
            brightness = int(50 + eng_norm * 150)
            mock_frame = np.full((480, 640, 3), brightness, dtype=np.uint8)
            frames = [mock_frame]
        else:
            frames = loader.extract_frames(sample["video_path"], frames_per_clip)

        # ---- 逐帧提取特征 ----
        frame_phi_vals = []
        for frame in frames:
            # 选择特征提取方案（FER 模型 OR 规则）
            if fer_extractor is not None and fer_extractor.available and not mock_mode:
                emotion, conf = fer_extractor.predict_from_frame(frame)
                student_state = fer_extractor.emotion_to_student_state(emotion)
                fer_emotion_counts[emotion] = fer_emotion_counts.get(emotion, 0) + 1
            else:
                vis_feats = extract_visual_features_opencv(frame)
                student_state = visual_features_to_student_state(vis_feats, labels)
            nm_state = nm_mapper.map_to_neuromodulators(student_state)

            # 构建 NCT 输入
            visual_input = build_nct_visual_input(frame)
            sensory_data = {"visual": visual_input}
            neurotrans = nm_state.to_dict()

            # 运行 NCT 周期
            try:
                nct_state = nct.process_cycle(sensory_data, neurotrans)
                phi = nct_state.consciousness_metrics.get("phi_value", 0.0)
            except Exception as e:
                logger.debug(f"NCT process_cycle 异常：{e}")
                phi = 0.0

            frame_phi_vals.append(phi)

            for k in ["DA", "5-HT", "NE", "ACh"]:
                nm_values[k].append(neurotrans.get(k, 0.5))

        clip_phi = float(np.mean(frame_phi_vals))

        for col in LABEL_COLS:
            if col in labels:
                category_phi[col].append((labels[col], clip_phi))

        if "Engagement" in labels:
            engagement_labels.append(labels["Engagement"])
            da_mean = float(np.mean(nm_values["DA"][-len(frames):]))
            engagement_da.append(da_mean)

        if (idx + 1) % 20 == 0:
            logger.info(f"  已处理 {idx + 1}/{len(samples)} 条，平均 Φ={clip_phi:.4f}")

    # ---- 统计分析 ----
    results = {
        "experiment":   "Phase1_DAiSEE",
        "n_samples":    len(samples),
        "mock_mode":    mock_mode,
        "fer_enhanced": fer_extractor is not None and fer_extractor.available,
        "fer_emotion_counts": fer_emotion_counts if fer_emotion_counts else None,
        "phi_by_label": {},
        "correlations": {},
        "nm_means":     {},
    }

    # Phi 均值按标签强度分组
    for col in LABEL_COLS:
        phi_by_level = {0: [], 1: [], 2: [], 3: []}
        for lv, phi in category_phi[col]:
            phi_by_level[lv].append(phi)
        results["phi_by_label"][col] = {
            str(lv): {
                "mean": float(np.mean(vals)) if vals else 0.0,
                "std":  float(np.std(vals))  if vals else 0.0,
                "n":    len(vals),
            }
            for lv, vals in phi_by_level.items()
        }

    # Engagement 与 DA 相关系数
    if len(engagement_labels) > 5:
        r, p = pearsonr(engagement_labels, engagement_da)
        results["correlations"]["engagement_vs_DA"] = {"r": round(r, 4), "p": round(p, 4)}
        logger.info(f"Engagement↔DA 相关系数 r={r:.4f}, p={p:.4f}")

    # 神经调质均值
    for k, vals in nm_values.items():
        results["nm_means"][k] = round(float(np.mean(vals)), 4) if vals else 0.0

    # ---- Phi vs Engagement 趋势 ----
    eng_phi = [phi for lv, phi in category_phi["Engagement"] if lv in (0, 1, 2, 3)]
    eng_lv  = [lv  for lv, phi in category_phi["Engagement"]]
    if eng_phi:
        r_ep, p_ep = pearsonr(eng_lv, eng_phi)
        results["correlations"]["engagement_level_vs_phi"] = {
            "r": round(r_ep, 4), "p": round(p_ep, 4)
        }
        logger.info(f"Engagement 强度↔Φ 相关系数 r={r_ep:.4f}, p={p_ep:.4f}")

    # ---- 保存结果 ----
    out_path = RESULTS_DIR / "phase1_daisee_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"结果已保存：{out_path}")

    # ---- 生成可视化 ----
    _plot_phase1(results)

    return results


def _plot_phase1(results: Dict):
    """生成 Phase 1 可视化图表"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Phase 1: DAiSEE × NCT - Phi 值与情感状态关系", fontsize=14)

        # 子图1：各情感类别在不同强度下的 Phi 均值
        ax = axes[0]
        colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12"]
        for i, col in enumerate(LABEL_COLS):
            phi_data = results["phi_by_label"].get(col, {})
            levels = [0, 1, 2, 3]
            means  = [phi_data.get(str(lv), {}).get("mean", 0) for lv in levels]
            ax.plot(levels, means, marker="o", label=col, color=colors[i], linewidth=2)
        ax.set_xlabel("情感强度 (0=very low, 3=very high)")
        ax.set_ylabel("平均 Φ 值")
        ax.set_title("各情感状态强度 vs Φ 值")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 子图2：神经调质均值柱状图
        ax2 = axes[1]
        nm = results.get("nm_means", {})
        keys = list(nm.keys())
        vals = [nm[k] for k in keys]
        bar_colors = ["#9B59B6", "#1ABC9C", "#E67E22", "#2980B9"]
        bars = ax2.bar(keys, vals, color=bar_colors[:len(keys)], width=0.5)
        ax2.axhline(y=0.5, color="gray", linestyle="--", alpha=0.7, label="基准值 0.5")
        ax2.set_ylabel("神经调质浓度")
        ax2.set_title("神经调质参数均值")
        ax2.set_ylim(0, 1)
        ax2.legend()
        for bar, val in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=9)

        # 添加相关系数注释
        corr = results.get("correlations", {})
        if "engagement_level_vs_phi" in corr:
            r = corr["engagement_level_vs_phi"]["r"]
            p = corr["engagement_level_vs_phi"]["p"]
            fig.text(0.5, 0.01,
                     f"Engagement × Φ 相关系数: r={r:.4f}, p={p:.4f}",
                     ha="center", fontsize=10, color="darkblue")

        plt.tight_layout(rect=[0, 0.04, 1, 1])
        out_path = RESULTS_DIR / "phase1_daisee_phi_analysis.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"可视化已保存：{out_path}")
    except Exception as e:
        logger.warning(f"可视化生成失败（不影响实验结果）：{e}")


# ==============================================================================
# 入口
# ==============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 1: DAiSEE × NCT 实验")
    parser.add_argument("--split",      type=str,  default="Test",
                        help="数据集分割（Train/Validation/Test，默认 Test）")
    parser.add_argument("--max-clips",  type=int,  default=200,
                        help="最大处理片段数（默认 200）")
    parser.add_argument("--frames",     type=int,  default=3,
                        help="每段抽帧数（默认 3）")
    parser.add_argument("--mock",       action="store_true",
                        help="强制使用 Mock 数据")
    parser.add_argument("--use-fer",    action="store_true",
                        help="启用 Phase 2 FER 模型增强视觉特征")
    args = parser.parse_args()

    results = run_daisee_experiment(
        max_clips=args.max_clips,
        frames_per_clip=args.frames,
        use_mock=args.mock,
        use_fer=args.use_fer,
        split=args.split,
    )

    print("\n========== Phase 1 实验结果摘要 ==========")
    print(f"样本数：{results['n_samples']}")
    print(f"Mock 模式：{results['mock_mode']}")
    print("\n神经调质均值：")
    for k, v in results["nm_means"].items():
        print(f"  {k}: {v:.4f}")
    print("\n相关系数：")
    for k, v in results["correlations"].items():
        print(f"  {k}: r={v['r']:.4f}, p={v['p']:.4f}")
    print(f"\n结果文件：{RESULTS_DIR / 'phase1_daisee_results.json'}")
