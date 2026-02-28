"""
Phase 2: Anomaly Detection with NCT Framework
==============================================
Anomaly detection using prediction error, Φ value, and salience mismatch.

Three complementary approaches:
1. Prediction Error-based (Free Energy minimization violation)
2. Φ-value based (Information integration drop)
3. Salience-based (Feature importance mismatch)

Author: WENG YONGGANG
Affiliation: NeuroConscious Lab, Universiti Teknologi Malaysia
Date: February 24, 2026
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Import NCT modules
import sys
sys.path.append('..')
from nct_modules.nct_manager import NCTManager
from nct_modules.nct_modules import NCTConfig

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

@dataclass
class AnomalyDetectionConfig:
    """Configuration for anomaly detection experiments"""
    # Model config
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 3
    d_ff: int = 768
    dropout_rate: float = 0.4
    
    # Training config
    batch_size: int = 128
    n_epochs: int = 50
    learning_rate: float = 0.001
    
    # Dataset config
    n_normal_samples: int = 500  # per class
    n_anomaly_samples: int = 100  # total anomalies
    
    # Anomaly types
    anomaly_types: list = None
    
    # Thresholds
    pe_threshold_percentile: float = 95.0  # Top 5% as anomaly
    phi_threshold_percentile: float = 5.0   # Bottom 5% as anomaly
    salience_threshold_percentile: float = 95.0
    
    # Output
    version: str = "v1"
    results_dir: str = None
    
    def __post_init__(self):
        if self.anomaly_types is None:
            self.anomaly_types = ['noise', 'rotation', 'occlusion', 'out_of_distribution']
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.results_dir is None:
            self.results_dir = Path(f'results/anomaly_detection_{timestamp}')
        else:
            self.results_dir = Path(self.results_dir)
        
        self.results_dir.mkdir(parents=True, exist_ok=True)


class AnomalyGenerator:
    """Generate different types of anomalies"""
    
    def __init__(self, img_size=28):
        self.img_size = img_size
    
    def add_noise(self, x, noise_level=0.5):
        """Add Gaussian noise"""
        noise = torch.randn_like(x) * noise_level
        return torch.clamp(x + noise, 0, 1)
    
    def rotate(self, x, angle=90):
        """Rotate image by fixed angle"""
        # Simple 90-degree rotation
        if angle == 90:
            return x.flip(2).transpose(2, 3)
        elif angle == 180:
            return x.flip(2).flip(3)
        elif angle == 270:
            return x.transpose(2, 3).flip(2)
        else:
            raise ValueError(f"Unsupported angle: {angle}")
    
    def occlude(self, x, patch_size=7):
        """Add black square occlusion"""
        batch_size = x.shape[0]
        occluded = x.clone()
        
        # Random position
        max_h = self.img_size - patch_size
        max_w = self.img_size - patch_size
        
        h_start = np.random.randint(0, max_h)
        w_start = np.random.randint(0, max_w)
        
        occluded[:, :, h_start:h_start+patch_size, w_start:w_start+patch_size] = 0
        
        return occluded
    
    def create_out_of_distribution(self, n_samples):
        """Create OOD samples (e.g., random patterns)"""
        # Pure noise images
        ood = torch.rand(n_samples, 1, self.img_size, self.img_size)
        return ood


class NCTAnomalyDetector:
    """
    Anomaly detection using NCT framework.
    
    Three complementary signals:
    1. Prediction Error (PE): High PE → anomaly
    2. Φ Value: Low Φ → poor integration → anomaly
    3. Salience Mismatch: Unexpected feature importance → anomaly
    """
    
    def __init__(self, config: AnomalyDetectionConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize NCT manager
        nct_config = NCTConfig(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            dim_ff=config.d_ff,
            dropout=config.dropout_rate
        )
        
        self.nct_manager = NCTManager(nct_config)
        self.nct_manager.to(self.device)
        
        # Anomaly generator
        self.anomaly_gen = AnomalyGenerator(img_size=28)
        
        # Statistics from training (for threshold setting)
        self.pe_stats = {'mean': 0.0, 'std': 1.0}
        self.phi_stats = {'mean': 0.0, 'std': 1.0}
        self.thresholds = {}
    
    def train_on_normal_data(self, train_loader, val_loader=None):
        """Train NCT on normal (non-anomalous) data"""
        print("=" * 60)
        print("Training NCT on Normal Data")
        print("=" * 60)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.nct_manager.parameters(), 
                               lr=self.config.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.n_epochs)
        
        best_val_acc = 0.0
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'free_energy': [],
            'phi': []
        }
        
        for epoch in range(self.config.n_epochs):
            # Training
            self.nct_manager.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            all_prediction_errors = []
            all_phi_values = []
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.to(self.device)
                target = target.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass through NCT using process_cycle
                # Process each sample in batch individually
                outputs = []
                prediction_errors = []
                phi_values = []
                
                for i in range(data.size(0)):
                    sample = data[i:i+1]  # Keep batch dimension
                    
                    # Convert to numpy for process_cycle
                    sample_np = sample.cpu().numpy()
                    
                    # Start NCT cycle
                    self.nct_manager.start()
                    state = self.nct_manager.process_cycle(
                        sensory_data={'visual': sample_np[0, 0]}  # Remove channel dim
                    )
                    
                    # Get output from state
                    if hasattr(state, 'workspace_content') and state.workspace_content is not None:
                        # Use workspace content as output
                        if hasattr(state.workspace_content, 'content'):
                            output = state.workspace_content.content  # Tensor
                        else:
                            # Fallback to dict representation
                            output = state.workspace_content.to_dict() if hasattr(state.workspace_content, 'to_dict') else torch.zeros(1, 10)
                    elif hasattr(state, 'consciousness_metrics'):
                        # Use metrics as proxy
                        metrics = state.consciousness_metrics
                        output = torch.tensor([metrics.get('phi', 0.0)] * 10).unsqueeze(0)
                    else:
                        output = torch.zeros(1, 10)
                    
                    outputs.append(output)
                    
                    # Get prediction error
                    if hasattr(state, 'self_representation') and 'prediction_error' in state.self_representation:
                        pe_val = state.self_representation['prediction_error']
                        if isinstance(pe_val, (int, float)):
                            prediction_errors.append(torch.tensor(pe_val))
                        else:
                            prediction_errors.append(pe_val)
                    
                    # Get Φ value
                    if hasattr(state, 'consciousness_metrics') and 'phi' in state.consciousness_metrics:
                        phi_val = state.consciousness_metrics['phi']
                        if isinstance(phi_val, (int, float)):
                            phi_values.append(torch.tensor(phi_val))
                        else:
                            phi_values.append(phi_val)
                
                # Stack outputs
                output = torch.cat(outputs, dim=0)
                
                # Compute loss
                loss = criterion(output, target)
                
                # Aggregate prediction errors
                if len(prediction_errors) > 0:
                    pe = torch.stack(prediction_errors).mean().item()
                    all_prediction_errors.append(pe)
                
                # Aggregate Φ values
                if len(phi_values) > 0:
                    phi = torch.stack(phi_values).mean().item()
                    all_phi_values.append(phi)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * data.size(0)
                _, predicted = output.max(1)
                train_total += data.size(0)
                train_correct += predicted.eq(target).sum().item()
            
            # Update statistics
            self.pe_stats['mean'] = np.mean(all_prediction_errors)
            self.pe_stats['std'] = np.std(all_prediction_errors)
            
            self.phi_stats['mean'] = np.mean(all_phi_values)
            self.phi_stats['std'] = np.std(all_phi_values)
            
            train_loss /= len(train_loader.dataset)
            train_acc = train_correct / train_total
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['free_energy'].append(np.mean(all_prediction_errors))
            history['phi'].append(np.mean(all_phi_values))
            
            # Validation
            val_acc = 0.0
            if val_loader is not None:
                val_loss, val_acc = self.evaluate(val_loader)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    # Save best model
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.nct_manager.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_acc': val_acc,
                        'pe_stats': self.pe_stats,
                        'phi_stats': self.phi_stats
                    }, self.config.results_dir / 'best_model.pt')
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{self.config.n_epochs}:")
                print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                if val_loader is not None:
                    print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                print(f"  Avg PE: {history['free_energy'][-1]:.4f}, Avg Φ: {history['phi'][-1]:.4f}")
            
            scheduler.step()
        
        # Set thresholds based on training distribution
        self._set_thresholds(all_prediction_errors, all_phi_values)
        
        # Save training history
        with open(self.config.results_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"\nTraining completed!")
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        print(f"PE stats: μ={self.pe_stats['mean']:.4f}, σ={self.pe_stats['std']:.4f}")
        print(f"Φ stats: μ={self.phi_stats['mean']:.4f}, σ={self.phi_stats['std']:.4f}")
        
        return history
    
    def _set_thresholds(self, pes, phis):
        """Set anomaly detection thresholds"""
        # Prediction error threshold (top 5%)
        self.thresholds['pe'] = np.percentile(pes, self.config.pe_threshold_percentile)
        
        # Φ value threshold (bottom 5%)
        self.thresholds['phi'] = np.percentile(phis, self.config.phi_threshold_percentile)
        
        print(f"\nThresholds set:")
        print(f"  PE threshold: {self.thresholds['pe']:.4f} (>{self.thresholds['pe']:.4f} → anomaly)")
        print(f"  Φ threshold: {self.thresholds['phi']:.4f} (<{self.thresholds['phi']:.4f} → anomaly)")
    
    def evaluate(self, data_loader):
        """Evaluate model on dataset"""
        self.nct_manager.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output_dict = self.nct_manager.forward(data)
                output = output_dict['output']
                
                loss = criterion(output, target)
                total_loss += loss.item() * data.size(0)
                
                _, predicted = output.max(1)
                total += data.size(0)
                correct += predicted.eq(target).sum().item()
        
        total_loss /= len(data_loader.dataset)
        accuracy = correct / total
        
        return total_loss, accuracy
    
    def detect_anomalies(self, test_loader, anomaly_type='normal'):
        """
        Detect anomalies using multiple signals.
        
        Returns:
            dict: Detection results including TP, FP, TN, FN
        """
        self.nct_manager.eval()
        
        all_predictions = []
        all_labels = []  # 1=anomaly, 0=normal
        all_pes = []
        all_phis = []
        all_confidences = []
        
        with torch.no_grad():
            for data, labels in test_loader:
                data = data.to(self.device)
                
                # Forward pass
                output_dict = self.nct_manager.forward(data)
                output = output_dict['output']
                
                # Get prediction error
                pe = output_dict.get('prediction_error', torch.tensor(0.0))
                if isinstance(pe, torch.Tensor):
                    pe = pe.pow(2).mean(dim=(1, 2, 3)).cpu().numpy()
                else:
                    pe = np.array([pe] * data.size(0))
                
                # Get Φ value
                phi = output_dict.get('phi', torch.tensor(0.0))
                if isinstance(phi, torch.Tensor):
                    phi = phi.cpu().numpy()
                else:
                    phi = np.array([phi] * data.size(0))
                
                # Get confidence (softmax probability)
                probs = torch.softmax(output, dim=1)
                confidence, _ = probs.max(dim=1)
                confidence = confidence.cpu().numpy()
                
                # Store
                all_pes.extend(pe)
                all_phis.extend(phi)
                all_confidences.extend(confidence)
                all_labels.extend(labels.numpy())
                
                # Predictions based on each signal
                pred_pe = (pe > self.thresholds['pe']).astype(int)
                pred_phi = (phi < self.thresholds['phi']).astype(int)
                pred_combined = ((pred_pe + pred_phi) >= 1).astype(int)
                
                all_predictions.append({
                    'pe': pred_pe,
                    'phi': pred_phi,
                    'combined': pred_combined
                })
        
        # Aggregate results
        results = self._compute_metrics(all_predictions, all_labels, all_pes, all_phis, all_confidences)
        results['anomaly_type'] = anomaly_type
        
        return results
    
    def _compute_metrics(self, predictions, true_labels, pes, phis, confidences):
        """Compute detection metrics"""
        true_labels = np.array(true_labels)
        
        metrics = {}
        
        for method in ['pe', 'phi', 'combined']:
            preds = np.concatenate([p[method] for p in predictions])
            
            # True positives, false positives, etc.
            tp = np.sum((preds == 1) & (true_labels == 1))
            fp = np.sum((preds == 1) & (true_labels == 0))
            tn = np.sum((preds == 0) & (true_labels == 0))
            fn = np.sum((preds == 0) & (true_labels == 1))
            
            # Rates
            precision = tp / (tp + fp + 1e-9)
            recall = tp / (tp + fn + 1e-9)  # Sensitivity
            specificity = tn / (tn + fp + 1e-9)
            f1 = 2 * precision * recall / (precision + recall + 1e-9)
            
            metrics[f'{method}_precision'] = precision
            metrics[f'{method}_recall'] = recall
            metrics[f'{method}_specificity'] = specificity
            metrics[f'{method}_f1'] = f1
            metrics[f'{method}_tp'] = int(tp)
            metrics[f'{method}_fp'] = int(fp)
            metrics[f'{method}_tn'] = int(tn)
            metrics[f'{method}_fn'] = int(fn)
        
        # Add statistics
        metrics['pe_mean'] = np.mean(pes)
        metrics['pe_std'] = np.std(pes)
        metrics['phi_mean'] = np.mean(phis)
        metrics['phi_std'] = np.std(phis)
        metrics['confidence_mean'] = np.mean(confidences)
        metrics['confidence_std'] = np.std(confidences)
        
        return metrics
    
    def generate_anomaly_dataset(self, normal_data, anomaly_types, n_anomalies_per_type=25):
        """
        Generate synthetic anomaly dataset.
        
        Args:
            normal_data: Tuple (images, labels)
            anomaly_types: List of anomaly types to generate
            n_anomalies_per_type: Number of anomalies per type
        
        Returns:
            DataLoader with mixed normal and anomaly samples
        """
        normal_images, normal_labels = normal_data
        
        all_images = []
        all_labels = []  # 1 = anomaly, 0 = normal
        
        # Add normal samples
        n_normal = len(normal_images)
        all_images.extend(normal_images)
        all_labels.extend([0] * n_normal)  # All normal labeled as 0
        
        # Generate anomalies
        for anomaly_type in anomaly_types:
            print(f"  Generating {anomaly_type} anomalies...")
            
            # Sample random normal images to corrupt
            indices = np.random.choice(len(normal_images), n_anomalies_per_type, replace=False)
            
            for idx in indices:
                img = normal_images[idx].clone()
                
                if anomaly_type == 'noise':
                    anomalous_img = self.anomaly_gen.add_noise(img, noise_level=0.5)
                elif anomaly_type == 'rotation':
                    anomalous_img = self.anomaly_gen.rotate(img, angle=90)
                elif anomaly_type == 'occlusion':
                    anomalous_img = self.anomaly_gen.occlude(img, patch_size=7)
                elif anomaly_type == 'out_of_distribution':
                    anomalous_img = self.anomaly_gen.create_out_of_distribution(1)[0]
                else:
                    raise ValueError(f"Unknown anomaly type: {anomaly_type}")
                
                all_images.append(anomalous_img)
                all_labels.append(1)  # Anomaly labeled as 1
        
        # Convert to tensors
        all_images = torch.stack(all_images)
        all_labels = torch.tensor(all_labels)
        
        print(f"  Total samples: {len(all_images)} (Normal: {n_normal}, Anomaly: {len(all_labels) - n_normal})")
        
        dataset = TensorDataset(all_images, all_labels)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)
        
        return loader


def main():
    """Main experiment runner"""
    print("=" * 80)
    print("Phase 2: NCT Anomaly Detection Experiment")
    print("=" * 80)
    
    # Configuration
    config = AnomalyDetectionConfig(
        d_model=384,
        n_heads=6,
        n_layers=3,
        batch_size=128,
        n_epochs=50,
        n_normal_samples=500,
        n_anomaly_samples=100,
        version="v1"
    )
    
    print(f"\nConfiguration:")
    print(f"  d_model: {config.d_model}")
    print(f"  n_heads: {config.n_heads}")
    print(f"  n_layers: {config.n_layers}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.n_epochs}")
    print(f"  Results dir: {config.results_dir}")
    
    # Load MNIST data
    print("\nLoading MNIST dataset...")
    from torchvision import datasets, transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Full training set for normal data
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('../data', train=False, download=True, transform=transform)
    
    # Subsample for faster training
    n_normal = config.n_normal_samples * 10  # 500 per class
    indices = np.random.choice(len(train_dataset), n_normal, replace=False)
    normal_train_data = [train_dataset[i][0] for i in indices]
    normal_train_labels = [train_dataset[i][1] for i in indices]
    
    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(torch.stack(normal_train_data), torch.tensor(normal_train_labels)),
        batch_size=config.batch_size, shuffle=True
    )
    
    # Validation set
    val_indices = np.random.choice(len(test_dataset), 1000, replace=False)
    val_data = [test_dataset[i][0] for i in val_indices]
    val_labels = [test_dataset[i][1] for i in val_indices]
    val_loader = DataLoader(
        TensorDataset(torch.stack(val_data), torch.tensor(val_labels)),
        batch_size=config.batch_size, shuffle=False
    )
    
    # Initialize detector
    detector = NCTAnomalyDetector(config)
    
    # Phase 1: Train on normal data
    print("\n" + "=" * 60)
    print("Phase 2.1: Training on Normal Data")
    print("=" * 60)
    
    training_history = detector.train_on_normal_data(train_loader, val_loader)
    
    # Phase 2: Generate and detect anomalies
    print("\n" + "=" * 60)
    print("Phase 2.2: Anomaly Detection Experiments")
    print("=" * 60)
    
    anomaly_types = ['noise', 'rotation', 'occlusion', 'out_of_distribution']
    all_results = []
    
    for anomaly_type in anomaly_types:
        print(f"\nTesting {anomaly_type} anomalies...")
        
        # Generate anomaly test set
        # Use subset of test data as normal baseline
        n_test_normal = 200
        test_indices = np.random.choice(len(test_dataset), n_test_normal, replace=False)
        test_normal_images = [test_dataset[i][0] for i in test_indices]
        
        anomaly_loader = detector.generate_anomaly_dataset(
            (test_normal_images, [0]*len(test_normal_images)),
            [anomaly_type],
            n_anomalies_per_type=25
        )
        
        # Run detection
        results = detector.detect_anomalies(anomaly_loader, anomaly_type=anomaly_type)
        all_results.append(results)
        
        # Print results
        print(f"\nResults for {anomaly_type}:")
        print(f"  PE-based:      Precision={results['pe_precision']:.3f}, Recall={results['pe_recall']:.3f}, F1={results['pe_f1']:.3f}")
        print(f"  Φ-based:       Precision={results['phi_precision']:.3f}, Recall={results['phi_recall']:.3f}, F1={results['phi_f1']:.3f}")
        print(f"  Combined:      Precision={results['combined_precision']:.3f}, Recall={results['combined_recall']:.3f}, F1={results['combined_f1']:.3f}")
    
    # Save all results
    results_summary = {
        'config': vars(config),
        'training_history': {k: [float(x) for x in v] for k, v in training_history.items()},
        'anomaly_detection_results': all_results,
        'thresholds': detector.thresholds,
        'pe_stats': detector.pe_stats,
        'phi_stats': detector.phi_stats
    }
    
    with open(config.results_dir / 'anomaly_detection_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nResults saved to: {config.results_dir}")
    
    # Create visualization
    print("\nCreating visualizations...")
    visualize_results(training_history, all_results, config.results_dir)
    
    print("\n" + "=" * 80)
    print("Phase 2 Completed Successfully!")
    print("=" * 80)
    
    return results_summary


def visualize_results(training_history, anomaly_results, save_dir):
    """Create comprehensive visualization"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Training curves
    ax = axes[0, 0]
    ax.plot(training_history['train_acc'], label='Train Acc', linewidth=2)
    ax.plot(training_history['val_acc'], label='Val Acc', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Free Energy trajectory
    ax = axes[0, 1]
    ax.plot(training_history['free_energy'], color='red', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Prediction Error (FE)')
    ax.set_title('Free Energy Minimization')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Φ value trajectory
    ax = axes[0, 2]
    ax.plot(training_history['phi'], color='green', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Φ Value')
    ax.set_title('Information Integration (Φ)')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Anomaly detection performance (bar chart)
    ax = axes[1, 0]
    anomaly_types = [r['anomaly_type'] for r in anomaly_results]
    f1_pe = [r['pe_f1'] for r in anomaly_results]
    f1_phi = [r['phi_f1'] for r in anomaly_results]
    f1_combined = [r['combined_f1'] for r in anomaly_results]
    
    x = np.arange(len(anomaly_types))
    width = 0.25
    
    ax.bar(x - width, f1_pe, width, label='PE-based', alpha=0.8)
    ax.bar(x, f1_phi, width, label='Φ-based', alpha=0.8)
    ax.bar(x + width, f1_combined, width, label='Combined', alpha=0.8)
    
    ax.set_xlabel('Anomaly Type')
    ax.set_ylabel('F1 Score')
    ax.set_title('Anomaly Detection Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(anomaly_types, rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Precision-Recall comparison
    ax = axes[1, 1]
    precision_pe = [r['pe_precision'] for r in anomaly_results]
    recall_pe = [r['pe_recall'] for r in anomaly_results]
    precision_phi = [r['phi_precision'] for r in anomaly_results]
    recall_phi = [r['phi_recall'] for r in anomaly_results]
    
    ax.scatter(recall_pe, precision_pe, s=100, label='PE-based', marker='o')
    ax.scatter(recall_phi, precision_phi, s=100, label='Φ-based', marker='s')
    
    for i, atype in enumerate(anomaly_types):
        ax.annotate(atype, (recall_pe[i], precision_pe[i]), fontsize=9)
        ax.annotate(atype, (recall_phi[i], precision_phi[i]), fontsize=9)
    
    ax.set_xlabel('Recall (Sensitivity)')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Trade-off')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Confusion matrix (combined method, average)
    ax = axes[1, 2]
    avg_tp = np.mean([r['combined_tp'] for r in anomaly_results])
    avg_fp = np.mean([r['combined_fp'] for r in anomaly_results])
    avg_tn = np.mean([r['combined_tn'] for r in anomaly_results])
    avg_fn = np.mean([r['combined_fn'] for r in anomaly_results])
    
    confusion_matrix = np.array([[avg_tp, avg_fp], [avg_fn, avg_tn]])
    
    im = ax.imshow(confusion_matrix, cmap='Blues', aspect='auto')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted Normal', 'Predicted Anomaly'])
    ax.set_yticklabels(['Actual Normal', 'Actual Anomaly'])
    ax.set_title('Average Confusion Matrix (Combined)')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{confusion_matrix[i, j]:.1f}', ha='center', va='center',
                   fontsize=12, color='white' if confusion_matrix[i, j] > 50 else 'black')
    
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'anomaly_detection_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {save_dir / 'anomaly_detection_results.png'}")


if __name__ == '__main__':
    results = main()
