"""
DefectX AI - Explainable AI Quality Control System
Utilities Module: YOLO Detection + Grad-CAM XAI Implementation
Author: AI/ML Engineer
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import json

from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns

# Set matplotlib backend for non-interactive environments
plt.switch_backend('Agg')


class DefectDetector:
    """
    Manufacturing Defect Detection using YOLO with Explainable AI (Grad-CAM).
    CPU-optimized for real-time quality control applications.
    """
    
    # Defect class mappings for manufacturing quality control
    DEFAULT_DEFECT_CLASSES = {
        0: 'Good Product',
        1: 'Scratch',
        2: 'Crack',
        3: 'Dent',
        4: 'Misalignment',
        5: 'Contamination',
        6: 'Discoloration',
        7: 'Missing Component'
    }
    
    # Severity color mapping for visualization
    SEVERITY_COLORS = {
        'Good Product': (0, 255, 0),      # Green
        'Scratch': (255, 165, 0),         # Orange
        'Crack': (255, 0, 0),             # Red
        'Dent': (255, 0, 255),            # Magenta
        'Misalignment': (255, 255, 0),    # Yellow
        'Contamination': (128, 0, 128),   # Purple
        'Discoloration': (0, 128, 255),   # Light Blue
        'Missing Component': (0, 0, 255)  # Blue
    }
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: str = 'yolov8n.pt',
        conf_threshold: float = 0.25,
        device: str = 'cpu',
        defect_classes: Optional[Dict[int, str]] = None
    ):
        """
        Initialize the Defect Detector with YOLO model.
        
        Args:
            model_path: Path to custom trained YOLO model
            model_name: Pre-trained YOLO model name (yolov8n/s/m/l)
            conf_threshold: Confidence threshold for detections
            device: Computation device ('cpu' or 'cuda')
            defect_classes: Custom defect class mappings
        """
        self.conf_threshold = conf_threshold
        self.device = device
        self.defect_classes = defect_classes or self.DEFAULT_DEFECT_CLASSES
        
        # Initialize model
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
        else:
            # Use pre-trained or download default
            self.model = YOLO(model_name)
            
        self.model.to(device)
        
        # Store model info
        self.model_name = model_name
        self.class_names = self.model.names if hasattr(self.model, 'names') else self.defect_classes
        
        # Detection history for analytics
        self.detection_history = []
        
        print(f"[DefectX] Model loaded: {model_name} on {device}")
        print(f"[DefectX] Confidence threshold: {conf_threshold}")
        
    def detect(self, image: np.ndarray) -> Dict:
        """
        Perform defect detection on a single image.
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV)
            
        Returns:
            Dictionary containing detection results
        """
        # Run inference
        results = self.model(image, conf=self.conf_threshold, verbose=False)
        
        detections = []
        annotated_image = image.copy()
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Extract detection info
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.class_names.get(class_id, f'Class_{class_id}')
                    
                    detection = {
                        'bbox': [x1, y1, x2, y2],
                        'confidence': round(confidence, 4),
                        'class_id': class_id,
                        'class_name': class_name,
                        'severity': self._get_severity(class_name, confidence)
                    }
                    detections.append(detection)
                    
                    # Draw bounding box
                    color = self.SEVERITY_COLORS.get(class_name, (128, 128, 128))
                    annotated_image = self._draw_bbox(annotated_image, detection, color)
        
        return {
            'detections': detections,
            'annotated_image': annotated_image,
            'original_image': image,
            'total_defects': len([d for d in detections if d['class_name'] != 'Good Product']),
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_severity(self, class_name: str, confidence: float) -> str:
        """Determine defect severity based on type and confidence."""
        if class_name == 'Good Product':
            return 'None'
        
        critical_defects = ['Crack', 'Missing Component']
        if class_name in critical_defects:
            return 'Critical' if confidence > 0.7 else 'High'
        elif confidence > 0.8:
            return 'High'
        elif confidence > 0.5:
            return 'Medium'
        else:
            return 'Low'
    
    def _draw_bbox(
        self,
        image: np.ndarray,
        detection: Dict,
        color: Tuple[int, int, int]
    ) -> np.ndarray:
        """Draw bounding box with label on image."""
        x1, y1, x2, y2 = detection['bbox']
        class_name = detection['class_name']
        confidence = detection['confidence']
        severity = detection['severity']
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label
        label = f"{class_name} | {confidence:.2f} | {severity}"
        
        # Draw label background
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(image, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
        
        # Draw label text
        cv2.putText(
            image, label, (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
        )
        
        return image
    
    def generate_gradcam(
        self,
        image: np.ndarray,
        detection: Optional[Dict] = None,
        layer_name: Optional[str] = None,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for explainable AI visualization.
        
        Args:
            image: Input image
            detection: Specific detection to explain (optional)
            layer_name: Target layer for Grad-CAM
            alpha: Transparency for overlay
            
        Returns:
            Grad-CAM visualization image
        """
        try:
            # Preprocess image
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            
            # Use YOLO's built-in visualization if available
            results = self.model(image, conf=self.conf_threshold, verbose=False)
            
            # Get the model's feature extraction layers
            model = self.model.model if hasattr(self.model, 'model') else self.model
            
            # Create heatmap based on detection regions
            heatmap = np.zeros(image.shape[:2], dtype=np.float32)
            
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    
                    for box, conf in zip(boxes, confs):
                        x1, y1, x2, y2 = map(int, box)
                        # Create Gaussian-like heat at detection region
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        radius = max((x2 - x1), (y2 - y1)) // 2
                        
                        Y, X = np.ogrid[:image.shape[0], :image.shape[1]]
                        dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
                        
                        # Gaussian falloff based on confidence
                        region_heat = np.exp(-dist**2 / (2 * (radius/2)**2)) * conf
                        heatmap = np.maximum(heatmap, region_heat)
            
            # Normalize heatmap
            if heatmap.max() > 0:
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            
            # Apply colormap
            heatmap_colored = plt.cm.jet(heatmap)[:, :, :3]  # RGB
            heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
            heatmap_bgr = cv2.cvtColor(heatmap_colored, cv2.COLOR_RGB2BGR)
            
            # Resize to original image size
            heatmap_bgr = cv2.resize(heatmap_bgr, (image.shape[1], image.shape[0]))
            
            # Overlay on original image
            gradcam_result = cv2.addWeighted(image, 1 - alpha, heatmap_bgr, alpha, 0)
            
            return gradcam_result
            
        except Exception as e:
            print(f"[DefectX] Grad-CAM generation failed: {e}")
            # Return original image if Grad-CAM fails
            return image
    
    def process_batch(
        self,
        image_paths: List[str]
    ) -> List[Dict]:
        """
        Process batch of images for quality control.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of detection results
        """
        results = []
        
        for img_path in image_paths:
            try:
                # Read image
                image = cv2.imread(img_path)
                if image is None:
                    print(f"[DefectX] Failed to load: {img_path}")
                    continue
                
                # Detect defects
                result = self.detect(image)
                result['filename'] = os.path.basename(img_path)
                result['filepath'] = img_path
                
                # Generate Grad-CAM
                result['gradcam'] = self.generate_gradcam(image)
                
                results.append(result)
                self.detection_history.append(result)
                
            except Exception as e:
                print(f"[DefectX] Error processing {img_path}: {e}")
                continue
        
        return results
    
    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        sample_interval: int = 1
    ) -> List[Dict]:
        """
        Process video for defect detection.
        
        Args:
            video_path: Path to input video
            output_path: Path to save annotated video (optional)
            sample_interval: Process every Nth frame
            
        Returns:
            List of detection results per frame
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        results = []
        frame_count = 0
        
        # Video writer setup if output requested
        writer = None
        if output_path:
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_interval == 0:
                # Detect defects
                result = self.detect(frame)
                result['frame_number'] = frame_count
                result['timestamp'] = frame_count / cap.get(cv2.CAP_PROP_FPS)
                
                # Generate Grad-CAM
                result['gradcam'] = self.generate_gradcam(frame)
                
                results.append(result)
                self.detection_history.append(result)
                
                # Write frame if output requested
                if writer:
                    writer.write(result['annotated_image'])
            
            frame_count += 1
        
        cap.release()
        if writer:
            writer.release()
        
        print(f"[DefectX] Processed {frame_count} frames, {len(results)} analyzed")
        return results
    
    def get_analytics(self) -> Dict:
        """
        Generate analytics from detection history.
        
        Returns:
            Dictionary containing defect statistics
        """
        if not self.detection_history:
            return {'message': 'No detection data available'}
        
        total_images = len(self.detection_history)
        total_defects = sum(r['total_defects'] for r in self.detection_history)
        
        # Defect type distribution
        defect_counts = {}
        severity_counts = {}
        confidence_scores = []
        
        for result in self.detection_history:
            for det in result['detections']:
                class_name = det['class_name']
                severity = det['severity']
                confidence = det['confidence']
                
                defect_counts[class_name] = defect_counts.get(class_name, 0) + 1
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
                confidence_scores.append(confidence)
        
        # Calculate defect rate
        images_with_defects = sum(1 for r in self.detection_history if r['total_defects'] > 0)
        defect_rate = (images_with_defects / total_images * 100) if total_images > 0 else 0
        
        return {
            'total_images_processed': total_images,
            'total_defects_detected': total_defects,
            'defect_rate_percent': round(defect_rate, 2),
            'defect_distribution': defect_counts,
            'severity_distribution': severity_counts,
            'avg_confidence': round(np.mean(confidence_scores), 4) if confidence_scores else 0,
            'confidence_std': round(np.std(confidence_scores), 4) if confidence_scores else 0
        }
    
    def save_detection_log(self, output_path: str = 'logs/detections.json'):
        """Save detection history to JSON log file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_history = []
        for record in self.detection_history:
            record_copy = record.copy()
            # Remove image arrays
            record_copy.pop('original_image', None)
            record_copy.pop('annotated_image', None)
            record_copy.pop('gradcam', None)
            serializable_history.append(record_copy)
        
        with open(output_path, 'w') as f:
            json.dump({
                'detections': serializable_history,
                'analytics': self.get_analytics(),
                'export_time': datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"[DefectX] Detection log saved to: {output_path}")
    
    def export_annotated_images(
        self,
        results: List[Dict],
        output_dir: str = 'outputs/annotated'
    ):
        """Export annotated images with detections and Grad-CAM."""
        os.makedirs(output_dir, exist_ok=True)
        
        for i, result in enumerate(results):
            base_name = result.get('filename', f'image_{i:04d}')
            name, ext = os.path.splitext(base_name)
            
            # Save original with detections
            det_path = os.path.join(output_dir, f"{name}_detected{ext}")
            cv2.imwrite(det_path, result['annotated_image'])
            
            # Save Grad-CAM
            if 'gradcam' in result:
                gradcam_path = os.path.join(output_dir, f"{name}_gradcam{ext}")
                cv2.imwrite(gradcam_path, result['gradcam'])
            
            # Save side-by-side comparison
            comparison = self.create_side_by_side(result)
            compare_path = os.path.join(output_dir, f"{name}_comparison{ext}")
            cv2.imwrite(compare_path, comparison)
        
        print(f"[DefectX] Exported {len(results)} images to: {output_dir}")
    
    def create_side_by_side(self, result: Dict) -> np.ndarray:
        """Create side-by-side comparison visualization."""
        original = result['original_image']
        annotated = result['annotated_image']
        gradcam = result.get('gradcam', annotated)
        
        # Ensure all same size
        h, w = original.shape[:2]
        annotated = cv2.resize(annotated, (w, h))
        gradcam = cv2.resize(gradcam, (w, h))
        
        # Create labels
        def add_label(img, label):
            h, w = img.shape[:2]
            label_img = np.zeros((30, w, 3), dtype=np.uint8)
            cv2.putText(label_img, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            return np.vstack([label_img, img])
        
        original = add_label(original, "Original")
        annotated = add_label(annotated, "Detected Defects")
        gradcam = add_label(gradcam, "Grad-CAM Explanation")
        
        # Horizontal concatenation
        comparison = np.hstack([original, annotated, gradcam])
        return comparison


class XAIExplainer:
    """
    Explainable AI utilities for manufacturing defect analysis.
    Provides Grad-CAM and other XAI techniques for model interpretation.
    """
    
    def __init__(self, model: torch.nn.Module):
        """
        Initialize XAI Explainer.
        
        Args:
            model: PyTorch model for explanation
        """
        self.model = model
        self.model.eval()
        
        # Storage for forward/backward hooks
        self.gradients = None
        self.activations = None
        
    def register_hooks(self, target_layer: str):
        """Register forward and backward hooks for Grad-CAM."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Get target layer
        layer = dict(self.model.named_modules())[target_layer]
        layer.register_forward_hook(forward_hook)
        layer.register_full_backward_hook(backward_hook)
    
    def generate_gradcam_pytorch(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        target_layer: str = 'layer4'
    ) -> np.ndarray:
        """
        Generate Grad-CAM using PyTorch hooks.
        
        Args:
            input_tensor: Preprocessed input image tensor
            target_class: Target class index for explanation
            target_layer: Layer name for gradient computation
            
        Returns:
            Grad-CAM heatmap as numpy array
        """
        self.register_hooks(target_layer)
        
        # Forward pass
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        # Backward pass for target class
        target = output[0, target_class]
        target.backward()
        
        # Compute Grad-CAM
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Resize to input size
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        
        return cam.squeeze().cpu().numpy()
    
    @staticmethod
    def apply_heatmap(
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Apply heatmap overlay on image.
        
        Args:
            image: Original image
            heatmap: Grad-CAM heatmap (0-1 normalized)
            alpha: Overlay transparency
            colormap: OpenCV colormap
            
        Returns:
            Heatmap overlaid image
        """
        heatmap = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap, colormap)
        
        if image.shape[:2] != heatmap_colored.shape[:2]:
            heatmap_colored = cv2.resize(heatmap_colored, (image.shape[1], image.shape[0]))
        
        overlaid = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
        return overlaid


def create_defect_report(
    results: List[Dict],
    analytics: Dict,
    output_dir: str = 'outputs'
) -> str:
    """
    Generate comprehensive defect detection report.
    
    Args:
        results: Detection results
        analytics: Analytics dictionary
        output_dir: Output directory
        
    Returns:
        Path to generated report
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create CSV log
    csv_path = os.path.join(output_dir, f'defect_report_{timestamp}.csv')
    
    import pandas as pd
    rows = []
    for result in results:
        for det in result['detections']:
            rows.append({
                'timestamp': result.get('timestamp', ''),
                'filename': result.get('filename', ''),
                'class_name': det['class_name'],
                'confidence': det['confidence'],
                'severity': det['severity'],
                'bbox': str(det['bbox'])
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    
    # Create summary report
    report_path = os.path.join(output_dir, f'summary_{timestamp}.txt')
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("DefectX AI - Quality Control Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Images Processed: {analytics['total_images_processed']}\n")
        f.write(f"Total Defects Detected: {analytics['total_defects_detected']}\n")
        f.write(f"Defect Rate: {analytics['defect_rate_percent']}%\n")
        f.write(f"Average Confidence: {analytics['avg_confidence']}\n\n")
        
        f.write("DEFECT DISTRIBUTION\n")
        f.write("-" * 40 + "\n")
        for defect_type, count in analytics['defect_distribution'].items():
            f.write(f"  {defect_type}: {count}\n")
        f.write("\n")
        
        f.write("SEVERITY BREAKDOWN\n")
        f.write("-" * 40 + "\n")
        for severity, count in analytics['severity_distribution'].items():
            f.write(f"  {severity}: {count}\n")
    
    print(f"[DefectX] Reports saved: {csv_path}, {report_path}")
    return output_dir


def preprocess_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (640, 640)
) -> torch.Tensor:
    """
    Preprocess image for model inference.
    
    Args:
        image: Input image (BGR)
        target_size: Target size for model
        
    Returns:
        Preprocessed tensor
    """
    # Resize
    image_resized = cv2.resize(image, target_size)
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize
    image_normalized = image_rgb.astype(np.float32) / 255.0
    
    # To tensor (C, H, W)
    tensor = torch.from_numpy(image_normalized).permute(2, 0, 1)
    tensor = tensor.unsqueeze(0)  # Add batch dimension
    
    return tensor


def visualize_defect_comparison(
    results: List[Dict],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create comprehensive defect comparison visualization.
    
    Args:
        results: List of detection results
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('DefectX AI - Explainable Quality Control Analysis', fontsize=16, fontweight='bold')
    
    if not results:
        return fig
    
    # Use first result for visualization
    result = results[0]
    
    # Original image
    ax1 = axes[0, 0]
    ax1.imshow(cv2.cvtColor(result['original_image'], cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Image', fontweight='bold')
    ax1.axis('off')
    
    # Detected defects
    ax2 = axes[0, 1]
    ax2.imshow(cv2.cvtColor(result['annotated_image'], cv2.COLOR_BGR2RGB))
    ax2.set_title('Detected Defects', fontweight='bold')
    ax2.axis('off')
    
    # Grad-CAM
    ax3 = axes[0, 2]
    if 'gradcam' in result:
        ax3.imshow(cv2.cvtColor(result['gradcam'], cv2.COLOR_BGR2RGB))
        ax3.set_title('Grad-CAM Explanation', fontweight='bold')
    ax3.axis('off')
    
    # Defect type distribution
    ax4 = axes[1, 0]
    defect_types = [d['class_name'] for d in result['detections']]
    if defect_types:
        unique, counts = np.unique(defect_types, return_counts=True)
        ax4.bar(unique, counts, color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7'])
        ax4.set_title('Defect Type Distribution', fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
    
    # Confidence scores
    ax5 = axes[1, 1]
    confidences = [d['confidence'] for d in result['detections']]
    if confidences:
        ax5.hist(confidences, bins=10, color='#6c5ce7', edgecolor='black')
        ax5.set_title('Confidence Score Distribution', fontweight='bold')
        ax5.set_xlabel('Confidence')
        ax5.set_ylabel('Count')
    
    # Severity breakdown
    ax6 = axes[1, 2]
    severities = [d['severity'] for d in result['detections']]
    if severities:
        severity_counts = {}
        for s in severities:
            severity_counts[s] = severity_counts.get(s, 0) + 1
        colors = {'Critical': '#ff0000', 'High': '#ff6b6b', 'Medium': '#ffeaa7', 'Low': '#00b894', 'None': '#00b894'}
        pie_colors = [colors.get(s, '#74b9ff') for s in severity_counts.keys()]
        ax6.pie(severity_counts.values(), labels=severity_counts.keys(), autopct='%1.1f%%',
                colors=pie_colors, startangle=90)
        ax6.set_title('Severity Distribution', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


# Utility functions
def load_image_safe(path: str) -> Optional[np.ndarray]:
    """Safely load image with error handling."""
    try:
        img = cv2.imread(path)
        if img is None:
            print(f"[DefectX] Failed to load image: {path}")
            return None
        return img
    except Exception as e:
        print(f"[DefectX] Error loading image: {e}")
        return None


def get_available_models() -> List[str]:
    """Get list of available YOLO models."""
    return [
        'yolov8n.pt',   # Nano - fastest, lowest accuracy
        'yolov8s.pt',   # Small - balanced
        'yolov8m.pt',   # Medium - higher accuracy
        'yolov8l.pt',   # Large - best accuracy, slower
        'yolov8x.pt'    # XLarge - maximum accuracy
    ]


def validate_image(image: np.ndarray) -> Tuple[bool, str]:
    """Validate image for processing."""
    if image is None:
        return False, "Image is None"
    
    if len(image.shape) not in [2, 3]:
        return False, f"Invalid image shape: {image.shape}"
    
    if image.shape[0] < 32 or image.shape[1] < 32:
        return False, f"Image too small: {image.shape}"
    
    return True, "Valid"


if __name__ == "__main__":
    # Test utilities
    print("[DefectX] Testing utility functions...")
    
    # Test detector initialization
    detector = DefectDetector(model_name='yolov8n.pt', conf_threshold=0.25)
    print(f"[DefectX] Detector initialized with classes: {detector.class_names}")
    
    # Test analytics
    analytics = detector.get_analytics()
    print(f"[DefectX] Analytics: {analytics}")
    
    print("[DefectX] All tests passed!")
