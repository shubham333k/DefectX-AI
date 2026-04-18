"""
DefectX AI - Explainable AI Quality Control System
Streamlit Web Application for Manufacturing Defect Detection
Author: AI/ML Engineer
"""

import os
import sys
import time
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="DefectX AI - Explainable Quality Control",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for professional styling
def load_css():
    st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #ff7f0e;
        --success-color: #2ca02c;
        --danger-color: #d62728;
        --warning-color: #ffbb78;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5em;
        font-weight: bold;
    }
    
    .main-header p {
        margin: 10px 0 0 0;
        font-size: 1.2em;
        opacity: 0.9;
    }
    
    /* XAI Section styling */
    .xai-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 25px;
        border-radius: 8px;
        margin: 15px 0;
        border-left: 5px solid #ffd700;
    }
    
    .xai-banner h3 {
        margin: 0;
        color: #ffd700;
        font-size: 1.3em;
    }
    
    .xai-banner p {
        margin: 8px 0 0 0;
        font-size: 0.95em;
        line-height: 1.5;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .metric-value {
        font-size: 2em;
        font-weight: bold;
        color: var(--primary-color);
    }
    
    .metric-label {
        font-size: 0.9em;
        color: #666;
        margin-top: 5px;
    }
    
    /* Defect severity badges */
    .badge-critical { background-color: #d62728; color: white; padding: 3px 10px; border-radius: 12px; font-size: 0.8em; }
    .badge-high { background-color: #ff7f0e; color: white; padding: 3px 10px; border-radius: 12px; font-size: 0.8em; }
    .badge-medium { background-color: #ffbb78; color: black; padding: 3px 10px; border-radius: 12px; font-size: 0.8em; }
    .badge-low { background-color: #2ca02c; color: white; padding: 3px 10px; border-radius: 12px; font-size: 0.8em; }
    .badge-none { background-color: #1f77b4; color: white; padding: 3px 10px; border-radius: 12px; font-size: 0.8em; }
    
    /* Section headers */
    .section-header {
        background: #f8f9fa;
        padding: 10px 15px;
        border-left: 4px solid #667eea;
        margin: 20px 0 15px 0;
        font-weight: bold;
        font-size: 1.2em;
    }
    
    /* Status indicators */
    .status-pass { color: #2ca02c; font-weight: bold; }
    .status-fail { color: #d62728; font-weight: bold; }
    .status-warning { color: #ff7f0e; font-weight: bold; }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 20px;
        color: #666;
        font-size: 0.9em;
        border-top: 1px solid #eee;
        margin-top: 40px;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background-color: #667eea;
    }
    
    /* Custom buttons */
    .stButton > button {
        background-color: #667eea;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        background-color: #764ba2;
    }
    
    /* Image containers */
    .image-container {
        border: 2px solid #e0e0e0;
        border-radius: 8px;
        padding: 10px;
        background: white;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 15px;
        border-radius: 4px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Import utilities
from utils import (
    DefectDetector, 
    XAIExplainer, 
    create_defect_report,
    visualize_defect_comparison,
    load_image_safe,
    get_available_models,
    validate_image,
    preprocess_image
)

# Initialize session state
def init_session_state():
    """Initialize Streamlit session state variables."""
    if 'detector' not in st.session_state:
        st.session_state.detector = None
    if 'detection_history' not in st.session_state:
        st.session_state.detection_history = []
    if 'current_results' not in st.session_state:
        st.session_state.current_results = []
    if 'processing_stats' not in st.session_state:
        st.session_state.processing_stats = {
            'total_images': 0,
            'total_defects': 0,
            'defect_types': {},
            'start_time': None
        }


def render_header():
    """Render application header."""
    st.markdown("""
    <div class="main-header">
        <h1>🔍 DefectX AI</h1>
        <p>Explainable AI Quality Control System for Manufacturing</p>
    </div>
    """, unsafe_allow_html=True)
    
    # XAI Banner
    st.markdown("""
    <div class="xai-banner">
        <h3>🧠 Explainable AI (XAI) Powered</h3>
        <p>
            DefectX AI combines <strong>YOLO object detection</strong> with <strong>Grad-CAM visual explanations</strong> 
            to provide transparent, interpretable quality control. Our system not only detects defects but also 
            <strong>explains WHY</strong> it made each decision, building trust in automated manufacturing inspection.
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render sidebar controls."""
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model Selection
        st.subheader("Model Settings")
        available_models = get_available_models()
        model_choice = st.selectbox(
            "Select YOLO Model",
            available_models,
            index=0,
            help="Nano (n) = Fastest, XLarge (x) = Most accurate"
        )
        
        # Confidence threshold
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.25,
            step=0.05,
            help="Minimum confidence for defect detection"
        )
        
        # XAI Settings
        st.subheader("XAI Settings")
        enable_gradcam = st.checkbox("Enable Grad-CAM", value=True, help="Generate explanation heatmaps")
        gradcam_alpha = st.slider(
            "Heatmap Intensity",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.1,
            help="Transparency of Grad-CAM overlay"
        ) if enable_gradcam else 0.5
        
        # Processing settings
        st.subheader("Processing")
        device = st.radio("Device", ["CPU", "CUDA"], index=0, help="CUDA requires NVIDIA GPU")
        
        # Initialize/Update detector
        device_str = "cuda" if device == "CUDA" else "cpu"
        
        if st.button("🚀 Initialize Model", type="primary"):
            with st.spinner(f"Loading {model_choice}..."):
                st.session_state.detector = DefectDetector(
                    model_name=model_choice,
                    conf_threshold=conf_threshold,
                    device=device_str
                )
                st.success(f"Model loaded: {model_choice}")
        
        # System info
        st.divider()
        st.subheader("📊 Session Stats")
        if st.session_state.detection_history:
            st.metric("Images Processed", len(st.session_state.detection_history))
            total_defects = sum(r['total_defects'] for r in st.session_state.detection_history)
            st.metric("Defects Found", total_defects)
        else:
            st.info("No images processed yet")
        
        # Reset button
        if st.button("🔄 Reset Session"):
            st.session_state.detection_history = []
            st.session_state.current_results = []
            st.session_state.processing_stats = {
                'total_images': 0,
                'total_defects': 0,
                'defect_types': {},
                'start_time': None
            }
            st.rerun()
        
        return {
            'model': model_choice,
            'conf_threshold': conf_threshold,
            'enable_gradcam': enable_gradcam,
            'gradcam_alpha': gradcam_alpha,
            'device': device_str
        }


def render_file_upload():
    """Render file upload section."""
    st.markdown('<div class="section-header">📤 Upload Inspection Data</div>', unsafe_allow_html=True)
    
    upload_type = st.radio(
        "Select Upload Type",
        ["Single Image", "Batch Images", "Video"],
        horizontal=True
    )
    
    if upload_type == "Single Image":
        uploaded_file = st.file_uploader(
            "Upload product image for inspection",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            accept_multiple_files=False
        )
        return upload_type, [uploaded_file] if uploaded_file else []
    
    elif upload_type == "Batch Images":
        uploaded_files = st.file_uploader(
            "Upload multiple product images",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            accept_multiple_files=True
        )
        return upload_type, uploaded_files or []
    
    else:  # Video
        uploaded_file = st.file_uploader(
            "Upload production line video",
            type=['mp4', 'avi', 'mov', 'mkv'],
            accept_multiple_files=False
        )
        return upload_type, [uploaded_file] if uploaded_file else []


def process_single_image(image_file, detector: DefectDetector, config: Dict) -> Optional[Dict]:
    """Process a single image file."""
    try:
        # Read image
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            st.error("Failed to load image")
            return None
        
        # Detect defects
        start_time = time.time()
        result = detector.detect(image)
        result['filename'] = image_file.name
        result['processing_time'] = time.time() - start_time
        
        # Generate Grad-CAM if enabled
        if config['enable_gradcam']:
            result['gradcam'] = detector.generate_gradcam(
                image, 
                alpha=config['gradcam_alpha']
            )
        
        return result
        
    except Exception as e:
        st.error(f"Processing error: {e}")
        return None


def render_results_panel(result: Dict, config: Dict):
    """Render detection results for a single image."""
    st.markdown('<div class="section-header">🔍 Inspection Results</div>', unsafe_allow_html=True)
    
    # Quality verdict
    is_defective = result['total_defects'] > 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if is_defective:
            st.markdown('<p class="status-fail">❌ QUALITY CHECK: FAILED</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-pass">✅ QUALITY CHECK: PASSED</p>', unsafe_allow_html=True)
    
    with col2:
        st.metric("Defects Found", result['total_defects'])
    
    with col3:
        st.metric("Processing Time", f"{result.get('processing_time', 0):.3f}s")
    
    # Image display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.markdown("**Original Image**")
        image_rgb = cv2.cvtColor(result['original_image'], cv2.COLOR_BGR2RGB)
        st.image(image_rgb, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.markdown("**Detected Defects**")
        annotated_rgb = cv2.cvtColor(result['annotated_image'], cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.markdown("**Grad-CAM Explanation**")
        if 'gradcam' in result:
            gradcam_rgb = cv2.cvtColor(result['gradcam'], cv2.COLOR_BGR2RGB)
            st.image(gradcam_rgb, use_container_width=True)
            st.caption("🧠 Regions that influenced AI decision")
        else:
            st.info("Grad-CAM not enabled")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Detection details
    if result['detections']:
        st.markdown("### 📋 Detection Details")
        
        detection_data = []
        for i, det in enumerate(result['detections']):
            detection_data.append({
                'ID': i + 1,
                'Defect Type': det['class_name'],
                'Confidence': f"{det['confidence']:.2%}",
                'Severity': det['severity'],
                'Bounding Box': f"({det['bbox'][0]}, {det['bbox'][1]}) - ({det['bbox'][2]}, {det['bbox'][3]})"
            })
        
        df = pd.DataFrame(detection_data)
        
        # Apply severity styling
        def highlight_severity(val):
            colors = {
                'Critical': 'background-color: #ffcccc',
                'High': 'background-color: #ffe6cc',
                'Medium': 'background-color: #ffffcc',
                'Low': 'background-color: #ccffcc',
                'None': 'background-color: #cce5ff'
            }
            return colors.get(val, '')
        
        styled_df = df.style.applymap(highlight_severity, subset=['Severity'])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    # Side-by-side comparison
    st.markdown("### 📊 Side-by-Side Comparison")
    comparison = detector.create_side_by_side(result)
    comparison_rgb = cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB)
    st.image(comparison_rgb, use_container_width=True, caption="Original | Detection | Explanation")
    
    # Download buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        annotated_bytes = cv2.imencode('.png', result['annotated_image'])[1].tobytes()
        st.download_button(
            "📥 Download Detection",
            annotated_bytes,
            f"detected_{result['filename']}",
            "image/png"
        )
    
    with col2:
        if 'gradcam' in result:
            gradcam_bytes = cv2.imencode('.png', result['gradcam'])[1].tobytes()
            st.download_button(
                "📥 Download Grad-CAM",
                gradcam_bytes,
                f"gradcam_{result['filename']}",
                "image/png"
            )
    
    with col3:
        comparison_bytes = cv2.imencode('.png', comparison)[1].tobytes()
        st.download_button(
            "📥 Download Comparison",
            comparison_bytes,
            f"comparison_{result['filename']}",
            "image/png"
        )


def render_analytics_dashboard(detector: DefectDetector):
    """Render analytics dashboard with visualizations."""
    st.markdown('<div class="section-header">📈 Quality Control Analytics</div>', unsafe_allow_html=True)
    
    analytics = detector.get_analytics()
    
    if 'message' in analytics:
        st.info("Process images to see analytics")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📸 Images Processed", analytics['total_images_processed'])
    
    with col2:
        st.metric("🔴 Total Defects", analytics['total_defects_detected'])
    
    with col3:
        st.metric("📊 Defect Rate", f"{analytics['defect_rate_percent']}%")
    
    with col4:
        st.metric("🎯 Avg Confidence", f"{analytics['avg_confidence']:.2%}")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Defect type distribution
        if analytics['defect_distribution']:
            fig = px.pie(
                values=list(analytics['defect_distribution'].values()),
                names=list(analytics['defect_distribution'].keys()),
                title="Defect Type Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3,
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No defects detected yet")
    
    with col2:
        # Severity distribution
        if analytics['severity_distribution']:
            severity_order = ['Critical', 'High', 'Medium', 'Low', 'None']
            severity_data = {k: analytics['severity_distribution'].get(k, 0) for k in severity_order}
            
            colors = {'Critical': '#d62728', 'High': '#ff7f0e', 'Medium': '#ffbb78', 
                     'Low': '#2ca02c', 'None': '#1f77b4'}
            
            fig = px.bar(
                x=list(severity_data.keys()),
                y=list(severity_data.values()),
                title="Defect Severity Distribution",
                labels={'x': 'Severity', 'y': 'Count'},
                color=list(severity_data.keys()),
                color_discrete_map=colors
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No severity data available")
    
    # Trend analysis
    if len(detector.detection_history) > 1:
        st.markdown("### 📉 Detection Trend")
        
        trend_data = []
        for i, record in enumerate(detector.detection_history):
            trend_data.append({
                'Index': i,
                'Defects': record['total_defects'],
                'Image': record.get('filename', f'Image_{i}')
            })
        
        trend_df = pd.DataFrame(trend_data)
        fig = px.line(
            trend_df,
            x='Index',
            y='Defects',
            title="Defect Detection Trend Over Time",
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)


def render_batch_results(results: List[Dict], detector: DefectDetector):
    """Render results for batch processing."""
    st.markdown('<div class="section-header">📦 Batch Processing Results</div>', unsafe_allow_html=True)
    
    st.success(f"✅ Processed {len(results)} images")
    
    # Summary statistics
    total_defects = sum(r['total_defects'] for r in results)
    defective_images = sum(1 for r in results if r['total_defects'] > 0)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Images", len(results))
    with col2:
        st.metric("Defective Images", defective_images)
    with col3:
        st.metric("Total Defects Found", total_defects)
    
    # Results table
    summary_data = []
    for r in results:
        status = "❌ FAIL" if r['total_defects'] > 0 else "✅ PASS"
        defect_types = ', '.join(set(d['class_name'] for d in r['detections']))
        summary_data.append({
            'Filename': r['filename'],
            'Status': status,
            'Defects': r['total_defects'],
            'Defect Types': defect_types if defect_types else 'None',
            'Confidence': f"{r['detections'][0]['confidence']:.2%}" if r['detections'] else 'N/A'
        })
    
    df = pd.DataFrame(summary_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Individual results expandable
    st.markdown("### 🔍 Individual Inspection Results")
    for i, result in enumerate(results):
        with st.expander(f"📄 {result['filename']} - {'❌ Defects Found' if result['total_defects'] > 0 else '✅ Good'}"):
            render_single_result_preview(result)
    
    # Export options
    st.markdown("### 📥 Export Reports")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Generate CSV Report", type="primary"):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
                # Create CSV
                rows = []
                for result in results:
                    for det in result['detections']:
                        rows.append({
                            'filename': result['filename'],
                            'timestamp': result.get('timestamp', ''),
                            'class_name': det['class_name'],
                            'confidence': det['confidence'],
                            'severity': det['severity'],
                            'bbox': str(det['bbox'])
                        })
                
                df_export = pd.DataFrame(rows)
                csv_path = f.name
                df_export.to_csv(csv_path, index=False)
                
                with open(csv_path, 'rb') as f_read:
                    st.download_button(
                        "📥 Download CSV",
                        f_read.read(),
                        f"defect_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv"
                    )
    
    with col2:
        if st.button("🖼️ Export All Images"):
            output_dir = f"outputs/batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            detector.export_annotated_images(results, output_dir)
            st.success(f"Images exported to: {output_dir}")


def render_single_result_preview(result: Dict):
    """Render a compact preview of a single result."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Original**")
        image_rgb = cv2.cvtColor(result['original_image'], cv2.COLOR_BGR2RGB)
        st.image(image_rgb, use_container_width=True)
    
    with col2:
        st.markdown("**Detection**")
        annotated_rgb = cv2.cvtColor(result['annotated_image'], cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, use_container_width=True)
    
    with col3:
        st.markdown("**Explanation**")
        if 'gradcam' in result:
            gradcam_rgb = cv2.cvtColor(result['gradcam'], cv2.COLOR_BGR2RGB)
            st.image(gradcam_rgb, use_container_width=True)
        else:
            st.image(annotated_rgb, use_container_width=True)


def main():
    """Main application entry point."""
    load_css()
    init_session_state()
    
    render_header()
    config = render_sidebar()
    
    # Check if detector is initialized
    if st.session_state.detector is None:
        st.info("👈 Please initialize the model from the sidebar to begin inspection")
        
        # Show sample workflow
        st.markdown("""
        <div class="info-box">
            <h4>🚀 Quick Start Guide</h4>
            <ol>
                <li>Select your preferred YOLO model (Nano for speed, XLarge for accuracy)</li>
                <li>Set confidence threshold (default: 25%)</li>
                <li>Enable Grad-CAM for explainable AI visualization</li>
                <li>Click <strong>Initialize Model</strong></li>
                <li>Upload images or video for inspection</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Features overview
        st.markdown("""
        ### 🎯 Key Features
        
        | Feature | Description |
        |---------|-------------|
        | **🔍 YOLO Detection** | Real-time defect detection with bounding boxes |
        | **🧠 Grad-CAM XAI** | Visual explanations showing model attention |
        | **📊 Analytics** | Defect rate tracking and trend analysis |
        | **📹 Video Support** | Process production line footage |
        | **📥 Export Reports** | CSV logs and annotated images |
        | **⚡ CPU Optimized** | Runs efficiently on standard hardware |
        
        ### 🏭 Supported Defect Types
        
        - ✅ Good Product | 🔴 Crack | 🟠 Scratch | 🟡 Dent | 🔵 Misalignment | 🟣 Contamination
        """)
        
        return
    
    # Get the detector
    detector = st.session_state.detector
    
    # Update detector settings
    detector.conf_threshold = config['conf_threshold']
    
    # File upload section
    upload_type, uploaded_files = render_file_upload()
    
    if uploaded_files and len(uploaded_files) > 0 and uploaded_files[0] is not None:
        # Process based on type
        if upload_type == "Single Image":
            with st.spinner("🔍 Analyzing image..."):
                result = process_single_image(uploaded_files[0], detector, config)
                
                if result:
                    st.session_state.current_results = [result]
                    st.session_state.detection_history.append(result)
                    render_results_panel(result, config)
                    
        elif upload_type == "Batch Images":
            progress_bar = st.progress(0)
            results = []
            
            for i, file in enumerate(uploaded_files):
                with st.spinner(f"Processing {i+1}/{len(uploaded_files)}: {file.name}..."):
                    file.seek(0)  # Reset file pointer
                    result = process_single_image(file, detector, config)
                    if result:
                        results.append(result)
                        st.session_state.detection_history.append(result)
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            if results:
                st.session_state.current_results = results
                render_batch_results(results, detector)
            
            progress_bar.empty()
            
        else:  # Video
            st.info("🎬 Video processing feature processes frames at 1-second intervals")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_files[0].read())
                video_path = tmp_file.name
            
            with st.spinner("Processing video..."):
                output_dir = f"outputs/video_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                os.makedirs(output_dir, exist_ok=True)
                
                # Use default FPS since predictor may not be initialized yet
                default_fps = 30
                try:
                    # Try to get FPS from video file
                    cap = cv2.VideoCapture(video_path)
                    fps = int(cap.get(cv2.CAP_PROP_FPS)) or default_fps
                    cap.release()
                except:
                    fps = default_fps
                
                results = detector.process_video(
                    video_path,
                    output_path=os.path.join(output_dir, "annotated_video.mp4"),
                    sample_interval=fps
                )
                
                st.success(f"✅ Processed {len(results)} frames from video")
                
                # Show sample frames
                st.markdown("### 📽️ Sample Frames")
                sample_indices = [0, len(results)//2, len(results)-1] if len(results) >= 3 else [0]
                
                cols = st.columns(len(sample_indices))
                for idx, frame_idx in enumerate(sample_indices):
                    if frame_idx < len(results):
                        with cols[idx]:
                            frame_rgb = cv2.cvtColor(results[frame_idx]['annotated_image'], cv2.COLOR_BGR2RGB)
                            st.image(frame_rgb, caption=f"Frame {frame_idx + 1}", use_container_width=True)
                
                # Video download
                if os.path.exists(os.path.join(output_dir, "annotated_video.mp4")):
                    with open(os.path.join(output_dir, "annotated_video.mp4"), 'rb') as f:
                        st.download_button(
                            "📥 Download Annotated Video",
                            f.read(),
                            "annotated_video.mp4",
                            "video/mp4"
                        )
            
            os.unlink(video_path)
    
    # Analytics Dashboard
    if st.session_state.detection_history:
        st.divider()
        render_analytics_dashboard(detector)
        
        # Save detection log
        detector.detection_history = st.session_state.detection_history
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        detector.save_detection_log(os.path.join(log_dir, "detections.json"))
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>🔍 DefectX AI - Explainable Quality Control System</p>
        <p>Built with YOLO + Grad-CAM | Streamlit | PyTorch</p>
        <p>© 2024 DefectX AI - Manufacturing Intelligence</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
