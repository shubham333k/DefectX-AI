"""
DefectX AI - Setup Verification Script
Tests environment, dependencies, and model loading
"""

import sys
import os

def test_imports():
    """Test all required imports."""
    print("=" * 60)
    print("🔍 Testing Package Imports...")
    print("=" * 60)
    
    packages = [
        ('streamlit', 'Streamlit'),
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('cv2', 'OpenCV'),
        ('PIL', 'Pillow'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('plotly', 'Plotly'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('ultralytics', 'Ultralytics'),
        ('sklearn', 'scikit-learn'),
    ]
    
    failed = []
    for module, name in packages:
        try:
            __import__(module)
            print(f"✅ {name:20} - OK")
        except ImportError as e:
            print(f"❌ {name:20} - FAILED: {e}")
            failed.append(name)
    
    return failed


def test_utils():
    """Test utils module loading."""
    print("\n" + "=" * 60)
    print("🔍 Testing Utils Module...")
    print("=" * 60)
    
    try:
        from utils import DefectDetector, XAIExplainer, get_available_models
        print("✅ Utils module loaded successfully")
        
        # Test model list
        models = get_available_models()
        print(f"✅ Available models: {', '.join(models[:3])}...")
        
        return True
    except Exception as e:
        print(f"❌ Utils module failed: {e}")
        return False


def test_detector():
    """Test DefectDetector initialization."""
    print("\n" + "=" * 60)
    print("🔍 Testing DefectDetector...")
    print("=" * 60)
    
    try:
        from utils import DefectDetector
        
        print("⏳ Initializing YOLOv8n (this may download ~6MB on first run)...")
        detector = DefectDetector(
            model_name='yolov8n.pt',
            conf_threshold=0.25,
            device='cpu'
        )
        
        print(f"✅ Model loaded: {detector.model_name}")
        print(f"✅ Device: {detector.device}")
        print(f"✅ Confidence threshold: {detector.conf_threshold}")
        print(f"✅ Classes: {list(detector.class_names.values())[:5]}...")
        
        return True
    except Exception as e:
        print(f"❌ Detector initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_xai():
    """Test XAI functionality."""
    print("\n" + "=" * 60)
    print("🔍 Testing XAI (Grad-CAM)...")
    print("=" * 60)
    
    try:
        import numpy as np
        from utils import DefectDetector
        
        # Create test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Initialize detector if not already done
        detector = DefectDetector(model_name='yolov8n.pt', device='cpu')
        
        # Run detection
        print("⏳ Running detection on test image...")
        result = detector.detect(test_image)
        print(f"✅ Detection completed - Found {result['total_defects']} defects")
        
        # Generate Grad-CAM
        print("⏳ Generating Grad-CAM...")
        gradcam = detector.generate_gradcam(test_image, alpha=0.5)
        print(f"✅ Grad-CAM generated: {gradcam.shape}")
        
        return True
    except Exception as e:
        print(f"❌ XAI test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "🚀" * 30)
    print("   DefectX AI - Setup Verification")
    print("🚀" * 30 + "\n")
    
    # Test imports
    failed_imports = test_imports()
    
    if failed_imports:
        print(f"\n❌ FAILED: {len(failed_imports)} packages missing")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)
    
    # Test utils
    if not test_utils():
        print("\n❌ FAILED: Utils module error")
        sys.exit(1)
    
    # Test detector
    if not test_detector():
        print("\n❌ FAILED: Detector initialization error")
        sys.exit(1)
    
    # Test XAI
    if not test_xai():
        print("\n❌ FAILED: XAI functionality error")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print("\n🎉 DefectX AI is ready to use!")
    print("\nTo launch the application:")
    print("   streamlit run app.py")
    print("\nThe dashboard will open at: http://localhost:8501")
    print("=" * 60)


if __name__ == "__main__":
    main()
