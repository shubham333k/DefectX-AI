"""
DefectX AI - Sample Data Generator
Creates synthetic manufacturing defect images for testing
"""

import os
import cv2
import numpy as np
from pathlib import Path


def create_base_product(size=(640, 480), product_type='pcb'):
    """Create base product image."""
    img = np.ones((size[1], size[0], 3), dtype=np.uint8) * 240
    
    if product_type == 'pcb':
        # PCB board base
        cv2.rectangle(img, (50, 50), (size[0]-50, size[1]-50), (34, 139, 34), -1)
        cv2.rectangle(img, (50, 50), (size[0]-50, size[1]-50), (0, 100, 0), 3)
        
        # Add some components
        for i in range(5):
            x = 100 + i * 100
            y = 100
            cv2.rectangle(img, (x-20, y-15), (x+20, y+15), (169, 169, 169), -1)
            cv2.rectangle(img, (x-20, y-15), (x+20, y+15), (100, 100, 100), 2)
        
        for i in range(4):
            x = 150 + i * 100
            y = 250
            cv2.circle(img, (x, y), 20, (139, 69, 19), -1)
            cv2.circle(img, (x, y), 20, (100, 50, 10), 2)
    
    elif product_type == 'metal':
        # Metal surface
        img = np.ones((size[1], size[0], 3), dtype=np.uint8) * 180
        # Add texture
        noise = np.random.normal(0, 10, (size[1], size[0], 3)).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return img


def add_scratch(img, severity='medium'):
    """Add scratch defect to image."""
    h, w = img.shape[:2]
    
    # Random scratch parameters
    start_point = (np.random.randint(50, w-50), np.random.randint(50, h-50))
    length = np.random.randint(30, 100)
    angle = np.random.uniform(0, 2*np.pi)
    
    end_point = (
        int(start_point[0] + length * np.cos(angle)),
        int(start_point[1] + length * np.sin(angle))
    )
    
    thickness = {'low': 1, 'medium': 2, 'high': 3}.get(severity, 2)
    color = (200, 200, 210) if severity == 'low' else (180, 180, 190)
    
    cv2.line(img, start_point, end_point, color, thickness)
    
    # Add some irregularities
    for _ in range(np.random.randint(2, 5)):
        offset = np.random.randint(-5, 5)
        mid_point = ((start_point[0] + end_point[0])//2 + offset,
                     (start_point[1] + end_point[1])//2 + offset)
        cv2.line(img, start_point, mid_point, color, thickness)
    
    return img


def add_crack(img, severity='medium'):
    """Add crack defect to image."""
    h, w = img.shape[:2]
    
    # Create jagged crack line
    num_points = np.random.randint(5, 10)
    points = []
    
    start_x = np.random.randint(50, w-100)
    start_y = np.random.randint(50, h//2)
    
    for i in range(num_points):
        x = start_x + i * 15 + np.random.randint(-10, 10)
        y = start_y + i * 20 + np.random.randint(-15, 15)
        points.append((x, y))
    
    # Draw crack
    for i in range(len(points) - 1):
        thickness = np.random.randint(1, 3)
        color = (50, 50, 50) if severity == 'high' else (80, 80, 80)
        cv2.line(img, points[i], points[i+1], color, thickness)
    
    # Add small branching cracks
    for _ in range(np.random.randint(1, 4)):
        idx = np.random.randint(1, len(points)-1)
        branch_end = (points[idx][0] + np.random.randint(-30, 30),
                      points[idx][1] + np.random.randint(-30, 30))
        cv2.line(img, points[idx], branch_end, (70, 70, 70), 1)
    
    return img


def add_dent(img, severity='medium'):
    """Add dent defect to image."""
    h, w = img.shape[:2]
    
    center = (np.random.randint(100, w-100), np.random.randint(100, h-100))
    radius = {'low': 15, 'medium': 25, 'high': 35}.get(severity, 25)
    
    # Create dent with gradient
    for r in range(radius, 0, -2):
        intensity = 200 - int((radius - r) * 3)
        color = (intensity, intensity, intensity)
        cv2.circle(img, center, r, color, 2)
    
    # Add shadow effect
    shadow_offset = 3
    shadow_center = (center[0] + shadow_offset, center[1] + shadow_offset)
    cv2.circle(img, shadow_center, radius, (150, 150, 150), 2)
    
    return img


def add_contamination(img, severity='medium'):
    """Add contamination/dust particles."""
    h, w = img.shape[:2]
    
    num_particles = {'low': 3, 'medium': 8, 'high': 15}.get(severity, 8)
    
    for _ in range(num_particles):
        center = (np.random.randint(50, w-50), np.random.randint(50, h-50))
        radius = np.random.randint(2, 8)
        color_var = np.random.randint(-20, 20)
        color = (150+color_var, 140+color_var, 130+color_var)
        cv2.circle(img, center, radius, color, -1)
    
    return img


def add_discoloration(img, severity='medium'):
    """Add discoloration patch."""
    h, w = img.shape[:2]
    
    # Create irregular patch
    num_points = np.random.randint(6, 12)
    points = []
    
    center_x = np.random.randint(150, w-150)
    center_y = np.random.randint(150, h-150)
    base_radius = {'low': 30, 'medium': 50, 'high': 70}.get(severity, 50)
    
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        radius = base_radius + np.random.randint(-15, 15)
        x = int(center_x + radius * np.cos(angle))
        y = int(center_y + radius * np.sin(angle))
        points.append([x, y])
    
    points = np.array(points, dtype=np.int32)
    
    # Color based on severity
    if severity == 'low':
        color = (220, 210, 200)
    elif severity == 'medium':
        color = (200, 180, 160)
    else:
        color = (180, 150, 130)
    
    cv2.fillPoly(img, [points], color)
    
    return img


def generate_defect_image(product_type='pcb', defect_type='good', severity='medium', size=(640, 480)):
    """Generate a single defect image."""
    img = create_base_product(size, product_type)
    
    if defect_type == 'good':
        return img, 'Good Product'
    
    # Add specific defect
    if defect_type == 'scratch':
        img = add_scratch(img, severity)
        return img, 'Scratch'
    
    elif defect_type == 'crack':
        img = add_crack(img, severity)
        return img, 'Crack'
    
    elif defect_type == 'dent':
        img = add_dent(img, severity)
        return img, 'Dent'
    
    elif defect_type == 'contamination':
        img = add_contamination(img, severity)
        return img, 'Contamination'
    
    elif defect_type == 'discoloration':
        img = add_discoloration(img, severity)
        return img, 'Discoloration'
    
    return img, 'Good Product'


def generate_dataset(output_dir='sample_data', num_images=50):
    """Generate complete sample dataset."""
    print("=" * 60)
    print("🎨 Generating Sample Manufacturing Data...")
    print("=" * 60)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    defect_types = ['good', 'scratch', 'crack', 'dent', 'contamination', 'discoloration']
    severities = ['low', 'medium', 'high']
    product_types = ['pcb', 'metal']
    
    generated = []
    
    for i in range(num_images):
        product_type = np.random.choice(product_types)
        defect_type = np.random.choice(defect_types, p=[0.3, 0.15, 0.15, 0.15, 0.15, 0.1])
        severity = np.random.choice(severities)
        
        img, label = generate_defect_image(product_type, defect_type, severity)
        
        # Add filename with metadata
        filename = f"{output_dir}/{product_type}_{defect_type}_{severity}_{i:04d}.jpg"
        cv2.imwrite(filename, img)
        
        generated.append({
            'filename': filename,
            'product_type': product_type,
            'defect_type': defect_type,
            'severity': severity,
            'label': label
        })
        
        if (i + 1) % 10 == 0:
            print(f"✅ Generated {i+1}/{num_images} images...")
    
    # Create summary
    print("\n📊 Generation Summary:")
    print("-" * 40)
    
    defect_counts = {}
    for g in generated:
        defect_counts[g['defect_type']] = defect_counts.get(g['defect_type'], 0) + 1
    
    for defect_type, count in sorted(defect_counts.items()):
        print(f"  {defect_type:20} : {count:3d} images")
    
    print("\n" + "=" * 60)
    print(f"✅ Dataset generated: {output_dir}/")
    print(f"   Total images: {num_images}")
    print("=" * 60)
    
    return generated


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate sample defect images')
    parser.add_argument('--output', type=str, default='sample_data',
                        help='Output directory')
    parser.add_argument('--count', type=int, default=50,
                        help='Number of images to generate')
    
    args = parser.parse_args()
    
    generate_dataset(args.output, args.count)
    
    print("\n🚀 Ready to test DefectX AI!")
    print(f"   streamlit run app.py")
    print(f"\nThen upload images from: {args.output}/")
