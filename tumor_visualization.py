import os
import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
from scipy import ndimage
import glob

def voxel2R(A):
    """Convert voxel volume to sphere radius (unit: mm)"""
    return (np.array(A)/4*3/np.pi)**(1/3)

def pixel2voxel(A, res):
    """Convert pixel count to volume (unit: mm³)"""
    return np.array(A)*(res[0]*res[1]*res[2])

def get_liver_bbox(mask):
    """Get bounding box of liver region"""
    x_start, x_end = np.where(np.any(mask, axis=(1, 2)))[0][[0, -1]]
    y_start, y_end = np.where(np.any(mask, axis=(0, 2)))[0][[0, -1]]
    z_start, z_end = np.where(np.any(mask, axis=(0, 1)))[0][[0, -1]]
    
    # Add margin
    margin = 5
    x_start, x_end = max(0, x_start - margin), min(mask.shape[0], x_end + margin)
    y_start, y_end = max(0, y_start - margin), min(mask.shape[1], y_end + margin)
    z_start, z_end = max(0, z_start - margin), min(mask.shape[2], z_end + margin)
    
    return (x_start, x_end), (y_start, y_end), (z_start, z_end)

def find_optimal_slice(tumor_mask, liver_mask):
    """Find the slice containing the largest tumor area"""
    # Ensure input is boolean type
    tumor_mask = tumor_mask.astype(bool)
    liver_mask = liver_mask.astype(bool)
    
    # Find slices containing tumor
    tumor_slices = {}
    for z in range(tumor_mask.shape[2]):
        slice_area = np.sum(tumor_mask[:, :, z])
        if slice_area > 0:
            tumor_slices[z] = slice_area
    
    # If no tumor is found, find the slice with most liver tissue
    if not tumor_slices:
        liver_slices = {}
        for z in range(liver_mask.shape[2]):
            slice_area = np.sum(liver_mask[:, :, z])
            if slice_area > 0:
                liver_slices[z] = slice_area
        if not liver_slices:
            return None
        return max(liver_slices, key=liver_slices.get)
    
    # Return the slice with largest tumor area
    return max(tumor_slices, key=tumor_slices.get)

def analyze_tumors(tumor_mask, spacing_mm):
    """Analyze tumor positions and sizes"""
    tumor_info = []
    if np.sum(tumor_mask) > 0:
        label_cc, label_num = ndimage.label(tumor_mask)
        for i in range(1, label_num + 1):
            tumor_region = (label_cc == i)
            tumor_size = np.sum(tumor_region)
            if tumor_size < 8:  # Ignore too small tumors
                continue
                
            # Calculate tumor radius (mm)
            tumor_volume_mm = pixel2voxel(tumor_size, spacing_mm)
            tumor_radius_mm = voxel2R(tumor_volume_mm)
            
            # Find tumor center
            pos = ndimage.center_of_mass(tumor_region)
            
            # Find 2D bounding box
            positions = np.where(tumor_region)
            z_indices = positions[2]
            if len(z_indices) == 0:
                continue
                
            size_category = ""
            if tumor_radius_mm <= 5:
                size_category = "0-5mm"
            elif tumor_radius_mm <= 10:
                size_category = "5-10mm"
            else:
                size_category = ">10mm"
                
            tumor_info.append({
                "center": pos,
                "radius_mm": tumor_radius_mm,
                "volume_mm3": tumor_volume_mm,
                "size_category": size_category,
                "region": tumor_region
            })
    
    return tumor_info

def create_visualization(scan_path, label_path, model_dirs, output_path):
    """Create visualization of tumor segmentation"""
    # Load scan and label
    scan_nib = nib.load(scan_path)
    label_nib = nib.load(label_path)
    
    scan_data = scan_nib.get_fdata()
    label_data = label_nib.get_fdata()
    
    # Get pixel spacing
    pixdim = label_nib.header['pixdim']
    spacing_mm = tuple(pixdim[1:4])
    
    # Extract liver and tumor masks
    liver_mask = np.zeros_like(label_data)
    liver_mask[label_data == 1] = 1
    liver_mask[label_data == 2] = 1  # Tumor region is also part of the liver
    
    tumor_mask = np.zeros_like(label_data)
    tumor_mask[label_data == 2] = 1
    
    # Analyze tumor information
    tumor_info = analyze_tumors(tumor_mask, spacing_mm)
    
    # Get liver region bounding box
    x_range, y_range, z_range = get_liver_bbox(liver_mask)
    
    # Find optimal slice
    optimal_slice = find_optimal_slice(tumor_mask, liver_mask)
    if optimal_slice is None:
        print("Could not find a suitable slice!")
        return
    
    # Ensure slice is within liver range
    optimal_slice = max(z_range[0], min(optimal_slice, z_range[1]))
    
    # Load model predictions
    model_predictions = []
    model_names = ["Model trained on real tumors", "SynTumor", "MedCraft"]
    for i, model_dir in enumerate(model_dirs):
        # Find prediction files
        pred_files = glob.glob(os.path.join(model_dir, "*.nii.gz"))
        if not pred_files:
            print(f"No prediction files found in directory {model_dir}")
            continue
            
        # Use prediction file matching scan name
        scan_name = os.path.basename(scan_path).split('.')[0]
        matching_files = [f for f in pred_files if scan_name in os.path.basename(f)]
        
        if matching_files:
            pred_path = matching_files[0]
        else:
            pred_path = pred_files[0]  # If no match, use the first one
            
        pred_nib = nib.load(pred_path)
        pred_data = pred_nib.get_fdata()
        
        # If prediction is class labels, convert to liver and tumor masks
        if len(pred_data.shape) == 3 and pred_data.dtype != bool:
            pred_liver_mask = np.zeros_like(pred_data)
            pred_liver_mask[pred_data == 1] = 1
            pred_liver_mask[pred_data == 2] = 1
            
            pred_tumor_mask = np.zeros_like(pred_data)
            pred_tumor_mask[pred_data == 2] = 1
            
            model_predictions.append((pred_liver_mask, pred_tumor_mask))
        else:
            # If prediction is already in mask form
            model_predictions.append((pred_data, pred_data))
    
    # Create visualization
    num_models = len(model_predictions)
    fig, axes = plt.subplots(1, num_models + 2, figsize=(5 * (num_models + 2), 5))
    
    # 1. Original scan image
    scan_slice = scan_data[:, :, optimal_slice].T
    
    # Normalize scan values to 0-1 range for display
    scan_min, scan_max = -150, 250  # Common CT window settings
    scan_slice_norm = np.clip(scan_slice, scan_min, scan_max)
    scan_slice_norm = (scan_slice_norm - scan_min) / (scan_max - scan_min)
    
    axes[0].imshow(scan_slice_norm, cmap='gray')
    axes[0].set_title("CT Scan", fontsize=14)
    
    # Mark tumor position and size on scan image
    for tumor in tumor_info:
        center = tumor["center"]
        radius_mm = tumor["radius_mm"]
        z = int(center[2])
        
        # Only mark tumors close to current slice
        if abs(z - optimal_slice) <= 3:
            y, x = int(center[0]), int(center[1])
            size = int(radius_mm * 2 / min(spacing_mm[0], spacing_mm[1]))  # Convert mm to pixel size
            
            rect = Rectangle((x - size//2, y - size//2), size, size, 
                            linewidth=2, edgecolor='r', facecolor='none')
            axes[0].add_patch(rect)
            axes[0].text(x - size//2, y - size//2 - 5, 
                        f"{radius_mm:.1f}mm", color='yellow', fontsize=12, 
                        fontweight='bold', bbox=dict(facecolor='black', alpha=0.7))
    
    # 2. Ground truth overlaid on scan
    axes[1].imshow(scan_slice_norm, cmap='gray')
    
    # Create liver mask overlay (green semi-transparent)
    liver_slice = liver_mask[:, :, optimal_slice].T
    liver_overlay = np.zeros((*liver_slice.shape, 4))
    liver_overlay[liver_slice == 1] = [0, 1, 0, 0.3]  # Green, 30% transparency
    
    # Create tumor mask overlay (red semi-transparent)
    tumor_slice = tumor_mask[:, :, optimal_slice].T
    tumor_overlay = np.zeros((*tumor_slice.shape, 4))
    tumor_overlay[tumor_slice == 1] = [1, 0, 0, 0.5]  # Red, 50% transparency
    
    axes[1].imshow(liver_overlay)
    axes[1].imshow(tumor_overlay)
    axes[1].set_title("Ground Truth", fontsize=14)
    
    # 3+ Model prediction results
    for i, ((pred_liver, pred_tumor), model_name) in enumerate(zip(model_predictions, model_names)):
        axes[i+2].imshow(scan_slice_norm, cmap='gray')
        
        pred_liver_slice = pred_liver[:, :, optimal_slice].T
        pred_liver_overlay = np.zeros((*pred_liver_slice.shape, 4))
        pred_liver_overlay[pred_liver_slice == 1] = [0, 1, 0, 0.3]  # Green, 30% transparency
        
        pred_tumor_slice = pred_tumor[:, :, optimal_slice].T
        pred_tumor_overlay = np.zeros((*pred_tumor_slice.shape, 4))
        pred_tumor_overlay[pred_tumor_slice == 1] = [1, 0, 0, 0.5]  # Red, 50% transparency
        
        axes[i+2].imshow(pred_liver_overlay)
        axes[i+2].imshow(pred_tumor_overlay)
        axes[i+2].set_title(model_name, fontsize=14)
    
    # Hide axis ticks
    for ax in axes:
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Tumor Segmentation Visualization Tool')
    parser.add_argument('--scan_path', type=str, required=True, help='Path to scan image')
    parser.add_argument('--label_path', type=str, required=True, help='Path to ground truth label')
    parser.add_argument('--model_dirs', type=str, nargs='+', help='Directories of model prediction results')
    parser.add_argument('--output_path', type=str, default='visualization.png', help='Output image path')
    
    args = parser.parse_args()
    
    if args.model_dirs and len(args.model_dirs) > 3:
        print("Warning: Maximum 3 model directories supported, using only the first 3.")
        args.model_dirs = args.model_dirs[:3]
    
    create_visualization(args.scan_path, args.label_path, args.model_dirs or [], args.output_path)

if __name__ == "__main__":
    main()
