import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def main():
    # Setup paths
    image_path = 'data/Task01_BrainTumour/imagesTr/BRATS_001.nii.gz'
    label_path = 'data/Task01_BrainTumour/labelsTr/BRATS_001.nii.gz'
    output_dir = 'tools/outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    img_nii = nib.load(image_path)
    img_data = img_nii.get_fdata()
    
    lbl_nii = nib.load(label_path)
    lbl_data = lbl_nii.get_fdata()
    
    if img_data.ndim == 4:
        img_t1 = img_data[..., 1] # try index 1 for T1w
    else:
        img_t1 = img_data
        
    # Normalize to [0, 1] over the volume
    img_max = np.max(img_t1)
    if img_max > 0:
        img_t1 = img_t1 / img_max
        
    # Pick a good axial slice
    tumor_slices = np.sum(lbl_data > 0, axis=(0, 1))
    best_z = np.argmax(tumor_slices)
    if tumor_slices[best_z] == 0:
        best_z = img_t1.shape[2] // 2
        
    slice_img = img_t1[:, :, best_z]
    slice_lbl = lbl_data[:, :, best_z]
    
    mask = slice_img > 0.01
    x = slice_img[mask]
    
    # CRITICAL VIZ RULE: fixed bins and axis limits
    global_bins = np.linspace(0, 1, 512)
    
    # 1. Base Histogram
    plt.figure(figsize=(4, 4), dpi=300)
    counts, _, _ = plt.hist(x, bins=global_bins, color='gray', alpha=0.7)
    max_y = counts.max() * 1.1 # Extract max count for global Y limit
    
    plt.xlim(0, 1)
    plt.ylim(0, max_y)
    plt.title("Fig 1: Base T1w Intensity Histogram (Masked)")
    plt.xlabel("Normalized Intensity")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "01_base_histogram.png"), bbox_inches='tight')
    plt.close()
    
    # 2. K=8 Chunked Histogram
    quantiles = np.linspace(0, 100, 9)
    q_edges = np.percentile(x, quantiles)
    q_edges[0] = 0.0 
    q_edges[-1] = 1.0
    
    chunks = np.digitize(x, q_edges) - 1
    chunks = np.clip(chunks, 0, 7)
    
    plt.figure(figsize=(4, 4), dpi=300)
    cmap = plt.colormaps.get_cmap('tab10') if hasattr(plt, 'colormaps') else plt.cm.get_cmap('tab10', 8)
    
    chunk_data = [x[chunks == c] for c in range(8)]
    colors = [cmap(c) for c in range(8)]
    labels = [f'Chunk {c}' for c in range(8)]
    
    plt.hist(chunk_data, bins=global_bins, color=colors, label=labels, stacked=True)
    plt.xlim(0, 1)
    plt.ylim(0, max_y)
    plt.title("Fig 2: K=8 Quantile Chunked Histogram")
    plt.xlabel("Normalized Intensity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "02_chunked_histogram.png"), bbox_inches='tight')
    plt.close()
    
    # 3. Spatial Chunk Map
    spatial_chunks = np.digitize(slice_img, q_edges) - 1
    spatial_chunks = np.clip(spatial_chunks, 0, 7)
    spatial_chunks_viz = spatial_chunks.astype(float)
    spatial_chunks_viz[~mask] = np.nan 
    
    plt.figure(figsize=(8, 8), dpi=300)
    cmap_viz = plt.colormaps.get_cmap('tab10').copy() if hasattr(plt, 'colormaps') else plt.cm.get_cmap('tab10', 8).copy()
    cmap_viz.set_bad(color='black')
    
    plt.imshow(spatial_chunks_viz.T, cmap=cmap_viz, origin='lower')
    plt.title("Fig 3: Spatial Chunk Map (Background=Black)")
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, "03_spatial_chunks.png"), bbox_inches='tight')
    plt.close()
    
    # 4. Texture-Preserving Remapped Histogram
    np.random.seed(42)
    mu_c = np.random.uniform(0, 1, 8)
    alpha_c = np.random.uniform(0.5, 2.0, 8)
    
    y = np.zeros_like(x)
    for c in range(8):
        mask_c = chunks == c
        q_c_minus_1 = q_edges[c]
        y[mask_c] = mu_c[c] + alpha_c[c] * (x[mask_c] - q_c_minus_1)
        
    y = np.clip(y, 0, 1) # clamp to [0, 1]
        
    plt.figure(figsize=(4, 4), dpi=300)
    remapped_chunk_data = [y[chunks == c] for c in range(8)]
    plt.hist(remapped_chunk_data, bins=global_bins, color=colors, label=labels, stacked=True)
        
    plt.xlim(0, 1)
    plt.ylim(0, max_y)
    plt.title("Fig 4: Texture-Preserving Remapped Histogram (v18_6)")
    plt.xlabel("Remapped Intensity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "04_remapped_histogram.png"), bbox_inches='tight')
    plt.close()
    
    # 5. Base Synthetic Image
    y_img = slice_img.copy()
    for c in range(8):
        mask_c = (spatial_chunks == c) & mask
        q_c_minus_1 = q_edges[c]
        y_img[mask_c] = mu_c[c] + alpha_c[c] * (slice_img[mask_c] - q_c_minus_1)
        
    y_img = np.clip(y_img, 0, 1)
    y_img[~mask] = 0.0
    
    plt.figure(figsize=(8, 8), dpi=300)
    plt.imshow(y_img.T, cmap='gray', origin='lower')
    plt.title("Fig 5: 2D Synthetic Image (Base v18_6 Layer)")
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, "05_base_synthetic_image.png"), bbox_inches='tight')
    plt.close()
    
    # 6. The v19 Override: Stochastic Semantic Decoupling
    tumor_mask = slice_lbl > 0
    y_v19 = y_img.copy()
    
    tumor_pixels = slice_img[tumor_mask]
    if len(tumor_pixels) > 0:
        mu_path = np.random.uniform(0.7, 1.0)
        alpha_path = np.random.uniform(0.5, 2.0)
        x_bar_c = np.mean(tumor_pixels)
        
        y_v19[tumor_mask] = mu_path + alpha_path * (slice_img[tumor_mask] - x_bar_c)
        y_v19 = np.clip(y_v19, 0, 1)
        
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=300)
    axes[0].imshow(slice_img.T, cmap='gray', origin='lower', vmin=0, vmax=1)
    axes[0].set_title("(A) Raw T1w Slice")
    axes[0].axis('off')
    
    axes[1].imshow(tumor_mask.T, cmap='inferno', origin='lower')
    axes[1].set_title("(B) Tumor Mask (Labels)")
    axes[1].axis('off')
    
    axes[2].imshow(y_v19.T, cmap='gray', origin='lower', vmin=0, vmax=1)
    axes[2].set_title("(C) Final v19 Stochastic Semantic Decoupling")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "06_v19_semantic_decoupling.png"), bbox_inches='tight')
    plt.close()
    
    print("Successfully generated all strictly-binned v19 visual assets in tools/outputs/")

if __name__ == '__main__':
    main()
