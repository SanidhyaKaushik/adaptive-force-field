import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader , TensorDataset
import numpy as np

def plot_confidence_heatmap(evaluator, resolution=100, range_lim=3.0, z_slice=0.0, save_path=None):
    """
    Plots a 2D spatial confidence heatmap using a pre-trained and calibrated evaluator.
    
    Args:
        evaluator: An instance of ForceFieldEvaluator (already trained and calibrated).
        resolution: Number of points along each axis (N x N).
        range_lim: The spatial extent to plot (from -range_lim to +range_lim).
        z_slice: The Z-coordinate at which to take the 2D slice.
        save_path: If provided, saves the plot to this file path.
    """
    
    # 1. Generate the 2D Grid
    x_range = np.linspace(-range_lim, range_lim, resolution)
    y_range = np.linspace(-range_lim, range_lim, resolution)
    xx, yy = np.meshgrid(x_range, y_range)
    
    # Create the (N*N, 3) input array for the evaluator
    zz = np.full_like(xx.ravel(), z_slice)
    grid_points = np.stack([xx.ravel(), yy.ravel(), zz], axis=1)

    # 2. Query the Evaluator
    # We use get_force_field which we've upgraded to handle batches
    _, confidences, fallback_flags = evaluator.get_force_field(grid_points)
    
    # Reshape results back to 2D
    conf_grid = confidences.reshape(resolution, resolution)
    fallback_grid = fallback_flags.reshape(resolution, resolution)

    # 3. Plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the confidence as a heatmap
    # cmap='RdYlGn' maps low confidence to red and high confidence to green
    im = ax.imshow(conf_grid, extent=[-range_lim, range_lim, -range_lim, range_lim],
                   origin='lower', cmap='RdYlGn', alpha=0.9, vmin=0, vmax=1)
    
    # Add a contour line exactly at the alpha threshold
    # This shows the "Decision Boundary" where the system switches to Brute Force
    ax.contour(xx, yy, conf_grid, levels=[evaluator.alpha], colors='black', 
               linestyles='dashed', linewidths=1.5)

    # 4. Overlay Charge Positions
    # charges and positions are pulled directly from the evaluator object
    for i, pos in enumerate(evaluator.positions):
        # Only plot charges that are near our Z-slice for visual clarity
        if abs(pos[2] - z_slice) < 0.5:
            color = 'red' if evaluator.charges[i] > 0 else 'blue'
            ax.scatter(pos[0], pos[1], marker='o', s=150, color=color, 
                       edgecolors='white', linewidth=2, zorder=5)
            
            label = f"{'+' if evaluator.charges[i] > 0 else ''}{evaluator.charges[i]}q"
            ax.text(pos[0]+0.1, pos[1]+0.1, label, fontsize=12, 
                    fontweight='bold', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

    # 5. Formatting
    cbar = plt.colorbar(im)
    cbar.set_label('Conformal Confidence Score', rotation=270, labelpad=15)
    
    ax.set_title(f"Spatial Confidence Heatmap (Z={z_slice})\n"
                 f"Alpha: {evaluator.alpha} | Max Error: {evaluator.max_error}", fontsize=14)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    
    # Add a custom legend entry for the fallback boundary
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='black', lw=1.5, linestyle='--')]
    ax.legend(custom_lines, [f'Fallback Boundary (α={evaluator.alpha})'], loc='upper right')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Heatmap saved to {save_path}")
    
    plt.show()