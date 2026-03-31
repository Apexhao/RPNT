"""
Plotting Utilities for Neural Connectivity Visualization

This module provides comprehensive plotting functions for visualizing
functional connectivity patterns from neural foundation model analysis.

Key Features:
- 16x16 connectivity heatmaps with anatomical context
- Site coordinate visualization with connectivity overlays
- Temporal connectivity dynamics plots
- Method comparison visualizations
- Publication-ready figure generation
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class ConnectivityPlotter:
    """
    Comprehensive plotting utilities for connectivity analysis visualization.
    
    **PLOT TYPES**:
    1. Connectivity heatmaps (16x16 matrices)
    2. Anatomical site layouts with connectivity overlays
    3. Temporal connectivity dynamics
    4. Method comparison plots
    5. Statistical summary visualizations
    
    **KEY FEATURES**:
    - Automatic diagonal removal for functional connectivity
    - Proper normalization to [0,1] range excluding diagonal
    - Publication-ready visualization standards
    """
    
    def __init__(self, 
                 figsize_default: Tuple[int, int] = (12, 10),
                 dpi: int = 300,
                 cmap_connectivity: str = 'viridis',
                 cmap_difference: str = 'RdBu_r'):
        """
        Initialize ConnectivityPlotter.
        
        Args:
            figsize_default: Default figure size
            dpi: DPI for high-quality figures
            cmap_connectivity: Colormap for connectivity matrices
            cmap_difference: Colormap for difference/comparison plots
        """
        
        self.figsize_default = figsize_default
        self.dpi = dpi
        self.cmap_connectivity = cmap_connectivity
        self.cmap_difference = cmap_difference
        self.logger = logging.getLogger(__name__)
        
        # Setup matplotlib parameters for publication quality
        plt.rcParams.update({
            'figure.dpi': dpi,
            'savefig.dpi': dpi,
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 11,
            'figure.titlesize': 16
        })
        
        self.logger.info("ConnectivityPlotter initialized")
    
    def generate_anatomical_site_labels(self, site_coordinates: torch.Tensor, site_ids: List[str]) -> List[str]:
        """
        Generate simple sequential site labels (S1, S2, ...) in original order.
        
        Args:
            site_coordinates: [S, 2] - (X, Y) coordinates (not used, kept for compatibility)
            site_ids: Original site identifiers
            
        Returns:
            List of site labels in simple sequential order S1, S2, S3, ...
        """
        
        # Generate simple S1, S2, S3, ... labels in original order
        sequential_labels = [f"S{i+1}" for i in range(len(site_ids))]
        
        return sequential_labels
    
    def remove_diagonal_and_normalize(self, connectivity_matrix: torch.Tensor) -> torch.Tensor:
        """
        Remove diagonal values and normalize to [0,1] range.
        
        Args:
            connectivity_matrix: [S, S] - connectivity matrix
            
        Returns:
            normalized_matrix: [S, S] - matrix with diagonal=0 and off-diagonal normalized to [0,1]
        """
        
        # Convert to numpy if needed
        if isinstance(connectivity_matrix, torch.Tensor):
            matrix = connectivity_matrix.detach().cpu().numpy().copy()
        else:
            matrix = connectivity_matrix.copy()
        
        # Set diagonal to 0
        np.fill_diagonal(matrix, 0)
        
        # Get off-diagonal values for normalization
        mask = ~np.eye(matrix.shape[0], dtype=bool)
        off_diagonal_values = matrix[mask]
        
        if len(off_diagonal_values) > 0 and off_diagonal_values.max() > off_diagonal_values.min():
            # Normalize off-diagonal values to [0, 1]
            min_val = off_diagonal_values.min()
            max_val = off_diagonal_values.max()
            
            # Apply normalization only to off-diagonal elements
            matrix[mask] = (off_diagonal_values - min_val) / (max_val - min_val)
        
        return matrix
    
    def prepare_connectivity_for_plotting(self, connectivity_matrix: torch.Tensor, 
                                        remove_diagonal: bool = True) -> Tuple[np.ndarray, float, float]:
        """
        Prepare connectivity matrix for plotting with proper normalization.
        
        Args:
            connectivity_matrix: [S, S] - connectivity matrix
            remove_diagonal: Whether to remove diagonal and normalize
            
        Returns:
            Tuple of (processed_matrix, vmin, vmax)
        """
        
        if remove_diagonal:
            matrix = self.remove_diagonal_and_normalize(connectivity_matrix)
            # After normalization, range is [0, 1] for off-diagonal
            vmin, vmax = 0.0, 1.0
        else:
            # Keep original matrix
            if isinstance(connectivity_matrix, torch.Tensor):
                matrix = connectivity_matrix.detach().cpu().numpy()
            else:
                matrix = connectivity_matrix
            vmin, vmax = matrix.min(), matrix.max()
        
        return matrix, vmin, vmax
    
    def plot_connectivity_heatmap(self, 
                                 connectivity_matrix: torch.Tensor,
                                 title: str = "Connectivity Matrix",
                                 site_ids: Optional[List[str]] = None,
                                 figsize: Optional[Tuple[int, int]] = None,
                                 save_path: Optional[str] = None,
                                 vmin: Optional[float] = None,
                                 vmax: Optional[float] = None,
                                 annotate: bool = False,  # Default to no annotation for cleaner plots
                                 remove_diagonal: bool = True,
                                 use_anatomical_labels: bool = True,
                                 site_coordinates: Optional[torch.Tensor] = None) -> plt.Figure:
        """
        Plot connectivity matrix as heatmap with diagonal removal and normalization.
        
        Args:
            connectivity_matrix: [S, S] - connectivity matrix
            title: Plot title
            site_ids: List of site identifiers for labels
            figsize: Figure size (uses default if None)
            save_path: Path to save figure
            vmin, vmax: Color scale limits (ignored if remove_diagonal=True)
            annotate: Whether to annotate cells with values
            remove_diagonal: Whether to remove diagonal and normalize to [0,1]
            use_anatomical_labels: Whether to use S1, S2, ... labels based on coordinates
            site_coordinates: [S, 2] - coordinates for anatomical labeling
            
        Returns:
            matplotlib Figure object
        """
        
        if figsize is None:
            figsize = self.figsize_default
        
        # Prepare matrix for plotting with proper normalization
        conn_matrix, plot_vmin, plot_vmax = self.prepare_connectivity_for_plotting(
            connectivity_matrix, remove_diagonal=remove_diagonal
        )
        
        # Use computed vmin/vmax if not specified or if using diagonal removal
        if remove_diagonal or (vmin is None and vmax is None):
            vmin, vmax = plot_vmin, plot_vmax
        
        S = conn_matrix.shape[0]
        
        # Create site labels
        if use_anatomical_labels and site_coordinates is not None and site_ids is not None:
            site_labels = self.generate_anatomical_site_labels(site_coordinates, site_ids)
        elif site_ids is None:
            site_labels = [f"Site {i}" for i in range(S)]
        else:
            site_labels = site_ids
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        im = ax.imshow(conn_matrix, 
                      cmap=self.cmap_connectivity,
                      vmin=vmin, vmax=vmax,
                      aspect='equal')
        
        # Customize axes
        ax.grid(False)
        ax.set_xticks(range(S))
        ax.set_yticks(range(S))
        ax.set_xticklabels(site_labels, rotation=45, ha='right')
        ax.set_yticklabels(site_labels)
        
        ax.set_xlabel('Target Site (j)')
        ax.set_ylabel('Source Site (i)')
        ax.set_title(title, fontsize=16, pad=20)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Connectivity Strength', rotation=270, labelpad=20)
        
        # Annotate cells if requested
        if annotate and S <= 20:  # Only annotate for reasonable matrix sizes
            for i in range(S):
                for j in range(S):
                    value = conn_matrix[i, j]
                    color = 'white' if value > (conn_matrix.max() * 0.6) else 'black'
                    ax.text(j, i, f'{value:.3f}', 
                           ha='center', va='center', color=color, fontsize=8)
        
        # Clean heatmap without grid lines
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.savefig(save_path.replace('.png', '.svg'), format='svg', dpi=300, bbox_inches='tight')
            self.logger.info(f"Figure saved to {save_path}")
        
        return fig
    
    def plot_temporal_connectivity_dynamics(self, 
                                          temporal_connectivity: torch.Tensor,
                                          timepoints: List[int],
                                          title: str = "Temporal Connectivity Dynamics",
                                          site_ids: Optional[List[str]] = None,
                                          figsize: Optional[Tuple[int, int]] = None,
                                          save_path: Optional[str] = None,
                                          remove_diagonal: bool = True,
                                          use_anatomical_labels: bool = True,
                                          site_coordinates: Optional[torch.Tensor] = None,
                                          show_colorbar: bool = False) -> plt.Figure:
        """
        Plot how connectivity changes across time with diagonal removal.
        
        Args:
            temporal_connectivity: [T, S, S] - connectivity at different timepoints
            timepoints: List of timepoint labels
            title: Plot title
            site_ids: Site identifiers
            figsize: Figure size
            save_path: Path to save figure
            remove_diagonal: Whether to remove diagonal and normalize each timepoint
            use_anatomical_labels: Whether to use S1, S2, ... labels
            site_coordinates: [S, 2] - coordinates for anatomical labeling
            show_colorbar: Whether to show colorbar
            
        Returns:
            matplotlib Figure object
        """
        
        if figsize is None:
            figsize = (14, 5)  # Slightly smaller for closer figures
        
        # Convert to numpy and process each timepoint
        if isinstance(temporal_connectivity, torch.Tensor):
            temp_conn_raw = temporal_connectivity.detach().cpu().numpy()
        else:
            temp_conn_raw = temporal_connectivity
        
        T, S, S = temp_conn_raw.shape
        
        # Process each timepoint with diagonal removal if requested
        if remove_diagonal:
            temp_conn = np.zeros_like(temp_conn_raw)
            for t in range(T):
                temp_conn[t], _, _ = self.prepare_connectivity_for_plotting(
                    torch.tensor(temp_conn_raw[t]), remove_diagonal=True
                )
        else:
            temp_conn = temp_conn_raw
        
        # Create site labels
        if use_anatomical_labels and site_coordinates is not None and site_ids is not None:
            site_labels = self.generate_anatomical_site_labels(site_coordinates, site_ids)
        elif site_ids is None:
            site_labels = [f"Site {i}" for i in range(S)]
        else:
            site_labels = site_ids
        
        # Create subplot for each timepoint with tighter spacing
        fig, axes = plt.subplots(1, T, figsize=figsize)
        if T == 1:
            axes = [axes]
        
        # Adjust spacing between subplots
        plt.subplots_adjust(wspace=0.1)
        
        # Determine global color scale based on processing method
        if remove_diagonal:
            # All timepoints are normalized to [0, 1] range
            vmin, vmax = 0.0, 1.0
        else:
            # Use global scale across all timepoints
            vmin = temp_conn.min()
            vmax = temp_conn.max()
        
        for t, ax in enumerate(axes):
            # Plot connectivity at this timepoint
            im = ax.imshow(temp_conn[t], 
                          cmap=self.cmap_connectivity,
                          vmin=vmin, vmax=vmax,
                          aspect='equal')
            
            # Customize axis
            ax.grid(False)
            ax.set_title(f't = {timepoints[t]}', fontsize=14)
            ax.set_xticks(range(S))
            ax.set_yticks(range(S))
            
            if t == 0:  # Only label y-axis for first subplot
                ax.set_yticklabels(site_labels)
                ax.set_ylabel('Source Site (i)')
            else:
                ax.set_yticklabels([])
            
            ax.set_xticklabels(site_labels, rotation=45, ha='right')
            ax.set_xlabel('Target Site (j)')
            
            # Clean temporal heatmap without grid lines
        
        # Add shared colorbar only if requested
        if show_colorbar:
            fig.subplots_adjust(right=0.9)
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label('Connectivity Strength', rotation=270, labelpad=20)
        
        fig.suptitle(title, fontsize=16, y=0.95)
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.savefig(save_path.replace('.png', '.svg'), format='svg', dpi=300, bbox_inches='tight')
            self.logger.info(f"Figure saved to {save_path}")
        
        return fig
    
    def plot_anatomical_connectivity_layout(self, 
                                          connectivity_matrix: torch.Tensor,
                                          site_coordinates: torch.Tensor,
                                          title: str = "Anatomical Connectivity Layout",
                                          threshold: float = 0.5,
                                          figsize: Optional[Tuple[int, int]] = None,
                                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot connectivity overlaid on anatomical site layout.
        
        Args:
            connectivity_matrix: [S, S] - connectivity matrix
            site_coordinates: [S, 2] - (X, Y) coordinates of sites
            title: Plot title
            threshold: Minimum connectivity strength to display connections
            figsize: Figure size
            save_path: Path to save figure
            
        Returns:
            matplotlib Figure object
        """
        
        if figsize is None:
            figsize = (12, 10)
        
        # Convert to numpy
        if isinstance(connectivity_matrix, torch.Tensor):
            conn_matrix = connectivity_matrix.detach().cpu().numpy()
        else:
            conn_matrix = connectivity_matrix
        
        if isinstance(site_coordinates, torch.Tensor):
            coords = site_coordinates.detach().cpu().numpy()
        else:
            coords = site_coordinates
        
        S = conn_matrix.shape[0]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot site locations
        ax.scatter(coords[:, 0], coords[:, 1], 
                  s=200, c='lightblue', edgecolors='black', 
                  linewidth=2, alpha=0.8, zorder=3)
        
        # Add site labels
        for i in range(S):
            ax.annotate(f'S{i+1}', (coords[i, 0], coords[i, 1]), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=10, fontweight='bold', zorder=4)
        
        # Plot connections above threshold (only if threshold <= 0.5 for cleaner view)
        connection_count = 0
        for i in range(S):
            for j in range(S):
                if i != j and conn_matrix[i, j] > threshold:
                    # Draw arrow from site i to site j
                    start = coords[i]
                    end = coords[j]
                    
                    # Arrow properties based on connection strength
                    strength = conn_matrix[i, j]
                    linewidth = 0.25 + 4 * strength  # Scale linewidth with strength
                    alpha = 0.25 + 0.75 * strength  # Scale alpha with strength
                    
                    # Draw arrow
                    ax.annotate('', xy=end, xytext=start,
                                arrowprops=dict(arrowstyle='->', 
                                            linewidth=linewidth,
                                            alpha=alpha,
                                            color='grey'),
                                zorder=1)
                    
                    connection_count += 1
        
        # Customize plot
        ax.grid(False)
        ax.set_xlabel('X Coordinate (mm)', fontsize=20)
        ax.set_ylabel('Y Coordinate (mm)', fontsize=20)
        ax.set_title(f'{title}\n({connection_count} connections > {threshold:.2f})', 
                    fontsize=20, pad=20)
        
        # Set equal aspect ratio for accurate anatomical representation
        ax.set_aspect('equal')
        
        ax.tick_params(axis='both', which='major', labelsize=20)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], color='red', linewidth=3, alpha=0.7, 
                      label=f'Strong Connection (> {threshold:.2f})'),
            plt.scatter([], [], s=200, c='lightblue', edgecolors='black', 
                       label='Recording Site')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.savefig(save_path.replace('.png', '.svg'), format='svg', dpi=300, bbox_inches='tight')
            self.logger.info(f"Figure saved to {save_path}")
        
        return fig
    
    def plot_method_comparison(self, 
                              attention_connectivity: torch.Tensor,
                              noise_connectivity: torch.Tensor,
                              correlation_connectivity: torch.Tensor,
                              site_ids: Optional[List[str]] = None,
                              figsize: Optional[Tuple[int, int]] = None,
                              save_path: Optional[str] = None,
                              remove_diagonal: bool = True,
                              use_anatomical_labels: bool = True,
                              site_coordinates: Optional[torch.Tensor] = None) -> plt.Figure:
        """
        Compare connectivity matrices from different methods with diagonal removal.
        
        Args:
            attention_connectivity: [S, S] - attention-based connectivity
            noise_connectivity: [S, S] - noise replacement connectivity
            correlation_connectivity: [S, S] - correlation-based connectivity
            site_ids: Site identifiers
            figsize: Figure size
            save_path: Path to save figure
            remove_diagonal: Whether to remove diagonal and normalize each method
            use_anatomical_labels: Whether to use S1, S2, ... labels
            site_coordinates: [S, 2] - coordinates for anatomical labeling
            
        Returns:
            matplotlib Figure object
        """
        
        figsize = (24, 6)
        
        # Prepare matrices with consistent processing
        methods = {
            'Attention-Based': attention_connectivity,
            'Noise Replacement': noise_connectivity,
            'Correlation Baseline': correlation_connectivity
        }
        
        matrices = {}
        for name, matrix in methods.items():
            if remove_diagonal:
                # Apply diagonal removal and normalization to each method
                processed_matrix, _, _ = self.prepare_connectivity_for_plotting(matrix, remove_diagonal=True)
                matrices[name] = processed_matrix
            else:
                # Keep original matrices
                if isinstance(matrix, torch.Tensor):
                    matrices[name] = matrix.detach().cpu().numpy()
                else:
                    matrices[name] = matrix
        
        S = list(matrices.values())[0].shape[0]
        
        # Create site labels
        if use_anatomical_labels and site_coordinates is not None and site_ids is not None:
            site_labels = self.generate_anatomical_site_labels(site_coordinates, site_ids)
        elif site_ids is None:
            site_labels = [f"S{i}" for i in range(S)]
        else:
            site_labels = site_ids
        
        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Determine color scale based on processing method
        if remove_diagonal:
            # All matrices are normalized to [0, 1] range
            vmin, vmax = 0.0, 1.0
        else:
            # Use global scale across all methods
            all_values = np.concatenate([matrix.flatten() for matrix in matrices.values()])
            vmin, vmax = np.percentile(all_values, [5, 95])  # Use 5-95 percentile for robustness
        
        for idx, (method_name, matrix) in enumerate(matrices.items()):
            ax = axes[idx]
            
            # Plot heatmap
            im = ax.imshow(matrix, 
                          cmap=self.cmap_connectivity,
                          vmin=vmin, vmax=vmax,
                          aspect='equal')
            
            # Customize axis
            ax.set_title(method_name, fontsize=14, pad=15)
            ax.set_xticks(range(S))
            ax.set_yticks(range(S))
            ax.set_xticklabels(site_labels, rotation=45, ha='right')
            ax.grid(False)
            if idx == 0:  # Only label y-axis for first subplot
                ax.set_yticklabels(site_labels)
                ax.set_ylabel('Source Site (i)', fontsize=12)
            else:
                ax.set_yticklabels([])
            
            ax.set_xlabel('Target Site (j)', fontsize=12)
            
            # Clean method comparison heatmap without grid lines
        
        # Add shared colorbar
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Connectivity Strength', rotation=270, labelpad=20)
        
        fig.suptitle('Connectivity Method Comparison', fontsize=16, y=0.95)
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.savefig(save_path.replace('.png', '.svg'), format='svg', dpi=300, bbox_inches='tight')
            self.logger.info(f"Figure saved to {save_path}")
        
        return fig
    
    def plot_connectivity_correlation_analysis(self, 
                                             attention_connectivity: torch.Tensor,
                                             noise_connectivity: torch.Tensor,
                                             figsize: Optional[Tuple[int, int]] = None,
                                             save_path: Optional[str] = None,
                                             remove_diagonal: bool = True) -> plt.Figure:
        """
        Plot correlation between attention and noise replacement methods with diagonal removal.
        
        Args:
            attention_connectivity: [S, S] - attention connectivity
            noise_connectivity: [S, S] - noise replacement connectivity
            figsize: Figure size
            save_path: Path to save figure
            remove_diagonal: Whether to remove diagonal and normalize before correlation
            
        Returns:
            matplotlib Figure object
        """
        
        if figsize is None:
            figsize = (10, 8)
        
        # Process connectivity matrices consistently
        if remove_diagonal:
            attn_processed, _, _ = self.prepare_connectivity_for_plotting(
                attention_connectivity, remove_diagonal=True
            )
            noise_processed, _, _ = self.prepare_connectivity_for_plotting(
                noise_connectivity, remove_diagonal=True
            )
        else:
            # Convert to numpy without processing
            if isinstance(attention_connectivity, torch.Tensor):
                attn_processed = attention_connectivity.detach().cpu().numpy()
            else:
                attn_processed = attention_connectivity
            
            if isinstance(noise_connectivity, torch.Tensor):
                noise_processed = noise_connectivity.detach().cpu().numpy()
            else:
                noise_processed = noise_connectivity
        
        S = attn_processed.shape[0]
        
        # Get off-diagonal elements (exclude self-connections)
        mask = ~np.eye(S, dtype=bool)
        attn_values = attn_processed[mask]
        noise_values = noise_processed[mask]
        
        # Compute correlation
        correlation = np.corrcoef(attn_values, noise_values)[0, 1]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Scatter plot
        ax.scatter(attn_values, noise_values, 
                  alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        # Add best fit line
        z = np.polyfit(attn_values, noise_values, 1)
        p = np.poly1d(z)
        x_line = np.linspace(attn_values.min(), attn_values.max(), 100)
        ax.plot(x_line, p(x_line), 'r-', linewidth=2, alpha=0.8)
        
        # Customize plot
        ax.set_xlabel('Attention-Based Connectivity', fontsize=12)
        ax.set_ylabel('Noise Replacement Connectivity', fontsize=12)
        ax.set_title(f'Method Correlation Analysis\n(r = {correlation:.3f})', 
                    fontsize=14, pad=20)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add correlation text
        ax.text(0.05, 0.95, f'Pearson r = {correlation:.3f}\nN = {len(attn_values)} connections',
               transform=ax.transAxes, fontsize=12,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Figure saved to {save_path}")
        
        return fig


def demo_plotting():
    """Demo function for plotting utilities."""
    
    print("📊 Connectivity Plotting Demo")
    print("=" * 40)
    
    # Create synthetic data for demo
    S = 16  # 16 sites
    
    # Generate synthetic connectivity matrices
    attention_conn = torch.rand(S, S) * 0.8 + 0.1
    attention_conn = attention_conn + attention_conn.T  # Make symmetric
    attention_conn.fill_diagonal_(0)  # No self-connections
    
    noise_conn = torch.rand(S, S) * 0.6 + 0.2
    noise_conn.fill_diagonal_(0)
    
    correlation_conn = torch.rand(S, S) * 0.4 + 0.3
    correlation_conn = correlation_conn + correlation_conn.T
    correlation_conn.fill_diagonal_(1)  # Perfect self-correlation
    
    # Generate synthetic coordinates
    coords = torch.randn(S, 2) * 5
    
    # Initialize plotter
    plotter = ConnectivityPlotter()
    
    try:
        # Demo heatmap
        fig1 = plotter.plot_connectivity_heatmap(
            attention_conn, 
            title="Demo Attention Connectivity",
            save_path="demo_heatmap.png"
        )
        
        # Demo method comparison
        fig2 = plotter.plot_method_comparison(
            attention_conn, noise_conn, correlation_conn,
            save_path="demo_comparison.png"
        )
        
        print("✅ Demo plots generated successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")


if __name__ == "__main__":
    demo_plotting()
