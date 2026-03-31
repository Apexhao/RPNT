"""
Main Script for Neural Foundation Model Connectivity Analysis

This script performs comprehensive connectivity analysis using the pretrained
CrossSiteFoundationMAE model to reveal functional organization patterns in
neural recordings.

Analysis Pipeline:
1. Load pretrained model and test data
2. Extract attention-based connectivity matrices
3. Perform noise replacement information flow analysis
4. Compute correlation baseline for validation
5. Generate temporal connectivity dynamics
6. Create publication-ready visualizations

Usage:
    python -m src.visualization.run_analysis --checkpoint_path <path> --output_dir <dir>
"""

import argparse
import logging
import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Setup path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.visualization.connectivity_analysis import ConnectivityAnalyzer
from src.visualization.plotting_utils import ConnectivityPlotter
from src.utils.helpers import set_seed


def setup_logging(log_level: str = 'INFO'):
    """Setup logging for analysis."""
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('connectivity_analysis.log')
        ]
    )


def run_connectivity_analysis(checkpoint_path: str,
                             output_dir: str,
                             batch_size: int = 32,
                             temporal_timepoints: list = [0, 15, 30, 45]):
    """
    Run complete connectivity analysis pipeline.
    
    Args:
        checkpoint_path: Path to pretrained model checkpoint
        output_dir: Directory to save results and figures
        batch_size: Batch size for analysis
        temporal_timepoints: Timepoints for temporal analysis
    """
    
    logger = logging.getLogger(__name__)
    
    logger.info("🧠 Starting Neural Foundation Model Connectivity Analysis")
    logger.info("=" * 80)
    
    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    figures_path = output_path / 'figures'
    figures_path.mkdir(exist_ok=True)
    
    results_path = output_path / 'results'
    results_path.mkdir(exist_ok=True)
    
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    
    try:
        # 1. Initialize analyzer
        logger.info("\n📊 Phase 1: Initializing Connectivity Analyzer")
        analyzer = ConnectivityAnalyzer(
            checkpoint_path=checkpoint_path,
            temporal_timepoints=temporal_timepoints
        )
        
        # 2. Run connectivity analysis
        logger.info("\n🔬 Phase 2: Running Connectivity Analysis")
        results = analyzer.analyze_connectivity(
            batch_size=batch_size,
            # Use default dataset parameters
            target_neurons=50,
            sample_times=1,
            target_trials_per_site=500,  # Smaller for analysis
            min_val_test_trials=50
        )
        
        # 3. Save results
        logger.info("\n💾 Phase 3: Saving Analysis Results")
        results_file = results_path / 'connectivity_results.pth'
        analyzer.save_results(str(results_file))
        
        # 4. Generate visualizations
        logger.info("\n📈 Phase 4: Generating Visualizations")
        plotter = ConnectivityPlotter(dpi=300)
        
        # Print summary statistics
        logger.info("\n📋 Analysis Summary:")
        logger.info(f"   Sites analyzed: {len(results.site_ids)}")
        logger.info(f"   Site IDs: {results.site_ids}")
        logger.info(f"   Attention connectivity shape: {results.attention_connectivity.shape}")
        logger.info(f"   Temporal dynamics shape: {results.attention_temporal.shape}")
        logger.info(f"   Noise connectivity shape: {results.noise_connectivity.shape}")
        
        # A. Attention-based connectivity heatmap (diagonal removed & normalized)
        logger.info("   Generating attention connectivity heatmap...")
        fig1 = plotter.plot_connectivity_heatmap(
            results.attention_connectivity,
            title="Attention-Based Functional Connectivity\n(Diagonal Removed, Normalized to [0,1])",
            site_ids=results.site_ids,
            save_path=str(figures_path / 'attention_connectivity_heatmap.png'),
            remove_diagonal=True,
            annotate=False,
            use_anatomical_labels=True,
            site_coordinates=results.site_coordinates
        )
        plt.close(fig1)
        
        # B. Noise replacement connectivity heatmap (diagonal removed & normalized)
        logger.info("   Generating noise replacement connectivity heatmap...")
        fig2 = plotter.plot_connectivity_heatmap(
            results.noise_connectivity_normalized,
            title="Noise Replacement Information Flow\n(Diagonal Removed, Poisson Loss Based, Robust Normalized)",
            site_ids=results.site_ids,
            save_path=str(figures_path / 'noise_replacement_connectivity_heatmap.png'),
            remove_diagonal=True,
            annotate=False,
            use_anatomical_labels=True,
            site_coordinates=results.site_coordinates
        )
        plt.close(fig2)
        
        # C. Temporal connectivity dynamics (diagonal removed & normalized)
        logger.info("   Generating temporal connectivity dynamics...")
        fig3 = plotter.plot_temporal_connectivity_dynamics(
            results.attention_temporal,
            results.temporal_timepoints,
            title="Temporal Connectivity Dynamics During Reaching\n(Diagonal Removed, Each Timepoint Normalized)",
            site_ids=results.site_ids,
            save_path=str(figures_path / 'temporal_connectivity_dynamics.png'),
            remove_diagonal=True,
            use_anatomical_labels=True,
            site_coordinates=results.site_coordinates,
            show_colorbar=False
        )
        plt.close(fig3)
        
        # D. Anatomical connectivity layout (blank layout with threshold > 0.5)
        logger.info("   Generating anatomical connectivity layout...")
        # Process attention connectivity for anatomical layout
        processed_attention, _, _ = plotter.prepare_connectivity_for_plotting(
            results.attention_connectivity, remove_diagonal=True
        )
        fig4 = plotter.plot_anatomical_connectivity_layout(
            torch.tensor(processed_attention),
            results.site_coordinates,
            title="Anatomical Layout of Recording Sites\n(Blank Layout for Cleaner View)",
            threshold=0.4,  
            save_path=str(figures_path / 'anatomical_connectivity_layout.png')
        )
        plt.close(fig4)
        
        # D2. Anatomical connectivity layout for noise replacement
        logger.info("   Generating noise replacement anatomical connectivity layout...")
        # Process noise replacement connectivity for anatomical layout
        processed_noise, _, _ = plotter.prepare_connectivity_for_plotting(
            results.noise_connectivity_normalized, remove_diagonal=True
        )
        fig4b = plotter.plot_anatomical_connectivity_layout(
            torch.tensor(processed_noise),
            results.site_coordinates,
            title="Anatomical Layout with Noise Replacement Connectivity\n(Information Flow Dependencies)",
            threshold=0.8, 
            save_path=str(figures_path / 'anatomical_noise_replacement_layout.png')
        )
        plt.close(fig4b)
        
        # E. Method comparison (all methods with diagonal removal)
        logger.info("   Generating method comparison...")
        fig5 = plotter.plot_method_comparison(
            results.attention_connectivity,
            results.noise_connectivity_normalized,
            results.correlation_connectivity,
            site_ids=results.site_ids,
            save_path=str(figures_path / 'method_comparison.png'),
            remove_diagonal=True,
            use_anatomical_labels=True,
            site_coordinates=results.site_coordinates
        )
        plt.close(fig5)
        
        # F. Correlation analysis between methods (with diagonal removal)
        logger.info("   Generating method correlation analysis...")
        fig6 = plotter.plot_connectivity_correlation_analysis(
            results.attention_connectivity,
            results.noise_connectivity_normalized,
            save_path=str(figures_path / 'method_correlation_analysis.png'),
            remove_diagonal=True
        )
        plt.close(fig6)
        
        # 5. Generate analysis summary
        logger.info("\n📝 Phase 5: Generating Analysis Summary")
        generate_analysis_summary(results, output_path)
        
        logger.info("\n✅ Connectivity Analysis Completed Successfully!")
        logger.info(f"Results saved to: {output_path}")
        logger.info(f"Figures saved to: {figures_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        raise e


def generate_analysis_summary(results, output_path: Path):
    """Generate a comprehensive text summary of analysis results."""
    
    summary_file = output_path / 'analysis_summary.txt'
    
    # Compute summary statistics
    attention_conn = results.attention_connectivity
    noise_conn = results.noise_connectivity_normalized
    correlation_conn = results.correlation_connectivity
    
    # Find strongest connections for each method
    def get_top_connections(matrix, method_name, top_k=5):
        S = matrix.shape[0]
        connections = []
        for i in range(S):
            for j in range(S):
                if i != j:
                    connections.append((i, j, matrix[i, j].item()))
        connections.sort(key=lambda x: x[2], reverse=True)
        return connections[:top_k]
    
    attention_top = get_top_connections(attention_conn, "Attention")
    noise_top = get_top_connections(noise_conn, "Noise Replacement")
    
    # Process connectivity matrices with diagonal removal before correlation
    # Import plotter to use the same processing
    from .plotting_utils import ConnectivityPlotter
    plotter = ConnectivityPlotter()
    
    # Process both matrices with diagonal removal and normalization
    attn_processed, _, _ = plotter.prepare_connectivity_for_plotting(attention_conn, remove_diagonal=True)
    noise_processed, _, _ = plotter.prepare_connectivity_for_plotting(noise_conn, remove_diagonal=True)
    
    # Compute method correlation on processed data
    S = attn_processed.shape[0]
    mask = ~np.eye(S, dtype=bool)
    attn_values = attn_processed[mask]
    noise_values = noise_processed[mask]
    method_correlation = np.corrcoef(attn_values, noise_values)[0, 1]
    
    # Write summary
    with open(summary_file, 'w') as f:
        f.write("NEURAL FOUNDATION MODEL CONNECTIVITY ANALYSIS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Analysis Configuration:\n")
        f.write(f"  - Sites analyzed: {len(results.site_ids)}\n")
        f.write(f"  - Site IDs: {results.site_ids}\n")
        f.write(f"  - Temporal timepoints: {results.temporal_timepoints}\n\n")
        
        f.write(f"Connectivity Statistics:\n")
        f.write(f"  Attention-based connectivity:\n")
        f.write(f"    - Mean: {attention_conn.mean():.4f}\n")
        f.write(f"    - Std: {attention_conn.std():.4f}\n")
        f.write(f"    - Range: [{attention_conn.min():.4f}, {attention_conn.max():.4f}]\n\n")
        
        f.write(f"  Noise replacement connectivity:\n")
        f.write(f"    - Mean: {noise_conn.mean():.4f}\n")
        f.write(f"    - Std: {noise_conn.std():.4f}\n")
        f.write(f"    - Range: [{noise_conn.min():.4f}, {noise_conn.max():.4f}]\n\n")
        
        f.write(f"Method Correlation:\n")
        f.write(f"  - Pearson correlation between attention and noise methods: {method_correlation:.4f}\n\n")
        
        f.write(f"Top Connections (Attention-Based):\n")
        for i, (source, target, strength) in enumerate(attention_top):
            f.write(f"  {i+1}. Site {results.site_ids[source]} → Site {results.site_ids[target]}: {strength:.4f}\n")
        
        f.write(f"\nTop Connections (Noise Replacement):\n")
        for i, (source, target, strength) in enumerate(noise_top):
            f.write(f"  {i+1}. Site {results.site_ids[source]} → Site {results.site_ids[target]}: {strength:.4f}\n")
        
        f.write(f"\nFigures Generated:\n")
        f.write(f"  - attention_connectivity_heatmap.png\n")
        f.write(f"  - noise_replacement_connectivity_heatmap.png\n")
        f.write(f"  - temporal_connectivity_dynamics.png\n")
        f.write(f"  - anatomical_connectivity_layout.png\n")
        f.write(f"  - anatomical_noise_replacement_layout.png\n")
        f.write(f"  - method_comparison.png\n")
        f.write(f"  - method_correlation_analysis.png\n")


def main():
    """Main entry point for connectivity analysis."""
    
    parser = argparse.ArgumentParser(description='Neural Foundation Model Connectivity Analysis')
    
    # Required arguments
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to pretrained model checkpoint')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save results and figures')
    
    # Optional arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for analysis (default: 32)')
    parser.add_argument('--temporal_timepoints', type=int, nargs='+', 
                       default=[0, 15, 30, 45],
                       help='Timepoints for temporal analysis (default: 0 15 30 45)')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    parser.add_argument('--seed', type=int, default=3407,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.log_level)
    set_seed(args.seed)
    
    # Run analysis
    results = run_connectivity_analysis(
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        temporal_timepoints=args.temporal_timepoints
    )
    
    print(f"\n🎉 Analysis completed! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
