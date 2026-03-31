#!/usr/bin/env python3
"""
Overnight Split Ratio Experiments with Pretraining Method Comparison
====================================================================
Systematically test different training data amounts while keeping test set fixed at 50%.
Compares two pretraining methods:
1. Regular chunking (mae_medium_bs64_lr5e-5_mn0.25_mt0.25_lc0.5_ex13122_14116)
2. First 100 neurons (mae_medium_bs64_lr5e-5_mn0.25_mt0.25_lc0.5_ex13122_14116_first100)
"""

import subprocess
import time
import logging
from datetime import datetime
import os
import argparse

def setup_logging():
    """Setup logging for the experiment runner."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"split_ratio_experiments_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def run_experiment(train_ratio, valid_ratio, test_ratio, dataset_id, pretraining_method, logger):
    """Run a single fine-tuning experiment with given split ratios and pretraining method."""
    
    # Create experiment name with pretraining method suffix
    train_pct = int(train_ratio * 100)
    if pretraining_method == "chunking":
        exp_name = f"split_study_train{train_pct}pct_chunking"
        pretrained_path = "./logs/checkpoints/mae_medium_bs64_lr5e-5_mn0.25_mt0.25_lc0.5_ex13122_14116/best_model.pth"
    else:  # first100
        exp_name = f"split_study_train{train_pct}pct_first100"
        pretrained_path = "./logs/checkpoints/mae_medium_bs64_lr5e-5_mn0.25_mt0.25_lc0.5_ex13122_14116_first100/best_model.pth"
    
    # Construct command (use the organized script path)
    cmd = [
        "python", "scripts/training/finetune_regression.py",
        "--training_mode", "frozen_encoder",
        "--dataset_id", dataset_id,
        "--target_type", "velocity", 
        "--split_ratios", str(train_ratio), str(valid_ratio), str(test_ratio),
        "--neuron_selection_strategy", "first_n",
        "--selected_neurons", "100",
        "--num_epochs", "200",
        "--experiment_name", exp_name,
        "--pretrained_path", pretrained_path
    ]
    
    logger.info(f"Starting experiment: Train={train_pct}%, Valid={int(valid_ratio*100)}%, Test={int(test_ratio*100)}% [{pretraining_method}]")
    logger.info(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    
    try:
        # Run the experiment
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            logger.info(f"✓ Experiment completed successfully in {duration:.1f}s")
            logger.info(f"  Final lines of output: {result.stdout.strip().split(chr(10))[-3:]}")
        else:
            logger.error(f"✗ Experiment failed with return code {result.returncode}")
            logger.error(f"  Error: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        logger.error(f"✗ Experiment timed out after 1 hour")
    except Exception as e:
        logger.error(f"✗ Experiment failed with exception: {e}")
    
    return result.returncode == 0 if 'result' in locals() else False

def main():
    """Main experiment runner."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run split ratio experiments with pretraining method comparison')
    parser.add_argument('--pretraining_method', type=str, choices=['chunking', 'first100', 'both'], 
                       default='both', help='Pretraining method to test')
    parser.add_argument('--dataset_id', type=str, default='te14116', 
                       help='Dataset ID to use for experiments')
    args = parser.parse_args()
    
    logger = setup_logging()
    
    logger.info("=" * 80)
    logger.info("SPLIT RATIO EXPERIMENTS WITH PRETRAINING COMPARISON")
    logger.info("=" * 80)
    logger.info("Testing how training data amount affects performance")
    logger.info("Fixed: Test=50%, Neuron selection=first_n, Training mode=frozen_encoder")
    logger.info("Variable: Training data percentage (1% to 25%)")
    logger.info(f"Pretraining method(s): {args.pretraining_method}")
    
    # Experiment configuration
    dataset_id = args.dataset_id
    test_ratio = 0.50  # Fixed at 50% as requested
    
    # Define training ratios to test (1%, 2%, 5%, 10%, 15%, 20%, 25%)
    train_ratios = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25]
    
    # Calculate validation ratios (remaining after train and test)
    split_configs = []
    for train_ratio in train_ratios:
        valid_ratio = 1.0 - train_ratio - test_ratio
        if valid_ratio > 0:
            split_configs.append((train_ratio, valid_ratio, test_ratio))
    
    # Determine which pretraining methods to test
    if args.pretraining_method == 'both':
        methods = ['chunking', 'first100']
    else:
        methods = [args.pretraining_method]
    
    # Create experiment list
    experiments = []
    for method in methods:
        for train_ratio, valid_ratio, test_ratio in split_configs:
            experiments.append((train_ratio, valid_ratio, test_ratio, method))
    
    logger.info(f"Will run {len(experiments)} experiments:")
    logger.info(f"Split configurations: {len(split_configs)}")
    logger.info(f"Pretraining methods: {methods}")
    for i, (train, valid, test, method) in enumerate(experiments, 1):
        logger.info(f"  {i}. Train={train*100:.0f}%, Valid={valid*100:.0f}%, Test={test*100:.0f}% [{method}]")
    
    # Run experiments
    successful = 0
    failed = 0
    total_start_time = time.time()
    
    for i, (train_ratio, valid_ratio, test_ratio, method) in enumerate(experiments, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"EXPERIMENT {i}/{len(experiments)}")
        logger.info(f"{'='*60}")
        
        success = run_experiment(train_ratio, valid_ratio, test_ratio, dataset_id, method, logger)
        
        if success:
            successful += 1
        else:
            failed += 1
            
        # Brief pause between experiments
        if i < len(experiments):
            logger.info("Pausing 30 seconds before next experiment...")
            time.sleep(30)
    
    # Final summary
    total_duration = time.time() - total_start_time
    logger.info(f"\n{'='*80}")
    logger.info("EXPERIMENT SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Total experiments: {len(experiments)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total time: {total_duration/3600:.1f} hours")
    logger.info(f"Average time per experiment: {total_duration/len(experiments)/60:.1f} minutes")
    
    # Analysis instructions
    logger.info(f"\n{'='*80}")
    logger.info("ANALYSIS INSTRUCTIONS")
    logger.info(f"{'='*80}")
    logger.info("After completion, compare results using TensorBoard:")
    logger.info("1. tensorboard --logdir ./logs/runs --port 6006")
    logger.info("2. Look for experiments named 'split_study_train*pct_chunking' and 'split_study_train*pct_first100'")
    logger.info("3. Compare final R2 scores and convergence rates between pretraining methods")
    logger.info("4. Expected trends:")
    logger.info("   - Higher training % → Better performance for both methods")
    logger.info("   - Compare which pretraining method performs better")
    logger.info("   - Analyze if first100 neurons are sufficient for good performance")
    
    if successful == len(experiments):
        logger.info("\n🎉 All experiments completed successfully!")
    else:
        logger.info(f"\n⚠️  {failed} experiments failed. Check logs above for details.")

if __name__ == "__main__":
    main() 