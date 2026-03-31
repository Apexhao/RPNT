#!/usr/bin/env python3
"""
Script to run pretraining with first 100 neurons only (no chunking).
This enables direct comparison between consistent neuron sets in pretraining and fine-tuning.
"""

import subprocess
import sys
import os
from datetime import datetime

def run_pretrain_first100():
    """Run pretraining with first 100 neurons only."""
    
    print("=" * 80)
    print("🧠 NEURAL FOUNDATION MODEL PRETRAINING - FIRST 100 NEURONS ONLY")
    print("=" * 80)
    print(f"⏰ Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Configuration
    config_file = "config/config_pretrain_first100.yaml"
    
    print("🔧 CONFIGURATION:")
    print(f"   Config File: {config_file}")
    print(f"   Mode: First 100 neurons only (no chunking)")
    print(f"   Data Consistency: Perfect match with fine-tuning")
    print()
    
    # Check if config file exists
    if not os.path.exists(config_file):
        print(f"❌ Error: Config file {config_file} not found!")
        return False
    
    # Build command (use the organized script path)
    cmd = [
        sys.executable, "scripts/training/pretrain.py",
        "--config", config_file
    ]
    
    print("🚀 COMMAND:")
    print(f"   {' '.join(cmd)}")
    print()
    
    try:
        # Run the command
        print("▶️  Starting pretraining...")
        print("=" * 80)
        
        result = subprocess.run(cmd, check=True)
        
        print("=" * 80)
        print(f"✅ Pretraining completed successfully!")
        print(f"⏰ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        print("📁 Output locations:")
        print("   Checkpoints: ./logs/checkpoints/mae_medium_*_first100/")
        print("   TensorBoard: ./logs/runs/mae_medium_*_first100/")
        print()
        print("🔬 NEXT STEPS:")
        print("   1. Compare with chunked pretraining results")
        print("   2. Run fine-tuning with first_n=100 strategy")
        print("   3. Analyze performance differences")
        print("=" * 80)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print("=" * 80)
        print(f"❌ Pretraining failed with exit code {e.returncode}")
        print("Check the logs above for error details.")
        return False
    except KeyboardInterrupt:
        print("=" * 80)
        print("⚠️  Pretraining interrupted by user")
        return False
    except Exception as e:
        print("=" * 80)
        print(f"❌ Unexpected error: {str(e)}")
        return False

def main():
    """Main function."""
    success = run_pretrain_first100()
    exit_code = 0 if success else 1
    sys.exit(exit_code)

if __name__ == "__main__":
    main() 