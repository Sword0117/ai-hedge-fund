#!/usr/bin/env python3
"""
Verification script to confirm the .model_dump() issue is completely fixed
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def verify_regime_detection():
    """Verify regime detection produces varied results"""
    from agents.regime_detector import get_current_market_regime
    
    print("Testing regime detection for varied results...")
    
    test_dates = ['2023-01-02', '2023-06-15', '2023-12-01']
    regimes = []
    
    for date in test_dates:
        regime = get_current_market_regime(date)
        regimes.append(regime.market_state)
        print(f"  {date}: {regime.market_state} (confidence: {regime.confidence:.3f})")
    
    unique_states = set(regimes)
    print(f"  Unique states found: {unique_states}")
    
    return len(unique_states) > 1  # Should have multiple states

def verify_training_pipeline():
    """Verify training pipeline runs without model_dump errors"""
    import subprocess
    import os
    
    print("Testing training pipeline...")
    
    os.chdir(Path(__file__).parent.parent)
    
    # Run training script and capture output
    result = subprocess.run([
        'poetry', 'run', 'python', 'scripts/train_signal_ensemble.py', 
        '--generate-sample'
    ], capture_output=True, text=True)
    
    has_model_dump_error = 'model_dump' in result.stderr
    completed_successfully = 'Successfully parsed sample data' in result.stdout or 'Successfully parsed sample data' in result.stderr
    
    print(f"  Model dump errors: {'NONE' if not has_model_dump_error else 'PRESENT'}")
    print(f"  Completed successfully: {'YES' if completed_successfully else 'NO'}")
    
    return not has_model_dump_error and completed_successfully

def main():
    """Run comprehensive verification"""
    print("=" * 60)
    print("REGIME DETECTION FIX VERIFICATION")
    print("=" * 60)
    
    # Test 1: Regime variation
    varied_regimes = verify_regime_detection()
    
    print()
    
    # Test 2: Training pipeline
    training_ok = verify_training_pipeline()
    
    print()
    print("=" * 60)
    print("VERIFICATION RESULTS")
    print("=" * 60)
    print(f"Regime Variation: {'PASS' if varied_regimes else 'FAIL'}")
    print(f"Training Pipeline: {'PASS' if training_ok else 'FAIL'}")
    
    if varied_regimes and training_ok:
        print("\nALL TESTS PASSED - Fix is successful!")
        print("- No more .model_dump() errors")
        print("- Regime detection produces varied bull/bear/neutral states")
        print("- Training pipeline runs successfully")
        print("- Ready for ML ensemble training with proper regime data")
    else:
        print("\nSome tests failed - fix needs more work")
    
    return varied_regimes and training_ok

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)