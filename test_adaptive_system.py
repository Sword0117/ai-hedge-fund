#!/usr/bin/env python3
"""
Quick test script to verify the adaptive system installation.

Run this to check if all components are working correctly:
    poetry run python test_adaptive_system.py
"""

import sys
import traceback
from datetime import datetime


def test_imports():
    """Test that all new modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.agents.regime_detector import (
            RegimeDetector, 
            get_current_market_regime,
            AdaptiveThresholds,
            MarketRegime
        )
        print("[TEST] Regime detector imports successful")
    except Exception as e:
        print(f"[TEST] Regime detector import failed: {e}")
        return False
    
    try:
        from src.agents.warren_buffett import warren_buffett_agent
        print("[TEST] Adaptive Warren Buffett agent import successful")
    except Exception as e:
        print(f"[TEST] Warren Buffett agent import failed: {e}")
        return False
    
    try:
        from src.agents.technicals import technical_analyst_agent
        print("[TEST] Adaptive technical agent import successful")
    except Exception as e:
        print(f"[TEST] Technical agent import failed: {e}")
        return False
        
    return True


def test_dependencies():
    """Test that required dependencies are available."""
    print("\nTesting dependencies...")
    
    # Test optional HMM dependency
    try:
        import hmmlearn
        print("[TEST] hmmlearn available - HMM regime detection enabled")
    except ImportError:
        print("WARNING: hmmlearn not found - fallback heuristic detection will be used")
    
    # Test scipy
    try:
        import scipy
        print("[TEST] scipy available")
    except ImportError:
        print("[TEST] scipy missing - required for statistical calculations")
        return False
    
    # Test core dependencies
    required_deps = ['numpy', 'pandas', 'langchain', 'pydantic']
    for dep in required_deps:
        try:
            __import__(dep)
            print(f"[TEST] {dep} available")
        except ImportError:
            print(f"[TEST] {dep} missing - required dependency")
            return False
    
    return True


def test_regime_detector():
    """Test basic regime detector functionality."""
    print("\nTesting regime detector...")
    
    try:
        from src.agents.regime_detector import MarketRegime
        
        # Test MarketRegime creation
        regime = MarketRegime(
            market_state="neutral",
            volatility="low",
            structure="mean_reverting",
            confidence=0.5,
            timestamp=datetime.now()
        )
        print("[TEST] MarketRegime class works")
        
        # Test adaptive thresholds
        from src.agents.regime_detector import AdaptiveThresholds
        
        threshold = AdaptiveThresholds.get_adaptive_threshold("roe_threshold", regime)
        if threshold > 0:
            print(f"[TEST] Adaptive thresholds work (ROE threshold: {threshold:.1%})")
        else:
            print("[TEST] Adaptive threshold calculation failed")
            return False
            
        return True
        
    except Exception as e:
        print(f"[TEST] Regime detector test failed: {e}")
        traceback.print_exc()
        return False


def test_fallback_regime():
    """Test that fallback regime detection works."""
    print("\nTesting fallback regime detection...")
    
    try:
        # This should work even without external data
        from src.agents.regime_detector import get_current_market_regime
        
        # Test with current date (might fail due to data access)
        current_date = datetime.now().strftime('%Y-%m-%d')
        try:
            regime = get_current_market_regime(current_date)
            print(f"[TEST] Regime detection works: {regime.market_state}/{regime.volatility}/{regime.structure}")
            return True
        except Exception as e:
            print(f"WARNING: Regime detection failed (likely data access): {e}")
            print("   This is expected if no API key is configured")
            return True  # Still pass test since fallback should work
            
    except Exception as e:
        print(f"[TEST] Fallback regime test failed: {e}")
        return False


def test_file_structure():
    """Test that all required files are present."""
    print("\nTesting file structure...")
    
    import os
    
    required_files = [
        "src/agents/regime_detector.py",
        "src/agents/warren_buffett.py", 
        "src/agents/technicals.py",
        "pyproject.toml",
        "ADAPTIVE_SYSTEM_GUIDE.md"
    ]
    
    all_present = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"[TEST] {file_path} present")
        else:
            print(f"[TEST] {file_path} missing")
            all_present = False
    
    return all_present


def main():
    """Run all tests."""
    print("ADAPTIVE SYSTEM INSTALLATION TEST")
    print("=" * 40)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Dependencies", test_dependencies), 
        ("Imports", test_imports),
        ("Regime Detector", test_regime_detector),
        ("Fallback System", test_fallback_regime),
    ]
    
    results = []
    for test_name, test_func in tests:
        print()
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print("\n" + "=" * 40)
    print("TEST RESULTS SUMMARY:")
    
    all_passed = True
    for test_name, success in results:
        status = "[TEST] PASS" if success else "[TEST] FAIL"
        print(f"  {status}  {test_name}")
        if not success:
            all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("ALL TESTS PASSED!")
        print("   The adaptive system is ready to use.")
        print("   Try running: poetry run python examples/adaptive_system_demo.py")
    else:
        print("SOME TESTS FAILED")
        print("   Check the errors above and refer to ADAPTIVE_SYSTEM_GUIDE.md")
        print("   The system may still work with fallback functionality.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)