#!/usr/bin/env python3
"""
Test regime detection variation to understand why all regimes are neutral
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_regime_variation():
    """Test regime detection across different dates to see variation"""
    from agents.regime_detector import get_current_market_regime
    
    # Test different dates
    test_dates = [
        '2023-01-02',
        '2023-01-15', 
        '2023-02-01',
        '2023-03-15',
        '2023-06-01',
        '2023-09-15',
        '2023-12-01'
    ]
    
    print("Testing regime detection across different dates:")
    print("=" * 80)
    
    regimes = {}
    
    for date in test_dates:
        try:
            regime = get_current_market_regime(date)
            regimes[date] = regime
            
            print(f"Date: {date}")
            print(f"  Market State: {regime.market_state}")
            print(f"  Volatility: {regime.volatility}") 
            print(f"  Structure: {regime.structure}")
            print(f"  Confidence: {regime.confidence:.3f}")
            print(f"  Raw Probabilities: {regime.raw_probabilities}")
            print()
            
        except Exception as e:
            print(f"Date: {date} - ERROR: {e}")
            print()
    
    # Analyze results
    market_states = [r.market_state for r in regimes.values()]
    volatility_states = [r.volatility for r in regimes.values()]
    structure_states = [r.structure for r in regimes.values()]
    
    print("SUMMARY:")
    print("=" * 40)
    print(f"Market states: {set(market_states)}")
    print(f"Volatility states: {set(volatility_states)}")
    print(f"Structure states: {set(structure_states)}")
    
    print(f"\nMarket state distribution:")
    for state in ['bull', 'bear', 'neutral']:
        count = market_states.count(state)
        print(f"  {state}: {count}/{len(market_states)} ({count/len(market_states)*100:.1f}%)")
    
    return regimes

def test_manual_regime_detection():
    """Test regime detection with manually created market data"""
    from agents.regime_detector import HeuristicRegimeDetector
    import random
    
    detector = HeuristicRegimeDetector()
    
    print("\nTesting manual regime detection:")
    print("=" * 80)
    
    # Create strongly bullish data
    bullish_prices = []
    base_price = 100.0
    for i in range(60):
        price = base_price + (i * 2.0) + random.uniform(-1, 1)  # Strong uptrend
        bullish_prices.append({
            'time': f'2023-{(i//30)+1:02d}-{(i%30)+1:02d}',
            'close': price,
            'volume': 1000000,
            'open': price - 0.5,
            'high': price + 1,
            'low': price - 1
        })
    
    # Create strongly bearish data  
    bearish_prices = []
    base_price = 200.0
    for i in range(60):
        price = base_price - (i * 2.0) + random.uniform(-1, 1)  # Strong downtrend
        bearish_prices.append({
            'time': f'2023-{(i//30)+1:02d}-{(i%30)+1:02d}',
            'close': price,
            'volume': 1000000,
            'open': price + 0.5,
            'high': price + 1,
            'low': price - 1
        })
    
    # Test bullish scenario
    bull_regime = detector.detect_regime(bullish_prices, '2023-02-28')
    print(f"Bullish test data result:")
    print(f"  Market State: {bull_regime.market_state}")
    print(f"  Confidence: {bull_regime.confidence:.3f}")
    
    # Test bearish scenario
    bear_regime = detector.detect_regime(bearish_prices, '2023-02-28')  
    print(f"Bearish test data result:")
    print(f"  Market State: {bear_regime.market_state}")
    print(f"  Confidence: {bear_regime.confidence:.3f}")
    
    return bull_regime, bear_regime

if __name__ == "__main__":
    print("REGIME DETECTION VARIATION TEST")
    print("=" * 80)
    
    # Test 1: Real date variation
    regimes = test_regime_variation()
    
    # Test 2: Manual scenarios
    bull_result, bear_result = test_manual_regime_detection()
    
    # Check if we can get non-neutral regimes
    has_non_neutral = any(r.market_state != 'neutral' for r in regimes.values())
    print(f"\nFound non-neutral regimes: {has_non_neutral}")
    
    if bull_result.market_state != 'neutral' or bear_result.market_state != 'neutral':
        print("✓ Heuristic detector can produce bull/bear regimes with proper data")
    else:
        print("✗ Even with extreme data, detector only produces neutral regimes")