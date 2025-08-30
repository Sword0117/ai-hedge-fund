#!/usr/bin/env python3
"""
Diagnostic Script for Regime Detection Issues

Systematically tests and debugs the regime detection system to locate
the exact source of .model_dump() errors.
"""

import sys
import os
from pathlib import Path
import traceback
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_api_data_format():
    """Test what format the API actually returns"""
    print("=" * 50)
    print("TEST 1: API Data Format")
    print("=" * 50)
    
    try:
        from tools.api import get_prices
        from dotenv import load_dotenv
        
        load_dotenv()
        api_key = os.getenv('FINANCIAL_DATASETS_API_KEY')
        
        # Test single day to minimize API calls
        result = get_prices('AAPL', '2023-01-02', '2023-01-02', api_key)
        
        print(f"Result type: {type(result)}")
        print(f"Result length: {len(result) if result else 0}")
        
        if result:
            first_item = result[0]
            print(f"First item type: {type(first_item)}")
            print(f"Has model_dump: {hasattr(first_item, 'model_dump')}")
            print(f"Has dict method: {hasattr(first_item, 'dict')}")
            print(f"Is dict: {isinstance(first_item, dict)}")
            
            if isinstance(first_item, dict):
                print(f"Dict keys: {list(first_item.keys())}")
            else:
                print(f"Object attributes: {[attr for attr in dir(first_item) if not attr.startswith('_')]}")
                
        return True
        
    except Exception as e:
        print(f"ERROR in API test: {e}")
        traceback.print_exc()
        return False

def test_regime_detector_init():
    """Test regime detector initialization"""
    print("\n" + "=" * 50)
    print("TEST 2: Regime Detector Initialization")
    print("=" * 50)
    
    try:
        from agents.regime_detector import get_regime_detector
        
        detector = get_regime_detector()
        print(f"Detector type: {type(detector)}")
        print(f"Has API key: {bool(detector.api_key)}")
        print(f"Cache dir: {detector.market_cache_dir}")
        print(f"Available tickers: {detector.available_tickers}")
        
        # Test cache stats
        cache_stats = detector.get_cache_stats()
        print(f"Cache stats: {cache_stats}")
        
        return True
        
    except Exception as e:
        print(f"ERROR in detector init: {e}")
        traceback.print_exc()
        return False

def test_market_data_fetch():
    """Test market data fetching with detailed debugging"""
    print("\n" + "=" * 50)
    print("TEST 3: Market Data Fetch")
    print("=" * 50)
    
    try:
        from agents.regime_detector import get_regime_detector
        
        detector = get_regime_detector()
        
        # Test the internal method that's likely failing
        print("Testing _get_market_proxy_data...")
        
        try:
            market_data = detector._get_market_proxy_data('2023-01-02', detector.api_key)
            
            print(f"Market data type: {type(market_data)}")
            print(f"Market data length: {len(market_data) if market_data else 0}")
            
            if market_data:
                first_item = market_data[0]
                print(f"First market data item type: {type(first_item)}")
                print(f"First item has model_dump: {hasattr(first_item, 'model_dump')}")
                print(f"First item is dict: {isinstance(first_item, dict)}")
                
                if isinstance(first_item, dict):
                    print(f"First item keys: {list(first_item.keys())}")
                
            return market_data
            
        except Exception as e:
            print(f"ERROR in _get_market_proxy_data: {e}")
            traceback.print_exc()
            return None
            
    except Exception as e:
        print(f"ERROR in market data fetch test: {e}")
        traceback.print_exc()
        return None

def test_heuristic_detector():
    """Test heuristic detector with sample data"""
    print("\n" + "=" * 50)
    print("TEST 4: Heuristic Detector")
    print("=" * 50)
    
    try:
        from agents.regime_detector import HeuristicRegimeDetector
        
        detector = HeuristicRegimeDetector()
        
        # Create sample price data as dicts
        sample_prices = []
        base_price = 150.0
        
        for i in range(60):  # 60 days of sample data
            price = base_price + (i * 0.5) + ((-1) ** i * 2.0)  # Trending with noise
            sample_prices.append({
                'time': f'2023-01-{(i % 30) + 1:02d}',
                'date': f'2023-01-{(i % 30) + 1:02d}',
                'open': price - 1.0,
                'high': price + 2.0,
                'low': price - 2.0,
                'close': price,
                'volume': 1000000 + (i * 10000)
            })
        
        print(f"Sample data created: {len(sample_prices)} price records")
        print(f"First record type: {type(sample_prices[0])}")
        print(f"First record: {sample_prices[0]}")
        
        # Test regime detection
        regime = detector.detect_regime(sample_prices, '2023-01-30')
        
        print(f"Regime result type: {type(regime)}")
        print(f"Regime result: {regime}")
        
        return True
        
    except Exception as e:
        print(f"ERROR in heuristic detector test: {e}")
        traceback.print_exc()
        return False

def test_full_regime_detection():
    """Test full regime detection pipeline"""
    print("\n" + "=" * 50)
    print("TEST 5: Full Regime Detection")
    print("=" * 50)
    
    # Add detailed logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger()
    
    # Create custom handler to capture regime detector logs
    class DebugHandler(logging.Handler):
        def __init__(self):
            super().__init__()
            self.messages = []
        
        def emit(self, record):
            if 'model_dump' in record.getMessage():
                self.messages.append(record)
                print(f"CAPTURED ERROR: {record.getMessage()}")
                print(f"File: {record.filename}:{record.lineno}")
    
    debug_handler = DebugHandler()
    logger.addHandler(debug_handler)
    
    try:
        from agents.regime_detector import get_current_market_regime
        
        print("Testing get_current_market_regime...")
        
        regime = get_current_market_regime('2023-01-02')
        
        print(f"Regime type: {type(regime)}")
        print(f"Regime: {regime}")
        
        if debug_handler.messages:
            print(f"Found {len(debug_handler.messages)} model_dump errors")
            for msg in debug_handler.messages:
                print(f"Error location: {msg.filename}:{msg.lineno}")
        
        return regime
        
    except Exception as e:
        print(f"ERROR in full regime detection: {e}")
        traceback.print_exc()
        return None
    finally:
        logger.removeHandler(debug_handler)

def main():
    """Run all diagnostic tests"""
    print("REGIME DETECTION DIAGNOSTIC SUITE")
    print("=" * 80)
    
    # Test 1: API Data Format
    api_ok = test_api_data_format()
    
    # Test 2: Detector Init
    init_ok = test_regime_detector_init()
    
    # Test 3: Market Data Fetch
    market_data = test_market_data_fetch()
    
    # Test 4: Heuristic Detector
    heuristic_ok = test_heuristic_detector()
    
    # Test 5: Full Pipeline
    regime = test_full_regime_detection()
    
    # Summary
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)
    print(f"API Data Format Test: {'✓ PASS' if api_ok else '✗ FAIL'}")
    print(f"Detector Init Test: {'✓ PASS' if init_ok else '✗ FAIL'}")
    print(f"Market Data Fetch Test: {'✓ PASS' if market_data else '✗ FAIL'}")
    print(f"Heuristic Detector Test: {'✓ PASS' if heuristic_ok else '✗ FAIL'}")
    print(f"Full Pipeline Test: {'✓ PASS' if regime else '✗ FAIL'}")
    
    if regime:
        print(f"\nFinal Regime Result:")
        print(f"Type: {type(regime)}")
        print(f"Market State: {regime.market_state}")
        print(f"Volatility: {regime.volatility}")
        print(f"Structure: {regime.structure}")
        print(f"Confidence: {regime.confidence}")

if __name__ == "__main__":
    main()