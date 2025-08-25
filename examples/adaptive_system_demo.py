#!/usr/bin/env python3
"""
Adaptive System Demo Script

This script demonstrates the new regime-aware adaptive trading system features.
Run this to see how market regime detection and adaptive parameters work.

Usage:
    poetry run python examples/adaptive_system_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.agents.regime_detector import (
    get_current_market_regime, 
    AdaptiveThresholds,
    get_adaptive_thresholds
)
from datetime import datetime, timedelta
import json


def demo_regime_detection():
    """Demonstrate regime detection capabilities."""
    print("ADAPTIVE AI HEDGE FUND - Phase 1 Demo")
    print("=" * 50)
    
    # Test regime detection on current date
    current_date = datetime.now().strftime('%Y-%m-%d')
    print(f"\nMarket Regime Analysis for {current_date}")
    print("-" * 40)
    
    try:
        regime = get_current_market_regime(current_date)
        
        print(f"Market State:   {regime.market_state.upper()}")
        print(f"Volatility:     {regime.volatility.upper()}")  
        print(f"Structure:      {regime.structure.replace('_', ' ').title()}")
        print(f"Confidence:     {regime.confidence:.2%}")
        
        # Show regime impact
        regime_description = get_regime_description(regime)
        print(f"\nWhat this means:")
        print(f"   {regime_description}")
        
    except Exception as e:
        print(f"Regime detection failed: {e}")
        print("   Using fallback neutral regime for demo...")
        from src.agents.regime_detector import MarketRegime
        regime = MarketRegime(
            market_state="neutral",
            volatility="low",
            structure="mean_reverting", 
            confidence=0.3,
            timestamp=datetime.now()
        )
    
    return regime


def demo_adaptive_thresholds(regime):
    """Demonstrate adaptive threshold system."""
    print(f"\nAdaptive Thresholds for {regime.market_state.upper()} Market")
    print("-" * 45)
    
    # Compare static vs adaptive thresholds
    print("Parameter                Static  ->  Adaptive    Change")
    print("-" * 55)
    
    key_parameters = [
        ("ROE Threshold", "roe_threshold", ".1%"),
        ("P/E Max", "pe_ratio_threshold", ".0f"), 
        ("Discount Rate", "discount_rate", ".1%"),
        ("Margin of Safety", "margin_of_safety", ".1%"),
        ("RSI Period", "rsi_period", ".0f"),
        ("RSI Oversold", "rsi_oversold", ".0f"),
        ("Bollinger Std Dev", "bollinger_std", ".1f"),
    ]
    
    for param_name, param_key, fmt_str in key_parameters:
        try:
            base_value = AdaptiveThresholds.BASE_THRESHOLDS[param_key]
            adaptive_value = AdaptiveThresholds.get_adaptive_threshold(param_key, regime)
            
            change_pct = (adaptive_value - base_value) / base_value * 100
            arrow = "UP" if change_pct > 0 else "DOWN" if change_pct < 0 else "-->"
            
            print(f"{param_name:<20} {base_value:{fmt_str}}  ->  {adaptive_value:{fmt_str}}    {arrow} {change_pct:+.1f}%")
            
        except KeyError:
            continue
    
    print(f"\nCHANGE Key Changes for {regime.market_state.title()} Market:")
    print_regime_specific_changes(regime)


def demo_historical_regimes():
    """Show regime detection for historical periods."""
    print(f"\nCHANGE Historical Regime Examples")
    print("-" * 30)
    
    historical_dates = [
        ("2020-03-20", "COVID Crash"),
        ("2021-06-15", "Bull Market Peak"),
        ("2022-06-15", "Bear Market"),  
        ("2023-03-15", "Recovery Phase"),
    ]
    
    for date_str, description in historical_dates:
        try:
            regime = get_current_market_regime(date_str)
            print(f"{description:<20} {date_str}  ->  {regime.market_state}/{regime.volatility}/{regime.structure} ({regime.confidence:.0%})")
        except Exception as e:
            print(f"{description:<20} {date_str}  ->  Error: {str(e)[:30]}...")


def demo_sector_adjustments(regime):
    """Show sector-specific adjustments."""
    print(f"\nCHANGE Sector-Specific Adaptations")
    print("-" * 35)
    
    sectors = ["technology", "utilities", "healthcare", "financials"]
    
    print(f"Sector         ROE Threshold    P/E Threshold")
    print("-" * 45)
    
    for sector in sectors:
        try:
            roe_threshold = AdaptiveThresholds.get_adaptive_threshold("roe_threshold", regime, sector)
            pe_threshold = AdaptiveThresholds.get_adaptive_threshold("pe_ratio_threshold", regime, sector)
            
            print(f"{sector.title():<12}   {roe_threshold:.1%}           {pe_threshold:.0f}")
        except Exception as e:
            print(f"{sector.title():<12}   Error: {str(e)[:20]}...")


def get_regime_description(regime):
    """Get human-readable regime description."""
    state_desc = {
        "bull": "Optimistic market conditions - slightly relaxed valuation criteria",
        "bear": "Pessimistic market conditions - stricter requirements for safety", 
        "neutral": "Balanced market conditions - standard criteria apply"
    }
    
    vol_desc = {
        "high": "with increased margin of safety due to uncertainty",
        "low": "with standard volatility adjustments"
    }
    
    struct_desc = {
        "trending": "Trend-following strategies emphasized in technical analysis",
        "mean_reverting": "Mean-reversion strategies emphasized in technical analysis"
    }
    
    return f"{state_desc[regime.market_state]} {vol_desc[regime.volatility]}. {struct_desc[regime.structure]}."


def print_regime_specific_changes(regime):
    """Print regime-specific changes."""
    if regime.market_state == "bull":
        print("  • Higher P/E ratios acceptable (growth premium)")
        print("  • Lower discount rates (lower risk perception)")
        print("  • Reduced margin of safety requirements")
    elif regime.market_state == "bear":
        print("  • Stricter fundamental requirements (higher ROE)")
        print("  • Higher discount rates (increased risk)")
        print("  • Larger margin of safety demanded")
    else:
        print("  • Balanced parameters between bull/bear extremes")
    
    if regime.volatility == "high":
        print("  • Wider Bollinger Bands for technical analysis")
        print("  • Additional margin of safety buffer")
        print("  • Longer indicator periods for stability")
    
    if regime.structure == "trending":
        print("  • Technical analysis emphasizes trend-following (35% weight)")
        print("  • Momentum strategies get confidence boost")
    else:
        print("  • Technical analysis emphasizes mean-reversion (35% weight)")
        print("  • Statistical arbitrage gets higher weighting")


def demo_usage_examples():
    """Show code usage examples."""
    print(f"\nCHANGE Usage Examples")
    print("-" * 18)
    
    print("# Basic regime detection:")
    print("from src.agents.regime_detector import get_current_market_regime")
    print("regime = get_current_market_regime('2024-03-15')")
    print("print(f'Market: {regime.market_state}, Confidence: {regime.confidence}')")
    
    print("\n# Get adaptive thresholds:")
    print("from src.agents.regime_detector import AdaptiveThresholds")  
    print("roe_threshold = AdaptiveThresholds.get_adaptive_threshold('roe_threshold', regime, 'technology')")
    
    print("\n# Run backtesting with adaptive system (no code changes needed):")
    print("poetry run python src/backtester.py --ticker AAPL,MSFT --start-date 2024-01-01")


def main():
    """Run the complete adaptive system demonstration."""
    try:
        # Demo regime detection
        regime = demo_regime_detection()
        
        # Demo adaptive thresholds
        demo_adaptive_thresholds(regime)
        
        # Demo historical regimes
        demo_historical_regimes()
        
        # Demo sector adjustments
        demo_sector_adjustments(regime)
        
        # Show usage examples
        demo_usage_examples()
        
        print(f"\nCHANGEAdaptive System Demo Complete!")
        print(f"   The system is now ready to use with regime-aware intelligence.")
        print(f"   Run normal backtesting commands - adaptive parameters activate automatically.")
        
    except Exception as e:
        print(f"\nCHANGEDemo failed with error: {e}")
        print(f"   This may indicate missing dependencies or data access issues.")
        print(f"   Please check the installation guide in ADAPTIVE_SYSTEM_GUIDE.md")


if __name__ == "__main__":
    main()