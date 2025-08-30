#!/usr/bin/env python3
"""
Cache Initialization Script

Pre-populates market data cache and provides cache management utilities.
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.regime_detector import get_regime_detector
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def init_cache_for_tickers(tickers, days_back=365):
    """
    Pre-populate cache with historical data for specified tickers.
    
    Args:
        tickers: List of ticker symbols
        days_back: Number of days of historical data to cache
    """
    detector = get_regime_detector()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    logger.info(f"Pre-populating cache for {len(tickers)} tickers from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    success_count = 0
    
    for ticker in tickers:
        try:
            logger.info(f"Caching data for {ticker}...")
            
            # This will automatically cache the data if not already cached
            detector._get_market_proxy_data(end_date.strftime('%Y-%m-%d'), detector.api_key)
            success_count += 1
            
        except Exception as e:
            logger.error(f"Failed to cache data for {ticker}: {e}")
    
    logger.info(f"Successfully cached data for {success_count}/{len(tickers)} tickers")
    return success_count

def show_cache_stats():
    """Display comprehensive cache statistics."""
    detector = get_regime_detector()
    stats = detector.get_cache_stats()
    
    if "error" in stats:
        logger.error(f"Cache stats error: {stats['error']}")
        return
    
    print("\n" + "="*50)
    print("MARKET DATA CACHE STATISTICS")
    print("="*50)
    print(f"Cache Location: {stats['cache_location']}")
    print(f"Total Cached Periods: {stats['total_cached_periods']}")
    print(f"Cache Size: {stats['cache_size_mb']} MB")
    
    if stats['ticker_breakdown']:
        print(f"\nCached Data by Ticker:")
        for ticker, count in stats['ticker_breakdown'].items():
            print(f"  {ticker}: {count} cache files")
    
    if stats['oldest_cache']:
        print(f"\nOldest Cache: {stats['oldest_cache'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    if stats['newest_cache']:
        print(f"Newest Cache: {stats['newest_cache'].strftime('%Y-%m-%d %H:%M:%S')}")

def clear_old_cache(older_than_days=30, ticker=None):
    """Clear cache files older than specified days."""
    detector = get_regime_detector()
    
    print(f"Clearing cache files older than {older_than_days} days...")
    if ticker:
        print(f"Filtering to ticker: {ticker}")
    
    detector.clear_market_cache(ticker=ticker, older_than_days=older_than_days)
    print("Cache cleanup completed")

def main():
    """Main cache management script."""
    parser = argparse.ArgumentParser(description='Market Data Cache Management')
    parser.add_argument('--stats', action='store_true', help='Show cache statistics')
    parser.add_argument('--init', action='store_true', help='Initialize cache with common tickers')
    parser.add_argument('--clear', type=int, help='Clear cache files older than N days')
    parser.add_argument('--ticker', help='Specific ticker to operate on')
    parser.add_argument('--days-back', type=int, default=365, help='Days of historical data to cache (default: 365)')
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Verify API key
    api_key = os.getenv('FINANCIAL_DATASETS_API_KEY')
    if not api_key:
        logger.error("FINANCIAL_DATASETS_API_KEY not found in .env file")
        logger.error("Please ensure your .env file contains: FINANCIAL_DATASETS_API_KEY=your_key_here")
        return 1
    
    logger.info(f"Using API key: {api_key[:8]}..." if len(api_key) >= 8 else "API key loaded")
    
    try:
        if args.stats:
            show_cache_stats()
        
        elif args.clear is not None:
            clear_old_cache(older_than_days=args.clear, ticker=args.ticker)
        
        elif args.init:
            # Common tickers for market data
            common_tickers = ['SPY', 'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'BRK.B']
            
            if args.ticker:
                tickers = [args.ticker]
            else:
                tickers = common_tickers
            
            success_count = init_cache_for_tickers(tickers, args.days_back)
            
            if success_count > 0:
                print(f"\n✓ Cache initialization completed successfully for {success_count} tickers")
                show_cache_stats()
            else:
                print("✗ Cache initialization failed")
                return 1
        
        else:
            # Default action: show stats
            show_cache_stats()
        
        return 0
        
    except Exception as e:
        logger.error(f"Cache management failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())