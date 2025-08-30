#!/usr/bin/env python3
"""
Agent Performance Tracking System - Phase 2 Implementation

Tracks historical performance of individual agents per ticker, calculates
dynamic weights based on accuracy and confidence calibration, and provides
performance-based signal weighting for the ML ensemble system.
"""

import json
import numpy as np
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import gzip

logger = logging.getLogger(__name__)


@dataclass
class AgentPerformanceMetric:
    """Single performance measurement for an agent on a specific date"""
    date: str
    ticker: str
    agent_name: str
    signal: str  # bullish, bearish, neutral
    confidence: float
    actual_return: float
    accuracy: float  # 1.0 if prediction correct, 0.0 if wrong
    return_contribution: float  # actual return if accurate, negative if inaccurate
    market_regime: str  # bull, bear, neutral
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentPerformanceMetric':
        """Create from dictionary"""
        return cls(**data)


@dataclass  
class AgentPerformanceSummary:
    """Summary statistics for an agent's performance"""
    agent_name: str
    ticker: str
    total_predictions: int
    accuracy: float
    avg_confidence: float
    confidence_calibration: float  # How well confidence predicts accuracy
    return_contribution: float
    hit_rate_by_regime: Dict[str, float]
    sharpe_ratio: float
    last_updated: str


class AgentPerformanceTracker:
    """
    Tracks and analyzes agent performance for dynamic weighting in ensemble system.
    
    Features:
    - Per-ticker performance tracking
    - Exponentially weighted performance calculation
    - Confidence calibration analysis  
    - Regime-aware performance metrics
    - Efficient JSON storage with compression
    - Real-time weight calculation
    """
    
    def __init__(self, storage_dir: str = "data/agent_performance"):
        """
        Initialize performance tracker.
        
        Args:
            storage_dir: Directory to store performance data
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        self.performance_window = 60  # days to consider for weight calculation
        self.min_predictions_for_weight = 5  # minimum predictions before trusting weights
        self.decay_rate = 0.05  # exponential decay rate for time weighting
        self.confidence_bins = 10  # bins for confidence calibration analysis
        
        # In-memory cache
        self.metrics_cache = {}  # {ticker: [AgentPerformanceMetric]}
        self.summary_cache = {}  # {(ticker, agent): AgentPerformanceSummary}
        self.cache_last_updated = {}  # {ticker: datetime}
        
        # Agent names (will be populated as we see agents)
        self.agent_names = set()
        
        logger.info(f"AgentPerformanceTracker initialized with storage at {storage_dir}")
    
    def update_performance(
        self, 
        ticker: str, 
        date: datetime, 
        agent_signals: Dict[str, Dict[str, Any]], 
        actual_return: float,
        market_regime: str = "neutral"
    ) -> None:
        """
        Update performance metrics for all agents based on actual returns.
        
        Args:
            ticker: Stock ticker symbol
            date: Date of the prediction
            agent_signals: Dict of {agent_name: {signal, confidence, reasoning, ...}}
            actual_return: Actual return achieved (positive for gains, negative for losses)
            market_regime: Current market regime (bull/bear/neutral)
        """
        try:
            date_str = date.strftime('%Y-%m-%d')
            new_metrics = []
            
            for agent_name, signal_data in agent_signals.items():
                if not isinstance(signal_data, dict) or 'signal' not in signal_data:
                    continue
                
                signal = signal_data['signal']
                confidence = signal_data.get('confidence', 0.0)
                
                # Calculate accuracy based on signal vs actual return
                accuracy = self._calculate_accuracy(signal, actual_return)
                
                # Calculate return contribution
                return_contribution = actual_return if accuracy > 0.5 else -abs(actual_return)
                
                # Create performance metric
                metric = AgentPerformanceMetric(
                    date=date_str,
                    ticker=ticker,
                    agent_name=agent_name,
                    signal=signal,
                    confidence=confidence,
                    actual_return=actual_return,
                    accuracy=accuracy,
                    return_contribution=return_contribution,
                    market_regime=market_regime
                )
                
                new_metrics.append(metric)
                self.agent_names.add(agent_name)
            
            # Add to cache
            if ticker not in self.metrics_cache:
                self.metrics_cache[ticker] = []
            
            self.metrics_cache[ticker].extend(new_metrics)
            self.cache_last_updated[ticker] = datetime.now()
            
            # Persist to disk
            self._save_metrics(ticker, new_metrics)
            
            # Update summaries
            self._update_summaries(ticker)
            
            logger.debug(f"Updated performance for {len(new_metrics)} agents on {ticker} for {date_str}")
            
        except Exception as e:
            logger.error(f"Error updating performance for {ticker}: {e}")
    
    def calculate_agent_weights(
        self, 
        ticker: str, 
        current_date: datetime
    ) -> Dict[str, float]:
        """
        Calculate performance-based weights for agents on a specific ticker.
        
        Uses exponentially weighted accuracy with confidence calibration adjustments.
        More recent performance has higher weight in the calculation.
        
        Args:
            ticker: Stock ticker symbol
            current_date: Current date for time-based weighting
            
        Returns:
            Dict of {agent_name: weight} where weights sum to 1.0
        """
        try:
            # Load recent performance history
            history = self._get_performance_history(ticker, current_date, self.performance_window)
            
            if not history:
                # No history available - use equal weights
                if self.agent_names:
                    equal_weight = 1.0 / len(self.agent_names)
                    return {agent: equal_weight for agent in self.agent_names}
                else:
                    # Default to common agents
                    return {
                        'warren_buffett_agent': 0.33,
                        'technical_analyst_agent': 0.33,
                        'sentiment_agent': 0.34
                    }
            
            # Group metrics by agent
            agent_metrics = defaultdict(list)
            for metric in history:
                agent_metrics[metric.agent_name].append(metric)
            
            # Calculate weights for each agent
            agent_scores = {}
            
            for agent_name, metrics in agent_metrics.items():
                if len(metrics) < self.min_predictions_for_weight:
                    # Not enough data - use default weight
                    agent_scores[agent_name] = 0.5
                    continue
                
                # Sort metrics by date (most recent first)
                metrics.sort(key=lambda m: m.date, reverse=True)
                
                # Calculate exponentially weighted performance
                weighted_accuracy = 0.0
                weighted_confidence_calibration = 0.0
                total_weight = 0.0
                
                for i, metric in enumerate(metrics):
                    # Time-based weight (recent performance matters more)
                    time_weight = np.exp(-self.decay_rate * i)
                    
                    # Accuracy contribution
                    accuracy_contribution = metric.accuracy * time_weight
                    weighted_accuracy += accuracy_contribution
                    
                    # Confidence calibration (how well confidence predicts accuracy)
                    expected_accuracy = metric.confidence / 100.0
                    calibration_error = abs(expected_accuracy - metric.accuracy)
                    calibration_score = 1.0 - calibration_error
                    weighted_confidence_calibration += calibration_score * time_weight
                    
                    total_weight += time_weight
                
                if total_weight > 0:
                    # Normalize weighted scores
                    avg_accuracy = weighted_accuracy / total_weight
                    avg_calibration = weighted_confidence_calibration / total_weight
                    
                    # Combined score (accuracy + confidence calibration)
                    combined_score = 0.7 * avg_accuracy + 0.3 * avg_calibration
                    agent_scores[agent_name] = max(0.01, combined_score)  # Minimum weight of 1%
                else:
                    agent_scores[agent_name] = 0.5
            
            # Normalize weights to sum to 1.0
            total_score = sum(agent_scores.values())
            if total_score > 0:
                normalized_weights = {
                    agent: score / total_score 
                    for agent, score in agent_scores.items()
                }
            else:
                # Fallback to equal weights
                equal_weight = 1.0 / len(agent_scores) if agent_scores else 0.33
                normalized_weights = {agent: equal_weight for agent in agent_scores}
            
            logger.debug(f"Calculated weights for {ticker}: {normalized_weights}")
            return normalized_weights
            
        except Exception as e:
            logger.error(f"Error calculating agent weights for {ticker}: {e}")
            # Return default weights
            return {
                'warren_buffett_agent': 0.33,
                'technical_analyst_agent': 0.33,
                'sentiment_agent': 0.34
            }
    
    def apply_weights(
        self, 
        agent_signals: Dict[str, Dict[str, Any]], 
        ticker: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        Apply performance-based weights to agent signals.
        
        Args:
            agent_signals: Dict of {agent_name: {signal, confidence, reasoning}}
            ticker: Stock ticker symbol
            
        Returns:
            Dict with added 'weight' field for each agent
        """
        try:
            # Get current weights
            weights = self.calculate_agent_weights(ticker, datetime.now())
            
            # Add weights to signals
            weighted_signals = {}
            for agent_name, signal_data in agent_signals.items():
                if isinstance(signal_data, dict):
                    weighted_signal = signal_data.copy()
                    weighted_signal['weight'] = weights.get(agent_name, 0.1)  # Default low weight
                    weighted_signals[agent_name] = weighted_signal
                else:
                    weighted_signals[agent_name] = signal_data
            
            return weighted_signals
            
        except Exception as e:
            logger.error(f"Error applying weights to signals for {ticker}: {e}")
            # Return signals unchanged
            return agent_signals
    
    def get_agent_summary(
        self, 
        ticker: str, 
        agent_name: str
    ) -> Optional[AgentPerformanceSummary]:
        """Get performance summary for a specific agent and ticker"""
        try:
            cache_key = (ticker, agent_name)
            
            # Check cache first
            if cache_key in self.summary_cache:
                return self.summary_cache[cache_key]
            
            # Load and calculate summary
            history = self._get_performance_history(ticker, datetime.now(), 365)  # 1 year
            agent_metrics = [m for m in history if m.agent_name == agent_name]
            
            if not agent_metrics:
                return None
            
            # Calculate summary statistics
            total_predictions = len(agent_metrics)
            accuracy = np.mean([m.accuracy for m in agent_metrics])
            avg_confidence = np.mean([m.confidence for m in agent_metrics])
            
            # Confidence calibration
            confidence_calibration = self._calculate_confidence_calibration(agent_metrics)
            
            # Return contribution
            return_contribution = np.sum([m.return_contribution for m in agent_metrics])
            
            # Hit rate by regime
            regime_metrics = defaultdict(list)
            for metric in agent_metrics:
                regime_metrics[metric.market_regime].append(metric.accuracy)
            
            hit_rate_by_regime = {
                regime: np.mean(accuracies) 
                for regime, accuracies in regime_metrics.items()
            }
            
            # Sharpe ratio (simplified - return contribution / volatility)
            returns = [m.return_contribution for m in agent_metrics]
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) if len(returns) > 1 else 0.0
            
            summary = AgentPerformanceSummary(
                agent_name=agent_name,
                ticker=ticker,
                total_predictions=total_predictions,
                accuracy=accuracy,
                avg_confidence=avg_confidence,
                confidence_calibration=confidence_calibration,
                return_contribution=return_contribution,
                hit_rate_by_regime=hit_rate_by_regime,
                sharpe_ratio=sharpe_ratio,
                last_updated=datetime.now().isoformat()
            )
            
            # Cache summary
            self.summary_cache[cache_key] = summary
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting agent summary for {agent_name} on {ticker}: {e}")
            return None
    
    def get_performance_report(self, ticker: str = None) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.
        
        Args:
            ticker: Specific ticker to report on, or None for all tickers
            
        Returns:
            Dict with performance statistics and rankings
        """
        try:
            report = {
                'generated_at': datetime.now().isoformat(),
                'ticker': ticker,
                'agent_summaries': {},
                'rankings': {},
                'overall_stats': {}
            }
            
            if ticker:
                # Report for specific ticker
                tickers_to_process = [ticker]
            else:
                # Report for all tickers
                tickers_to_process = list(self.metrics_cache.keys())
                if not tickers_to_process:
                    tickers_to_process = [f for f in self.storage_dir.glob('*.json.gz')]
                    tickers_to_process = [f.stem.replace('.json', '') for f in tickers_to_process]
            
            all_summaries = []
            
            for tick in tickers_to_process:
                for agent_name in self.agent_names:
                    summary = self.get_agent_summary(tick, agent_name)
                    if summary:
                        report['agent_summaries'][f"{tick}_{agent_name}"] = asdict(summary)
                        all_summaries.append(summary)
            
            # Generate rankings
            if all_summaries:
                # Rank by accuracy
                accuracy_ranking = sorted(
                    all_summaries, 
                    key=lambda s: s.accuracy, 
                    reverse=True
                )
                
                # Rank by Sharpe ratio
                sharpe_ranking = sorted(
                    all_summaries,
                    key=lambda s: s.sharpe_ratio,
                    reverse=True
                )
                
                # Rank by return contribution
                return_ranking = sorted(
                    all_summaries,
                    key=lambda s: s.return_contribution,
                    reverse=True
                )
                
                report['rankings'] = {
                    'by_accuracy': [(s.agent_name, s.ticker, s.accuracy) for s in accuracy_ranking[:10]],
                    'by_sharpe_ratio': [(s.agent_name, s.ticker, s.sharpe_ratio) for s in sharpe_ranking[:10]], 
                    'by_return_contribution': [(s.agent_name, s.ticker, s.return_contribution) for s in return_ranking[:10]]
                }
                
                # Overall statistics
                report['overall_stats'] = {
                    'total_agents': len(self.agent_names),
                    'total_tickers': len(tickers_to_process),
                    'avg_accuracy': np.mean([s.accuracy for s in all_summaries]),
                    'avg_confidence_calibration': np.mean([s.confidence_calibration for s in all_summaries]),
                    'total_return_contribution': np.sum([s.return_contribution for s in all_summaries])
                }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {'error': str(e)}
    
    def _calculate_accuracy(self, signal: str, actual_return: float) -> float:
        """
        Calculate accuracy score based on signal vs actual return.
        
        Returns 1.0 for correct predictions, 0.0 for incorrect, 0.5 for neutral
        """
        if signal == "bullish":
            return 1.0 if actual_return > 0 else 0.0
        elif signal == "bearish":  
            return 1.0 if actual_return < 0 else 0.0
        elif signal == "neutral":
            # Neutral is considered correct if return is within +/-2%
            return 1.0 if abs(actual_return) < 0.02 else 0.5
        else:
            return 0.5  # Unknown signal type
    
    def _calculate_confidence_calibration(self, metrics: List[AgentPerformanceMetric]) -> float:
        """
        Calculate how well agent confidence scores predict actual accuracy.
        
        Perfect calibration = 1.0, random calibration = 0.0
        """
        if len(metrics) < 5:
            return 0.5  # Not enough data
        
        try:
            # Bin metrics by confidence level
            confidence_bins = np.linspace(0, 100, self.confidence_bins + 1)
            calibration_errors = []
            
            for i in range(len(confidence_bins) - 1):
                bin_start = confidence_bins[i]
                bin_end = confidence_bins[i + 1]
                
                # Find metrics in this confidence bin
                bin_metrics = [
                    m for m in metrics 
                    if bin_start <= m.confidence < bin_end
                ]
                
                if len(bin_metrics) < 2:
                    continue
                
                # Calculate expected vs actual accuracy for this bin
                expected_accuracy = np.mean([m.confidence / 100.0 for m in bin_metrics])
                actual_accuracy = np.mean([m.accuracy for m in bin_metrics])
                
                # Calibration error for this bin
                bin_error = abs(expected_accuracy - actual_accuracy)
                calibration_errors.append(bin_error)
            
            if calibration_errors:
                # Overall calibration score (1 - mean calibration error)
                mean_calibration_error = np.mean(calibration_errors)
                calibration_score = max(0.0, 1.0 - mean_calibration_error)
                return calibration_score
            else:
                return 0.5  # No valid bins
                
        except Exception as e:
            logger.error(f"Error calculating confidence calibration: {e}")
            return 0.5
    
    def _get_performance_history(
        self, 
        ticker: str, 
        end_date: datetime, 
        days: int
    ) -> List[AgentPerformanceMetric]:
        """Load performance history for a ticker within date range"""
        try:
            # Check cache first
            if (ticker in self.cache_last_updated and 
                (datetime.now() - self.cache_last_updated[ticker]).seconds < 300):  # 5 minute cache
                cached_metrics = self.metrics_cache.get(ticker, [])
                
                # Filter by date range
                start_date = end_date - timedelta(days=days)
                filtered_metrics = [
                    m for m in cached_metrics
                    if start_date <= datetime.strptime(m.date, '%Y-%m-%d') <= end_date
                ]
                return filtered_metrics
            
            # Load from disk
            metrics = self._load_metrics(ticker)
            
            # Filter by date range
            start_date = end_date - timedelta(days=days)
            filtered_metrics = [
                m for m in metrics
                if start_date <= datetime.strptime(m.date, '%Y-%m-%d') <= end_date
            ]
            
            return filtered_metrics
            
        except Exception as e:
            logger.error(f"Error loading performance history for {ticker}: {e}")
            return []
    
    def _load_metrics(self, ticker: str) -> List[AgentPerformanceMetric]:
        """Load metrics from disk for a specific ticker"""
        try:
            filename = self.storage_dir / f"{ticker}.json.gz"
            
            if not filename.exists():
                return []
            
            with gzip.open(filename, 'rt') as f:
                data = json.load(f)
            
            metrics = [AgentPerformanceMetric.from_dict(item) for item in data]
            
            # Update cache
            self.metrics_cache[ticker] = metrics
            self.cache_last_updated[ticker] = datetime.now()
            
            # Update agent names
            for metric in metrics:
                self.agent_names.add(metric.agent_name)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error loading metrics from disk for {ticker}: {e}")
            return []
    
    def _save_metrics(self, ticker: str, new_metrics: List[AgentPerformanceMetric]) -> None:
        """Save new metrics to disk (append mode)"""
        try:
            filename = self.storage_dir / f"{ticker}.json.gz"
            
            # Load existing metrics
            existing_metrics = []
            if filename.exists():
                existing_metrics = self._load_metrics(ticker)
            
            # Combine and deduplicate
            all_metrics = existing_metrics + new_metrics
            
            # Remove duplicates (same agent, ticker, date)
            seen = set()
            unique_metrics = []
            for metric in all_metrics:
                key = (metric.agent_name, metric.ticker, metric.date)
                if key not in seen:
                    seen.add(key)
                    unique_metrics.append(metric)
            
            # Sort by date (newest first)
            unique_metrics.sort(key=lambda m: m.date, reverse=True)
            
            # Keep only recent metrics (e.g., last 2 years)
            cutoff_date = datetime.now() - timedelta(days=730)
            filtered_metrics = [
                m for m in unique_metrics
                if datetime.strptime(m.date, '%Y-%m-%d') >= cutoff_date
            ]
            
            # Save to compressed JSON
            with gzip.open(filename, 'wt') as f:
                json.dump([m.to_dict() for m in filtered_metrics], f, indent=2)
            
            logger.debug(f"Saved {len(filtered_metrics)} metrics for {ticker}")
            
        except Exception as e:
            logger.error(f"Error saving metrics to disk for {ticker}: {e}")
    
    def _update_summaries(self, ticker: str) -> None:
        """Update cached summaries for a ticker"""
        try:
            # Clear cache for this ticker
            keys_to_remove = [key for key in self.summary_cache.keys() if key[0] == ticker]
            for key in keys_to_remove:
                del self.summary_cache[key]
            
            # Summaries will be regenerated on next access
            
        except Exception as e:
            logger.error(f"Error updating summaries for {ticker}: {e}")
    
    def cleanup_old_data(self, days_to_keep: int = 730) -> None:
        """Clean up old performance data to save space"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            cleaned_count = 0
            
            for ticker_file in self.storage_dir.glob("*.json.gz"):
                ticker = ticker_file.stem.replace('.json', '')
                
                # Load metrics
                metrics = self._load_metrics(ticker)
                
                # Filter recent metrics
                recent_metrics = [
                    m for m in metrics
                    if datetime.strptime(m.date, '%Y-%m-%d') >= cutoff_date
                ]
                
                if len(recent_metrics) < len(metrics):
                    # Save filtered metrics
                    with gzip.open(ticker_file, 'wt') as f:
                        json.dump([m.to_dict() for m in recent_metrics], f)
                    
                    cleaned_count += len(metrics) - len(recent_metrics)
            
            logger.info(f"Cleaned up {cleaned_count} old performance records")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


if __name__ == "__main__":
    # Test the performance tracker
    print("Testing Agent Performance Tracker...")
    
    tracker = AgentPerformanceTracker("test_data/performance")
    
    # Simulate some performance updates
    test_signals = {
        'warren_buffett_agent': {'signal': 'bullish', 'confidence': 75.0},
        'technical_analyst_agent': {'signal': 'bearish', 'confidence': 60.0},
        'sentiment_agent': {'signal': 'neutral', 'confidence': 50.0}
    }
    
    # Update performance (simulate a winning trade)
    tracker.update_performance(
        ticker="AAPL",
        date=datetime.now(),
        agent_signals=test_signals,
        actual_return=0.05,  # 5% gain
        market_regime="bull"
    )
    
    # Calculate weights
    weights = tracker.calculate_agent_weights("AAPL", datetime.now())
    print(f"Calculated weights: {weights}")
    
    # Apply weights to signals
    weighted_signals = tracker.apply_weights(test_signals, "AAPL")
    print(f"Weighted signals: {weighted_signals}")
    
    # Generate performance report
    report = tracker.get_performance_report("AAPL")
    print(f"Performance report: {json.dumps(report, indent=2)}")