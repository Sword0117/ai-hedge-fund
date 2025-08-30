#!/usr/bin/env python3
"""
A/B Testing Results Comparison Script

Compares performance between LLM and ML ensemble portfolio managers
and generates statistical analysis of the differences.
"""

import sys
import json
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Graceful imports
try:
    import scipy.stats as stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not available - some statistical tests will be skipped")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("WARNING: matplotlib/seaborn not available - plots will be skipped")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ABTestAnalyzer:
    """
    Analyzes A/B test results between LLM and ML ensemble portfolio managers.
    """
    
    def __init__(self):
        self.llm_results = None
        self.ml_results = None
        self.comparison_metrics = {}
        
    def load_results(self, llm_file: str, ml_file: str) -> bool:
        """Load backtest results from JSON files"""
        try:
            with open(llm_file, 'r') as f:
                self.llm_results = json.load(f)
            logger.info(f"Loaded LLM results: {len(self.llm_results)} records")
            
            with open(ml_file, 'r') as f:
                self.ml_results = json.load(f)
            logger.info(f"Loaded ML results: {len(self.ml_results)} records")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading results: {e}")
            return False
    
    def calculate_performance_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Calculate performance metrics from backtest results"""
        if not results:
            return {}
        
        # Extract returns
        returns = [r.get('return', 0.0) for r in results if 'return' in r]
        
        if not returns:
            logger.warning("No return data found in results")
            return {}
        
        returns = np.array(returns)
        
        # Basic metrics
        total_return = np.prod(1 + returns) - 1
        avg_return = np.mean(returns)
        volatility = np.std(returns)
        sharpe_ratio = avg_return / (volatility + 1e-8) * np.sqrt(252)  # Annualized
        
        # Risk metrics
        max_drawdown = self._calculate_max_drawdown(returns)
        
        # Success rate
        winning_trades = np.sum(returns > 0)
        total_trades = len(returns)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Confidence analysis
        confidences = [r.get('confidence', 0) for r in results if 'confidence' in r]
        avg_confidence = np.mean(confidences) if confidences else 0
        
        return {
            'total_return': total_return,
            'avg_daily_return': avg_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_confidence': avg_confidence,
            'total_trades': total_trades,
            'winning_trades': winning_trades
        }
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown from returns"""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)
    
    def compare_performance(self) -> Dict[str, Any]:
        """Compare performance between LLM and ML systems"""
        logger.info("Comparing performance metrics...")
        
        # Calculate metrics for both systems
        llm_metrics = self.calculate_performance_metrics(self.llm_results)
        ml_metrics = self.calculate_performance_metrics(self.ml_results)
        
        if not llm_metrics or not ml_metrics:
            logger.error("Could not calculate metrics for comparison")
            return {}
        
        # Calculate differences
        comparison = {
            'llm_metrics': llm_metrics,
            'ml_metrics': ml_metrics,
            'differences': {},
            'statistical_tests': {},
            'analysis_date': datetime.now().isoformat()
        }
        
        # Calculate metric differences
        for metric in llm_metrics:
            if metric in ml_metrics:
                diff = ml_metrics[metric] - llm_metrics[metric]
                pct_diff = (diff / (llm_metrics[metric] + 1e-8)) * 100
                
                comparison['differences'][metric] = {
                    'absolute_diff': diff,
                    'percent_diff': pct_diff,
                    'ml_better': diff > 0 if 'drawdown' not in metric else diff < 0
                }
        
        # Statistical significance tests
        if HAS_SCIPY:
            comparison['statistical_tests'] = self._run_statistical_tests()
        
        self.comparison_metrics = comparison
        return comparison
    
    def _run_statistical_tests(self) -> Dict[str, Any]:
        """Run statistical significance tests"""
        logger.info("Running statistical significance tests...")
        
        # Extract returns for statistical tests
        llm_returns = np.array([r.get('return', 0.0) for r in self.llm_results if 'return' in r])
        ml_returns = np.array([r.get('return', 0.0) for r in self.ml_results if 'return' in r])
        
        tests = {}
        
        if len(llm_returns) > 0 and len(ml_returns) > 0:
            # T-test for difference in means
            try:
                t_stat, p_value = stats.ttest_ind(ml_returns, llm_returns, equal_var=False)
                tests['ttest_returns'] = {
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05,
                    'ml_better_mean': t_stat > 0
                }
            except Exception as e:
                logger.warning(f"T-test failed: {e}")
            
            # Mann-Whitney U test (non-parametric)
            try:
                u_stat, p_value = stats.mannwhitneyu(ml_returns, llm_returns, alternative='two-sided')
                tests['mann_whitney'] = {
                    'u_statistic': float(u_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05
                }
            except Exception as e:
                logger.warning(f"Mann-Whitney test failed: {e}")
            
            # Kolmogorov-Smirnov test for distribution differences
            try:
                ks_stat, p_value = stats.ks_2samp(ml_returns, llm_returns)
                tests['kolmogorov_smirnov'] = {
                    'ks_statistic': float(ks_stat),
                    'p_value': float(p_value),
                    'distributions_different': p_value < 0.05
                }
            except Exception as e:
                logger.warning(f"KS test failed: {e}")
        
        return tests
    
    def generate_report(self, output_file: str = None) -> str:
        """Generate comprehensive comparison report"""
        logger.info("Generating comparison report...")
        
        if not self.comparison_metrics:
            self.compare_performance()
        
        report = self._generate_text_report()
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {output_file}")
        
        return report
    
    def _generate_text_report(self) -> str:
        """Generate text-based comparison report"""
        if not self.comparison_metrics:
            return "No comparison data available"
        
        llm = self.comparison_metrics['llm_metrics']
        ml = self.comparison_metrics['ml_metrics']
        diffs = self.comparison_metrics['differences']
        
        report = []
        report.append("=" * 80)
        report.append("AI HEDGE FUND - PHASE 2 A/B TEST RESULTS")
        report.append("=" * 80)
        report.append(f"Analysis Date: {self.comparison_metrics['analysis_date']}")
        report.append(f"LLM Trades: {llm.get('total_trades', 0)}")
        report.append(f"ML Trades: {ml.get('total_trades', 0)}")
        report.append("")
        
        # Performance comparison table
        report.append("PERFORMANCE COMPARISON")
        report.append("-" * 50)
        report.append(f"{'Metric':<20} {'LLM':<12} {'ML':<12} {'Difference':<15} {'Winner'}")
        report.append("-" * 70)
        
        key_metrics = [
            ('Total Return', 'total_return', '{:.2%}'),
            ('Sharpe Ratio', 'sharpe_ratio', '{:.3f}'),
            ('Win Rate', 'win_rate', '{:.2%}'),
            ('Max Drawdown', 'max_drawdown', '{:.2%}'),
            ('Avg Confidence', 'avg_confidence', '{:.1f}'),
            ('Volatility', 'volatility', '{:.2%}')
        ]
        
        for display_name, metric_key, fmt in key_metrics:
            if metric_key in llm and metric_key in ml:
                llm_val = llm[metric_key]
                ml_val = ml[metric_key]
                diff_info = diffs.get(metric_key, {})
                
                llm_str = fmt.format(llm_val)
                ml_str = fmt.format(ml_val)
                diff_str = f"{diff_info.get('percent_diff', 0):+.1f}%"
                winner = "ML" if diff_info.get('ml_better', False) else "LLM"
                
                report.append(f"{display_name:<20} {llm_str:<12} {ml_str:<12} {diff_str:<15} {winner}")
        
        # Statistical significance
        if 'statistical_tests' in self.comparison_metrics:
            report.append("")
            report.append("STATISTICAL SIGNIFICANCE TESTS")
            report.append("-" * 40)
            
            tests = self.comparison_metrics['statistical_tests']
            
            if 'ttest_returns' in tests:
                ttest = tests['ttest_returns']
                significance = "YES" if ttest['significant'] else "NO"
                report.append(f"T-test (returns difference): p={ttest['p_value']:.4f}, Significant: {significance}")
            
            if 'mann_whitney' in tests:
                mw = tests['mann_whitney']
                significance = "YES" if mw['significant'] else "NO"
                report.append(f"Mann-Whitney U (distribution): p={mw['p_value']:.4f}, Significant: {significance}")
        
        # Summary and recommendations
        report.append("")
        report.append("SUMMARY & RECOMMENDATIONS")
        report.append("-" * 30)
        
        ml_wins = sum(1 for diff in diffs.values() if diff.get('ml_better', False))
        total_metrics = len(diffs)
        
        if ml_wins > total_metrics * 0.6:
            report.append("✅ RECOMMENDATION: Deploy ML Ensemble")
            report.append(f"   ML outperforms on {ml_wins}/{total_metrics} key metrics")
        elif ml_wins < total_metrics * 0.4:
            report.append("✅ RECOMMENDATION: Continue with LLM")
            report.append(f"   LLM outperforms on {total_metrics - ml_wins}/{total_metrics} key metrics")
        else:
            report.append("⚖️  RECOMMENDATION: Extended Testing Needed")
            report.append(f"   Mixed results: ML wins {ml_wins}/{total_metrics} metrics")
        
        # Add regime-specific analysis if available
        report.append("")
        report.append("REGIME-SPECIFIC PERFORMANCE")
        report.append("-" * 30)
        report.append("(Analyze by market conditions - bull/bear/neutral)")
        report.append("(Analyze by volatility - high/low)")
        report.append("(Consider implementing hybrid approach)")
        
        return "\n".join(report)
    
    def plot_comparison(self, output_dir: str = "plots") -> None:
        """Generate comparison plots"""
        if not HAS_PLOTTING:
            logger.warning("Plotting libraries not available")
            return
        
        logger.info("Generating comparison plots...")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Returns comparison
        self._plot_returns_comparison(output_path)
        
        # Performance metrics radar chart
        self._plot_performance_radar(output_path)
        
        logger.info(f"Plots saved to {output_path}")
    
    def _plot_returns_comparison(self, output_path: Path) -> None:
        """Plot returns comparison"""
        try:
            llm_returns = [r.get('return', 0.0) for r in self.llm_results if 'return' in r]
            ml_returns = [r.get('return', 0.0) for r in self.ml_results if 'return' in r]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Cumulative returns
            llm_cumulative = np.cumprod(1 + np.array(llm_returns))
            ml_cumulative = np.cumprod(1 + np.array(ml_returns))
            
            ax1.plot(llm_cumulative, label='LLM', alpha=0.8)
            ax1.plot(ml_cumulative, label='ML Ensemble', alpha=0.8)
            ax1.set_title('Cumulative Returns Comparison')
            ax1.set_xlabel('Trading Days')
            ax1.set_ylabel('Cumulative Return')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Returns distribution
            ax2.hist(llm_returns, bins=30, alpha=0.7, label='LLM', density=True)
            ax2.hist(ml_returns, bins=30, alpha=0.7, label='ML Ensemble', density=True)
            ax2.set_title('Daily Returns Distribution')
            ax2.set_xlabel('Daily Return')
            ax2.set_ylabel('Density')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path / 'returns_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting returns comparison: {e}")
    
    def _plot_performance_radar(self, output_path: Path) -> None:
        """Plot performance metrics radar chart"""
        try:
            if not self.comparison_metrics:
                return
            
            llm = self.comparison_metrics['llm_metrics']
            ml = self.comparison_metrics['ml_metrics']
            
            # Normalize metrics for radar chart (0-1 scale)
            metrics = ['sharpe_ratio', 'win_rate', 'avg_confidence']
            metric_labels = ['Sharpe Ratio', 'Win Rate', 'Avg Confidence']
            
            llm_values = []
            ml_values = []
            
            for metric in metrics:
                llm_val = abs(llm.get(metric, 0))
                ml_val = abs(ml.get(metric, 0))
                max_val = max(llm_val, ml_val, 1e-8)
                
                llm_values.append(llm_val / max_val)
                ml_values.append(ml_val / max_val)
            
            # Add drawdown (inverted - lower is better)
            if 'max_drawdown' in llm and 'max_drawdown' in ml:
                llm_dd = abs(llm['max_drawdown'])
                ml_dd = abs(ml['max_drawdown'])
                max_dd = max(llm_dd, ml_dd, 1e-8)
                
                # Invert for radar chart (higher is better)
                llm_values.append(1 - (llm_dd / max_dd))
                ml_values.append(1 - (ml_dd / max_dd))
                metric_labels.append('Drawdown Control')
            
            # Create radar chart
            angles = np.linspace(0, 2 * np.pi, len(metric_labels), endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle
            
            llm_values += llm_values[:1]
            ml_values += ml_values[:1]
            
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            ax.plot(angles, llm_values, 'o-', linewidth=2, label='LLM', alpha=0.8)
            ax.fill(angles, llm_values, alpha=0.25)
            
            ax.plot(angles, ml_values, 'o-', linewidth=2, label='ML Ensemble', alpha=0.8)
            ax.fill(angles, ml_values, alpha=0.25)
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metric_labels)
            ax.set_ylim(0, 1)
            ax.set_title('Performance Metrics Comparison\n(Normalized Scale)', size=16, pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
            ax.grid(True)
            
            plt.tight_layout()
            plt.savefig(output_path / 'performance_radar.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting performance radar: {e}")


def main():
    """Main comparison script"""
    parser = argparse.ArgumentParser(description='Compare A/B test results between LLM and ML ensemble')
    parser.add_argument('--llm-results', required=True, help='Path to LLM backtest results JSON')
    parser.add_argument('--ml-results', required=True, help='Path to ML ensemble backtest results JSON')
    parser.add_argument('--output-report', default='reports/ab_comparison.txt', help='Output report file')
    parser.add_argument('--plot-dir', default='plots', help='Directory for plots')
    parser.add_argument('--no-plots', action='store_true', help='Skip generating plots')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = ABTestAnalyzer()
    
    # Load results
    if not analyzer.load_results(args.llm_results, args.ml_results):
        print("Failed to load results")
        return 1
    
    # Run comparison
    comparison = analyzer.compare_performance()
    
    if not comparison:
        print("Failed to generate comparison")
        return 1
    
    # Generate report
    report_path = Path(args.output_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    report = analyzer.generate_report(str(report_path))
    print(report)
    
    # Generate plots
    if not args.no_plots:
        analyzer.plot_comparison(args.plot_dir)
    
    print(f"\nComparison completed. Report saved to: {report_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())