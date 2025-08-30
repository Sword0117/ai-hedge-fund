#!/usr/bin/env python3
"""
Enhanced Signal Fusion System - Phase 2 Implementation

ML-based ensemble system that combines agent signals and dynamically weights agents
based on their historical performance. Operates in parallel with LLM portfolio manager
for A/B testing comparison.
"""

import numpy as np
import pandas as pd
import logging
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import joblib

from pydantic import BaseModel, Field
from typing_extensions import Literal

# Graceful imports for ML libraries
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, log_loss
    from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logging.warning("scikit-learn not available - ML ensemble will use fallback mode")

# Import existing portfolio decision format
from src.agents.portfolio_manager import PortfolioDecision, PortfolioManagerOutput
from src.agents.regime_detector import get_current_market_regime, MarketRegime

logger = logging.getLogger(__name__)


@dataclass
class AgentSignal:
    """Standardized agent signal structure"""
    agent_name: str
    signal: str  # bullish, bearish, neutral
    confidence: float
    reasoning: str
    timestamp: datetime


@dataclass
class MarketContext:
    """Market context features for ensemble models"""
    regime: MarketRegime
    sector: str
    market_cap: float
    volatility: float
    day_of_week: int
    month: int
    days_to_earnings: int


class SignalFeatures(BaseModel):
    """Feature vector for ML ensemble models"""
    # Agent signals (one-hot encoded)
    warren_buffett_bullish: float = 0.0
    warren_buffett_bearish: float = 0.0
    warren_buffett_confidence: float = 0.0
    
    technical_analyst_bullish: float = 0.0
    technical_analyst_bearish: float = 0.0
    technical_analyst_confidence: float = 0.0
    
    # Additional agents (can be expanded)
    sentiment_bullish: float = 0.0
    sentiment_bearish: float = 0.0
    sentiment_confidence: float = 0.0
    
    # Interaction features
    agent_agreement: float = 0.0  # How much agents agree
    confidence_variance: float = 0.0  # Variance in confidence scores
    bullish_consensus: float = 0.0  # Fraction of agents that are bullish
    
    # Market context
    regime_bull: float = 0.0
    regime_bear: float = 0.0
    regime_volatility_high: float = 0.0
    regime_structure_trending: float = 0.0
    
    # Temporal features
    day_of_week: float = 0.0
    month: float = 0.0
    
    # Performance-based weights (calculated by AgentPerformanceTracker)
    warren_buffett_weight: float = 0.33
    technical_analyst_weight: float = 0.33
    sentiment_weight: float = 0.33


class SignalEnsemble:
    """
    ML Ensemble system for combining agent signals with dynamic weighting.
    
    Uses regime-specific models and performance-based agent weighting to generate
    trading decisions that can be A/B tested against the LLM portfolio manager.
    """
    
    def __init__(self, regime_detector=None, models_dir: str = "data/models"):
        """
        Initialize the Signal Ensemble system.
        
        Args:
            regime_detector: Market regime detection system
            models_dir: Directory to store trained models
        """
        self.regime_detector = regime_detector
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Import performance tracker here to avoid circular imports
        from src.agents.performance_tracker import AgentPerformanceTracker
        self.agent_weights = AgentPerformanceTracker()
        
        # Initialize models
        if HAS_SKLEARN:
            self.models = {
                'bull': GradientBoostingClassifier(
                    n_estimators=100, 
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                ),
                'bear': GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                ),
                'neutral': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=5,
                    random_state=42
                )
            }
            self.scaler = StandardScaler()
            self.is_trained = False
        else:
            self.models = None
            self.scaler = None
            self.is_trained = False
            logger.warning("SignalEnsemble initialized without scikit-learn - using heuristic fallback")
        
        # Feature column names for consistency
        self.feature_columns = list(SignalFeatures().model_fields.keys())
        
        # Model metadata
        self.model_version = "1.0"
        self.last_training_date = None
        self.training_samples = 0
        
        logger.info(f"SignalEnsemble initialized with {'ML models' if HAS_SKLEARN else 'heuristic fallback'}")

    def generate_decision(
        self, 
        agent_signals: Dict[str, Dict], 
        ticker: str, 
        current_date: datetime,
        market_context: Optional[Dict] = None
    ) -> PortfolioDecision:
        """
        Generate trading decision using ML ensemble or heuristic fallback.
        
        Args:
            agent_signals: Dict of {agent_name: {signal, confidence, reasoning}}
            ticker: Stock ticker symbol
            current_date: Current trading date
            market_context: Additional market context data
            
        Returns:
            PortfolioDecision compatible with existing portfolio manager
        """
        try:
            # Get current market regime
            regime = self._get_market_regime(current_date)
            
            # Apply performance-based weights to agent signals
            weighted_signals = self.agent_weights.apply_weights(agent_signals, ticker)
            
            # Engineer features from signals
            features = self._engineer_features(weighted_signals, regime, ticker, market_context)
            
            if HAS_SKLEARN and self.is_trained:
                # Use ML ensemble
                decision = self._ml_predict(features, regime, weighted_signals)
            else:
                # Use heuristic fallback
                decision = self._heuristic_predict(weighted_signals, regime)
            
            # Add ensemble metadata to reasoning
            decision.reasoning = f"[ML Ensemble] {decision.reasoning}"
            
            logger.debug(f"Generated decision for {ticker}: {decision.action} (confidence: {decision.confidence:.1f})")
            return decision
            
        except Exception as e:
            logger.error(f"Error generating ensemble decision for {ticker}: {e}")
            # Return safe default
            return PortfolioDecision(
                action="hold",
                quantity=0,
                confidence=0.0,
                reasoning=f"Ensemble error - defaulting to hold: {str(e)}"
            )

    def _get_market_regime(self, current_date: datetime) -> MarketRegime:
        """Get current market regime with fallback"""
        try:
            if self.regime_detector:
                return self.regime_detector.get_current_regime(current_date.strftime('%Y-%m-%d'))
            else:
                # Use the global regime detection function
                return get_current_market_regime(current_date.strftime('%Y-%m-%d'))
        except Exception as e:
            logger.warning(f"Failed to get market regime: {e}")
            # Return default neutral regime
            from src.agents.regime_detector import MarketRegime
            return MarketRegime(
                market_state="neutral",
                volatility="low",
                structure="mean_reverting",
                confidence=0.3,
                timestamp=current_date
            )

    def _engineer_features(
        self, 
        agent_signals: Dict[str, Dict], 
        regime: MarketRegime, 
        ticker: str,
        market_context: Optional[Dict] = None
    ) -> SignalFeatures:
        """
        Engineer features from agent signals and market context.
        
        Creates a comprehensive feature vector including:
        - One-hot encoded agent signals
        - Confidence scores
        - Interaction features (agreement, variance)
        - Market regime features
        - Temporal features
        - Performance-based agent weights
        """
        features = SignalFeatures()
        
        # Extract agent signals and confidences
        signals = {}
        confidences = {}
        
        for agent_name, signal_data in agent_signals.items():
            if isinstance(signal_data, dict) and 'signal' in signal_data:
                signals[agent_name] = signal_data['signal']
                confidences[agent_name] = signal_data.get('confidence', 0.0) / 100.0  # Normalize to 0-1
        
        # Warren Buffett features
        if 'warren_buffett_agent' in signals:
            signal = signals['warren_buffett_agent']
            features.warren_buffett_bullish = 1.0 if signal == 'bullish' else 0.0
            features.warren_buffett_bearish = 1.0 if signal == 'bearish' else 0.0
            features.warren_buffett_confidence = confidences.get('warren_buffett_agent', 0.0)
        
        # Technical analyst features
        if 'technical_analyst_agent' in signals:
            signal = signals['technical_analyst_agent']
            features.technical_analyst_bullish = 1.0 if signal == 'bullish' else 0.0
            features.technical_analyst_bearish = 1.0 if signal == 'bearish' else 0.0
            features.technical_analyst_confidence = confidences.get('technical_analyst_agent', 0.0)
        
        # Sentiment features
        if 'sentiment_agent' in signals:
            signal = signals['sentiment_agent']
            features.sentiment_bullish = 1.0 if signal == 'bullish' else 0.0
            features.sentiment_bearish = 1.0 if signal == 'bearish' else 0.0
            features.sentiment_confidence = confidences.get('sentiment_agent', 0.0)
        
        # Interaction features
        signal_values = [1 if s == 'bullish' else -1 if s == 'bearish' else 0 for s in signals.values()]
        confidence_values = list(confidences.values())
        
        if signal_values:
            # Agent agreement (standard deviation of signals)
            features.agent_agreement = 1.0 - (np.std(signal_values) / 2.0 if len(signal_values) > 1 else 0.0)
            
            # Confidence variance
            features.confidence_variance = np.var(confidence_values) if len(confidence_values) > 1 else 0.0
            
            # Bullish consensus
            bullish_count = sum(1 for s in signals.values() if s == 'bullish')
            features.bullish_consensus = bullish_count / len(signals) if signals else 0.0
        
        # Market regime features
        features.regime_bull = 1.0 if regime.market_state == 'bull' else 0.0
        features.regime_bear = 1.0 if regime.market_state == 'bear' else 0.0
        features.regime_volatility_high = 1.0 if regime.volatility == 'high' else 0.0
        features.regime_structure_trending = 1.0 if regime.structure == 'trending' else 0.0
        
        # Temporal features
        now = datetime.now()
        features.day_of_week = now.weekday() / 6.0  # Normalize to 0-1
        features.month = (now.month - 1) / 11.0  # Normalize to 0-1
        
        # Performance-based agent weights (from performance tracker)
        agent_perf_weights = self.agent_weights.calculate_agent_weights(ticker, now)
        features.warren_buffett_weight = agent_perf_weights.get('warren_buffett_agent', 0.33)
        features.technical_analyst_weight = agent_perf_weights.get('technical_analyst_agent', 0.33)
        features.sentiment_weight = agent_perf_weights.get('sentiment_agent', 0.33)
        
        return features

    def _ml_predict(
        self, 
        features: SignalFeatures, 
        regime: MarketRegime, 
        weighted_signals: Dict[str, Dict]
    ) -> PortfolioDecision:
        """Make prediction using trained ML models"""
        try:
            # Select model based on regime
            model = self.models[regime.market_state]
            
            # Convert features to array
            feature_array = np.array([[getattr(features, col) for col in self.feature_columns]])
            
            # Scale features
            feature_scaled = self.scaler.transform(feature_array)
            
            # Get prediction probabilities
            proba = model.predict_proba(feature_scaled)[0]
            
            # Convert probabilities to trading decision
            # Classes are typically [bearish, bullish, neutral] or similar
            action, quantity, confidence = self._interpret_ml_prediction(proba, weighted_signals)
            
            # Generate reasoning
            reasoning = self._generate_ml_reasoning(proba, features, regime, weighted_signals)
            
            return PortfolioDecision(
                action=action,
                quantity=quantity,
                confidence=confidence,
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            # Fallback to heuristic
            return self._heuristic_predict(weighted_signals, regime)

    def _heuristic_predict(
        self, 
        weighted_signals: Dict[str, Dict], 
        regime: MarketRegime
    ) -> PortfolioDecision:
        """
        Heuristic fallback prediction when ML models are unavailable.
        
        Uses weighted voting with regime-aware adjustments.
        """
        try:
            # Calculate weighted signal scores
            total_weight = 0
            weighted_score = 0
            confidence_sum = 0
            
            signal_details = []
            
            for agent_name, signal_data in weighted_signals.items():
                if isinstance(signal_data, dict) and 'signal' in signal_data:
                    signal = signal_data['signal']
                    confidence = signal_data.get('confidence', 0.0) / 100.0
                    weight = signal_data.get('weight', 1.0)
                    
                    # Convert signal to numeric score
                    score = 1.0 if signal == 'bullish' else -1.0 if signal == 'bearish' else 0.0
                    
                    # Apply weight and confidence
                    contribution = score * weight * confidence
                    weighted_score += contribution
                    total_weight += weight
                    confidence_sum += confidence * weight
                    
                    signal_details.append(f"{agent_name}: {signal} (conf: {confidence:.2f}, weight: {weight:.2f})")
            
            # Normalize scores
            if total_weight > 0:
                final_score = weighted_score / total_weight
                avg_confidence = min(confidence_sum / total_weight, 1.0)
            else:
                final_score = 0.0
                avg_confidence = 0.0
            
            # Apply regime adjustments
            regime_multiplier = self._get_regime_multiplier(regime)
            adjusted_score = final_score * regime_multiplier
            
            # Convert to trading decision
            if adjusted_score > 0.3:
                action = "buy"
                quantity = max(1, int(100 * adjusted_score))  # Scale quantity based on conviction
                confidence = min(avg_confidence * 100, 95.0)
            elif adjusted_score < -0.3:
                action = "sell"
                quantity = max(1, int(100 * abs(adjusted_score)))
                confidence = min(avg_confidence * 100, 95.0)
            else:
                action = "hold"
                quantity = 0
                confidence = min(avg_confidence * 100, 50.0)
            
            # Generate reasoning
            reasoning = self._generate_heuristic_reasoning(
                signal_details, regime, final_score, adjusted_score, regime_multiplier
            )
            
            return PortfolioDecision(
                action=action,
                quantity=quantity,
                confidence=confidence,
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Heuristic prediction failed: {e}")
            return PortfolioDecision(
                action="hold",
                quantity=0,
                confidence=0.0,
                reasoning=f"Heuristic fallback error: {str(e)}"
            )

    def _get_regime_multiplier(self, regime: MarketRegime) -> float:
        """Get regime-based adjustment multiplier for signals"""
        # Base multiplier
        multiplier = 1.0
        
        # Market state adjustments
        if regime.market_state == 'bull':
            multiplier *= 1.1  # Slightly favor bullish signals
        elif regime.market_state == 'bear':
            multiplier *= 0.9  # Slightly dampen bullish signals
        
        # Volatility adjustments
        if regime.volatility == 'high':
            multiplier *= 0.8  # Be more conservative in volatile markets
        
        # Structure adjustments
        if regime.structure == 'trending':
            multiplier *= 1.05  # Slightly favor directional signals
        elif regime.structure == 'mean_reverting':
            multiplier *= 0.95  # Slightly dampen directional signals
        
        return multiplier

    def _interpret_ml_prediction(
        self, 
        proba: np.ndarray, 
        weighted_signals: Dict[str, Dict]
    ) -> Tuple[str, int, float]:
        """Convert ML model probabilities to trading decision"""
        # Assuming model classes are [bearish, neutral, bullish]
        bearish_prob = proba[0] if len(proba) > 0 else 0.0
        neutral_prob = proba[1] if len(proba) > 1 else 0.0
        bullish_prob = proba[2] if len(proba) > 2 else 0.0
        
        max_prob = max(bearish_prob, neutral_prob, bullish_prob)
        
        # Decision thresholds
        if max_prob < 0.4:
            # Low confidence - hold
            return "hold", 0, max_prob * 100
        elif bullish_prob == max_prob and bullish_prob > 0.4:
            # Buy signal
            quantity = max(1, int(200 * bullish_prob))  # Scale with confidence
            return "buy", quantity, bullish_prob * 100
        elif bearish_prob == max_prob and bearish_prob > 0.4:
            # Sell signal
            quantity = max(1, int(200 * bearish_prob))
            return "sell", quantity, bearish_prob * 100
        else:
            # Hold (neutral or low confidence)
            return "hold", 0, max_prob * 100

    def _generate_ml_reasoning(
        self, 
        proba: np.ndarray, 
        features: SignalFeatures, 
        regime: MarketRegime,
        weighted_signals: Dict[str, Dict]
    ) -> str:
        """Generate reasoning for ML-based decision"""
        bearish_prob = proba[0] if len(proba) > 0 else 0.0
        neutral_prob = proba[1] if len(proba) > 1 else 0.0
        bullish_prob = proba[2] if len(proba) > 2 else 0.0
        
        reasoning = f"ML ensemble prediction based on {len(weighted_signals)} agent signals. "
        reasoning += f"Probabilities - Bullish: {bullish_prob:.2f}, Bearish: {bearish_prob:.2f}, Neutral: {neutral_prob:.2f}. "
        
        # Add regime context
        reasoning += f"Market regime: {regime.market_state}/{regime.volatility}/{regime.structure} (confidence: {regime.confidence:.2f}). "
        
        # Add key feature insights
        if features.agent_agreement > 0.8:
            reasoning += "Strong agent agreement detected. "
        elif features.agent_agreement < 0.5:
            reasoning += "Agents disagree significantly. "
        
        if features.bullish_consensus > 0.7:
            reasoning += "Bullish consensus among agents. "
        elif features.bullish_consensus < 0.3:
            reasoning += "Bearish consensus among agents. "
        
        return reasoning

    def _generate_heuristic_reasoning(
        self, 
        signal_details: List[str], 
        regime: MarketRegime, 
        raw_score: float, 
        adjusted_score: float,
        regime_multiplier: float
    ) -> str:
        """Generate reasoning for heuristic-based decision"""
        reasoning = f"Heuristic ensemble combining {len(signal_details)} agents. "
        reasoning += f"Raw weighted score: {raw_score:.3f}, "
        reasoning += f"regime-adjusted score: {adjusted_score:.3f} (multiplier: {regime_multiplier:.3f}). "
        
        # Add regime context
        reasoning += f"Market regime: {regime.market_state}/{regime.volatility}/{regime.structure}. "
        
        # Add agent details
        reasoning += "Signals: " + "; ".join(signal_details[:3])  # Limit to avoid too long reasoning
        if len(signal_details) > 3:
            reasoning += f" and {len(signal_details) - 3} more."
        
        return reasoning

    def load_models(self) -> bool:
        """Load trained models from disk"""
        if not HAS_SKLEARN:
            logger.warning("Cannot load ML models - scikit-learn not available")
            return False
        
        try:
            model_files = {
                'bull': self.models_dir / 'bull_model.joblib',
                'bear': self.models_dir / 'bear_model.joblib', 
                'neutral': self.models_dir / 'neutral_model.joblib',
                'scaler': self.models_dir / 'scaler.joblib',
                'metadata': self.models_dir / 'model_metadata.json'
            }
            
            # Check if all files exist
            if not all(f.exists() for f in model_files.values()):
                logger.warning("Some model files missing - ensemble will use heuristic mode")
                return False
            
            # Load models
            self.models['bull'] = joblib.load(model_files['bull'])
            self.models['bear'] = joblib.load(model_files['bear'])
            self.models['neutral'] = joblib.load(model_files['neutral'])
            self.scaler = joblib.load(model_files['scaler'])
            
            # Load metadata
            with open(model_files['metadata'], 'r') as f:
                metadata = json.load(f)
                self.model_version = metadata.get('version', '1.0')
                self.last_training_date = datetime.fromisoformat(metadata.get('training_date', datetime.now().isoformat()))
                self.training_samples = metadata.get('training_samples', 0)
            
            self.is_trained = True
            logger.info(f"Loaded ML ensemble models (version {self.model_version}, {self.training_samples} samples)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load ML models: {e}")
            self.is_trained = False
            return False

    def save_models(self) -> bool:
        """Save trained models to disk"""
        if not HAS_SKLEARN or not self.is_trained:
            logger.warning("Cannot save models - not trained or scikit-learn unavailable")
            return False
        
        try:
            # Save models
            joblib.dump(self.models['bull'], self.models_dir / 'bull_model.joblib')
            joblib.dump(self.models['bear'], self.models_dir / 'bear_model.joblib')
            joblib.dump(self.models['neutral'], self.models_dir / 'neutral_model.joblib')
            joblib.dump(self.scaler, self.models_dir / 'scaler.joblib')
            
            # Save metadata
            metadata = {
                'version': self.model_version,
                'training_date': datetime.now().isoformat(),
                'training_samples': self.training_samples,
                'feature_columns': self.feature_columns
            }
            
            with open(self.models_dir / 'model_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved ML ensemble models (version {self.model_version})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about current models"""
        return {
            'has_sklearn': HAS_SKLEARN,
            'is_trained': self.is_trained,
            'model_version': self.model_version,
            'last_training_date': self.last_training_date.isoformat() if self.last_training_date else None,
            'training_samples': self.training_samples,
            'feature_columns': self.feature_columns,
            'models_available': list(self.models.keys()) if self.models else []
        }


# Factory function for creating ensemble
def create_signal_ensemble(config: Dict[str, Any] = None) -> SignalEnsemble:
    """Factory function to create and initialize SignalEnsemble"""
    config = config or {}
    
    # Import regime detector
    try:
        from src.agents.regime_detector import RegimeDetector
        regime_detector = RegimeDetector()
    except ImportError:
        regime_detector = None
        logger.warning("Could not import regime detector for signal ensemble")
    
    # Create ensemble
    ensemble = SignalEnsemble(
        regime_detector=regime_detector,
        models_dir=config.get('models_dir', 'data/models')
    )
    
    # Try to load existing models
    ensemble.load_models()
    
    return ensemble


if __name__ == "__main__":
    # Test the signal ensemble system
    print("Testing Signal Ensemble System...")
    
    # Create test signals
    test_signals = {
        'warren_buffett_agent': {'signal': 'bullish', 'confidence': 75.0, 'reasoning': 'Strong fundamentals'},
        'technical_analyst_agent': {'signal': 'bullish', 'confidence': 60.0, 'reasoning': 'Upward trend'},
        'sentiment_agent': {'signal': 'neutral', 'confidence': 50.0, 'reasoning': 'Mixed sentiment'}
    }
    
    # Create ensemble
    ensemble = create_signal_ensemble()
    
    # Generate test decision
    decision = ensemble.generate_decision(test_signals, "AAPL", datetime.now())
    print(f"Test decision: {decision}")
    
    # Print model info
    info = ensemble.get_model_info()
    print(f"Model info: {info}")