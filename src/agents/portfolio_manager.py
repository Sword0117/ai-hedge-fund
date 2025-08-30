import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from src.graph.state import AgentState, show_agent_reasoning
from pydantic import BaseModel, Field
from typing_extensions import Literal
from src.utils.progress import progress
from src.utils.llm import call_llm

logger = logging.getLogger(__name__)


class PortfolioDecision(BaseModel):
    action: Literal["buy", "sell", "short", "cover", "hold"]
    quantity: int = Field(description="Number of shares to trade")
    confidence: float = Field(description="Confidence in the decision, between 0.0 and 100.0")
    reasoning: str = Field(description="Reasoning for the decision")


class PortfolioManagerOutput(BaseModel):
    decisions: dict[str, PortfolioDecision] = Field(description="Dictionary of ticker to trading decisions")


class EnhancedPortfolioManager:
    """
    Enhanced Portfolio Manager with dual-mode operation (LLM vs ML Ensemble).
    
    Supports A/B testing between traditional LLM-based decision making and
    ML ensemble-based signal fusion for performance comparison.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize enhanced portfolio manager.
        
        Args:
            config: Configuration dict with ensemble settings
        """
        self.config = config or {}
        self.use_ml_ensemble = self.config.get('use_ml_ensemble', False)
        self.ensemble_config = self.config.get('ensemble_config', {})
        
        # Initialize ML ensemble if enabled
        self.ml_ensemble = None
        if self.use_ml_ensemble:
            try:
                from src.agents.signal_fusion import create_signal_ensemble
                self.ml_ensemble = create_signal_ensemble(self.ensemble_config)
                logger.info("ML Ensemble initialized for portfolio management")
            except ImportError as e:
                logger.warning(f"Could not initialize ML ensemble: {e}. Using LLM mode only.")
                self.use_ml_ensemble = False
        
        # Performance tracking
        self.performance_tracker = None
        if self.config.get('performance_tracking', {}).get('enabled', True):
            try:
                from src.agents.performance_tracker import AgentPerformanceTracker
                self.performance_tracker = AgentPerformanceTracker(
                    storage_dir=self.config.get('performance_tracking', {}).get('storage_path', 'data/agent_performance')
                )
                logger.info("Performance tracking enabled")
            except ImportError as e:
                logger.warning(f"Could not initialize performance tracker: {e}")
        
        # Decision logging for A/B testing analysis
        self.decision_log = []
        
        logger.info(f"EnhancedPortfolioManager initialized - Mode: {'ML Ensemble' if self.use_ml_ensemble else 'LLM'}")
    
    def make_decision(
        self,
        tickers: list[str],
        signals_by_ticker: dict[str, dict],
        current_prices: dict[str, float], 
        max_shares: dict[str, int],
        portfolio: dict[str, float],
        agent_id: str,
        state: AgentState,
        current_date: datetime = None
    ) -> PortfolioManagerOutput:
        """
        Make portfolio decisions using either ML ensemble or LLM based on configuration.
        
        Args:
            tickers: List of ticker symbols
            signals_by_ticker: Agent signals by ticker
            current_prices: Current prices for each ticker
            max_shares: Maximum shares allowed per ticker
            portfolio: Current portfolio state
            agent_id: Agent identifier
            state: Current agent state
            current_date: Current trading date
            
        Returns:
            PortfolioManagerOutput with trading decisions
        """
        current_date = current_date or datetime.now()
        
        try:
            if self.use_ml_ensemble and self.ml_ensemble:
                # Use ML ensemble for decision making
                decisions = self._make_ml_ensemble_decisions(
                    tickers, signals_by_ticker, current_prices, max_shares, 
                    portfolio, current_date
                )
                decision_source = 'ML'
            else:
                # Use traditional LLM-based decision making
                decisions = self._make_llm_decisions(
                    tickers, signals_by_ticker, current_prices, max_shares,
                    portfolio, agent_id, state
                )
                decision_source = 'LLM'
            
            # Log decisions for A/B testing analysis
            self._log_decisions(tickers, decisions, decision_source, current_date)
            
            # Create output format
            result = PortfolioManagerOutput(decisions=decisions)
            
            logger.debug(f"Generated {decision_source} decisions for {len(tickers)} tickers")
            return result
            
        except Exception as e:
            logger.error(f"Error in portfolio decision making: {e}")
            # Fallback to default decisions
            default_decisions = {
                ticker: PortfolioDecision(
                    action="hold", 
                    quantity=0, 
                    confidence=0.0, 
                    reasoning=f"Error in decision making: {str(e)}"
                )
                for ticker in tickers
            }
            return PortfolioManagerOutput(decisions=default_decisions)
    
    def _make_ml_ensemble_decisions(
        self,
        tickers: list[str],
        signals_by_ticker: dict[str, dict],
        current_prices: dict[str, float],
        max_shares: dict[str, int],
        portfolio: dict[str, float],
        current_date: datetime
    ) -> dict[str, PortfolioDecision]:
        """Generate decisions using ML ensemble"""
        decisions = {}
        
        for ticker in tickers:
            try:
                agent_signals = signals_by_ticker.get(ticker, {})
                
                if not agent_signals:
                    decisions[ticker] = PortfolioDecision(
                        action="hold",
                        quantity=0,
                        confidence=0.0,
                        reasoning="No agent signals available"
                    )
                    continue
                
                # Get ML ensemble decision
                ml_decision = self.ml_ensemble.generate_decision(
                    agent_signals=agent_signals,
                    ticker=ticker,
                    current_date=current_date,
                    market_context={
                        'current_price': current_prices.get(ticker, 0),
                        'max_shares': max_shares.get(ticker, 0),
                        'portfolio_cash': portfolio.get('cash', 0)
                    }
                )
                
                # Apply position and risk limits
                adjusted_decision = self._apply_risk_limits(
                    ml_decision, ticker, current_prices, max_shares, portfolio
                )
                
                decisions[ticker] = adjusted_decision
                
            except Exception as e:
                logger.error(f"Error in ML ensemble decision for {ticker}: {e}")
                decisions[ticker] = PortfolioDecision(
                    action="hold",
                    quantity=0,
                    confidence=0.0,
                    reasoning=f"ML ensemble error: {str(e)}"
                )
        
        return decisions
    
    def _make_llm_decisions(
        self,
        tickers: list[str],
        signals_by_ticker: dict[str, dict],
        current_prices: dict[str, float],
        max_shares: dict[str, int],
        portfolio: dict[str, float],
        agent_id: str,
        state: AgentState
    ) -> dict[str, PortfolioDecision]:
        """Generate decisions using traditional LLM approach"""
        # Use the original LLM-based decision making
        result = generate_trading_decision(
            tickers=tickers,
            signals_by_ticker=signals_by_ticker,
            current_prices=current_prices,
            max_shares=max_shares,
            portfolio=portfolio,
            agent_id=agent_id,
            state=state
        )
        
        # Add LLM identifier to reasoning
        llm_decisions = {}
        for ticker, decision in result.decisions.items():
            enhanced_decision = PortfolioDecision(
                action=decision.action,
                quantity=decision.quantity,
                confidence=decision.confidence,
                reasoning=f"[LLM Decision] {decision.reasoning}"
            )
            llm_decisions[ticker] = enhanced_decision
        
        return llm_decisions
    
    def _apply_risk_limits(
        self,
        decision: PortfolioDecision,
        ticker: str,
        current_prices: dict[str, float],
        max_shares: dict[str, int],
        portfolio: dict[str, float]
    ) -> PortfolioDecision:
        """Apply risk management limits to ML ensemble decisions"""
        current_price = current_prices.get(ticker, 0)
        max_allowed = max_shares.get(ticker, 0)
        
        if current_price <= 0:
            return PortfolioDecision(
                action="hold",
                quantity=0,
                confidence=0.0,
                reasoning="Invalid price - holding position"
            )
        
        # Apply quantity limits
        if decision.action in ["buy", "short"]:
            # Limit buy/short quantity to max_shares
            limited_quantity = min(decision.quantity, max_allowed)
            
            if limited_quantity != decision.quantity:
                reasoning_addendum = f" (Limited from {decision.quantity} to {limited_quantity} shares by risk limits)"
            else:
                reasoning_addendum = ""
            
            return PortfolioDecision(
                action=decision.action,
                quantity=limited_quantity,
                confidence=decision.confidence,
                reasoning=decision.reasoning + reasoning_addendum
            )
        
        # For sell/cover actions, quantities should be validated against current positions
        # (This would require current position data which isn't directly available here)
        return decision
    
    def _log_decisions(
        self,
        tickers: list[str],
        decisions: dict[str, PortfolioDecision],
        source: str,
        timestamp: datetime
    ) -> None:
        """Log decisions for A/B testing analysis"""
        try:
            log_entry = {
                'timestamp': timestamp.isoformat(),
                'source': source,
                'tickers': tickers,
                'decisions': {
                    ticker: {
                        'action': decision.action,
                        'quantity': decision.quantity,
                        'confidence': decision.confidence
                    }
                    for ticker, decision in decisions.items()
                }
            }
            
            self.decision_log.append(log_entry)
            
            # Keep only recent decisions (last 1000 entries)
            if len(self.decision_log) > 1000:
                self.decision_log = self.decision_log[-1000:]
                
        except Exception as e:
            logger.error(f"Error logging decisions: {e}")
    
    def update_performance(
        self,
        ticker: str,
        date: datetime,
        agent_signals: Dict[str, Dict[str, Any]],
        actual_return: float,
        market_regime: str = "neutral"
    ) -> None:
        """Update agent performance metrics"""
        if self.performance_tracker:
            try:
                self.performance_tracker.update_performance(
                    ticker=ticker,
                    date=date,
                    agent_signals=agent_signals,
                    actual_return=actual_return,
                    market_regime=market_regime
                )
            except Exception as e:
                logger.error(f"Error updating performance for {ticker}: {e}")
    
    def get_decision_log(self) -> list:
        """Get recent decision log for analysis"""
        return self.decision_log.copy()
    
    def get_performance_report(self, ticker: str = None) -> Dict[str, Any]:
        """Get performance report from tracker"""
        if self.performance_tracker:
            return self.performance_tracker.get_performance_report(ticker)
        else:
            return {"error": "Performance tracking not enabled"}


# Global enhanced portfolio manager instance
_enhanced_portfolio_manager = None

def get_enhanced_portfolio_manager(config: Dict[str, Any] = None) -> EnhancedPortfolioManager:
    """Get or create enhanced portfolio manager singleton"""
    global _enhanced_portfolio_manager
    
    if _enhanced_portfolio_manager is None:
        _enhanced_portfolio_manager = EnhancedPortfolioManager(config)
    
    return _enhanced_portfolio_manager


##### Portfolio Management Agent #####
def portfolio_management_agent(state: AgentState, agent_id: str = "portfolio_manager"):
    """Makes final trading decisions and generates orders for multiple tickers"""

    # Get the portfolio and analyst signals
    portfolio = state["data"]["portfolio"]
    analyst_signals = state["data"]["analyst_signals"]
    tickers = state["data"]["tickers"]

    # Get position limits, current prices, and signals for every ticker
    position_limits = {}
    current_prices = {}
    max_shares = {}
    signals_by_ticker = {}
    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Processing analyst signals")

        # Get position limits and current prices for the ticker
        # Find the corresponding risk manager for this portfolio manager
        if agent_id.startswith("portfolio_manager_"):
            suffix = agent_id.split('_')[-1]
            risk_manager_id = f"risk_management_agent_{suffix}"
        else:
            risk_manager_id = "risk_management_agent"  # Fallback for CLI
        
        risk_data = analyst_signals.get(risk_manager_id, {}).get(ticker, {})
        position_limits[ticker] = risk_data.get("remaining_position_limit", 0)
        current_prices[ticker] = risk_data.get("current_price", 0)

        # Calculate maximum shares allowed based on position limit and price
        if current_prices[ticker] > 0:
            max_shares[ticker] = int(position_limits[ticker] / current_prices[ticker])
        else:
            max_shares[ticker] = 0

        # Get signals for the ticker
        ticker_signals = {}
        for agent, signals in analyst_signals.items():
            # Skip all risk management agents (they have different signal structure)
            if not agent.startswith("risk_management_agent") and ticker in signals:
                ticker_signals[agent] = {"signal": signals[ticker]["signal"], "confidence": signals[ticker]["confidence"]}
        signals_by_ticker[ticker] = ticker_signals

    # Add current_prices to the state data so it's available throughout the workflow
    state["data"]["current_prices"] = current_prices

    progress.update_status(agent_id, None, "Generating trading decisions")

    # Check if enhanced portfolio manager is configured
    config = state.get("config", {})
    portfolio_config = config.get("portfolio_manager", {})
    
    if portfolio_config.get("use_enhanced_manager", False):
        # Use enhanced portfolio manager with ML ensemble capability
        try:
            enhanced_manager = get_enhanced_portfolio_manager(portfolio_config)
            
            result = enhanced_manager.make_decision(
                tickers=tickers,
                signals_by_ticker=signals_by_ticker,
                current_prices=current_prices,
                max_shares=max_shares,
                portfolio=portfolio,
                agent_id=agent_id,
                state=state,
                current_date=datetime.fromisoformat(state["data"]["end_date"]) if "end_date" in state["data"] else datetime.now()
            )
            
            logger.info(f"Used enhanced portfolio manager ({'ML Ensemble' if enhanced_manager.use_ml_ensemble else 'LLM'} mode)")
            
        except Exception as e:
            logger.error(f"Enhanced portfolio manager failed: {e}. Falling back to traditional LLM.")
            # Fallback to original method
            result = generate_trading_decision(
                tickers=tickers,
                signals_by_ticker=signals_by_ticker,
                current_prices=current_prices,
                max_shares=max_shares,
                portfolio=portfolio,
                agent_id=agent_id,
                state=state,
            )
    else:
        # Use traditional LLM-based portfolio management
        result = generate_trading_decision(
            tickers=tickers,
            signals_by_ticker=signals_by_ticker,
            current_prices=current_prices,
            max_shares=max_shares,
            portfolio=portfolio,
            agent_id=agent_id,
            state=state,
        )

    # Create the portfolio management message
    message = HumanMessage(
        content=json.dumps({ticker: decision.model_dump() for ticker, decision in result.decisions.items()}),
        name=agent_id,
    )

    # Print the decision if the flag is set
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning({ticker: decision.model_dump() for ticker, decision in result.decisions.items()}, "Portfolio Manager")

    progress.update_status(agent_id, None, "Done")

    return {
        "messages": state["messages"] + [message],
        "data": state["data"],
    }


def generate_trading_decision(
    tickers: list[str],
    signals_by_ticker: dict[str, dict],
    current_prices: dict[str, float],
    max_shares: dict[str, int],
    portfolio: dict[str, float],
    agent_id: str,
    state: AgentState,
) -> PortfolioManagerOutput:
    """Attempts to get a decision from the LLM with retry logic"""
    # Create the prompt template
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a portfolio manager making final trading decisions based on multiple tickers.

              IMPORTANT: You are managing an existing portfolio with current positions. The portfolio_positions shows:
              - "long": number of shares currently held long
              - "short": number of shares currently held short
              - "long_cost_basis": average price paid for long shares
              - "short_cost_basis": average price received for short shares
              
              Trading Rules:
              - For long positions:
                * Only buy if you have available cash
                * Only sell if you currently hold long shares of that ticker
                * Sell quantity must be ≤ current long position shares
                * Buy quantity must be ≤ max_shares for that ticker
              
              - For short positions:
                * Only short if you have available margin (position value × margin requirement)
                * Only cover if you currently have short shares of that ticker
                * Cover quantity must be ≤ current short position shares
                * Short quantity must respect margin requirements
              
              - The max_shares values are pre-calculated to respect position limits
              - Consider both long and short opportunities based on signals
              - Maintain appropriate risk management with both long and short exposure

              Available Actions:
              - "buy": Open or add to long position
              - "sell": Close or reduce long position (only if you currently hold long shares)
              - "short": Open or add to short position
              - "cover": Close or reduce short position (only if you currently hold short shares)
              - "hold": Maintain current position without any changes (quantity should be 0 for hold)

              Inputs:
              - signals_by_ticker: dictionary of ticker → signals
              - max_shares: maximum shares allowed per ticker
              - portfolio_cash: current cash in portfolio
              - portfolio_positions: current positions (both long and short)
              - current_prices: current prices for each ticker
              - margin_requirement: current margin requirement for short positions (e.g., 0.5 means 50%)
              - total_margin_used: total margin currently in use
              """,
            ),
            (
                "human",
                """Based on the team's analysis, make your trading decisions for each ticker.

              Here are the signals by ticker:
              {signals_by_ticker}

              Current Prices:
              {current_prices}

              Maximum Shares Allowed For Purchases:
              {max_shares}

              Portfolio Cash: {portfolio_cash}
              Current Positions: {portfolio_positions}
              Current Margin Requirement: {margin_requirement}
              Total Margin Used: {total_margin_used}

              IMPORTANT DECISION RULES:
              - If you currently hold LONG shares of a ticker (long > 0), you can:
                * HOLD: Keep your current position (quantity = 0)
                * SELL: Reduce/close your long position (quantity = shares to sell)
                * BUY: Add to your long position (quantity = additional shares to buy)
                
              - If you currently hold SHORT shares of a ticker (short > 0), you can:
                * HOLD: Keep your current position (quantity = 0)
                * COVER: Reduce/close your short position (quantity = shares to cover)
                * SHORT: Add to your short position (quantity = additional shares to short)
                
              - If you currently hold NO shares of a ticker (long = 0, short = 0), you can:
                * HOLD: Stay out of the position (quantity = 0)
                * BUY: Open a new long position (quantity = shares to buy)
                * SHORT: Open a new short position (quantity = shares to short)

              Output strictly in JSON with the following structure:
              {{
                "decisions": {{
                  "TICKER1": {{
                    "action": "buy/sell/short/cover/hold",
                    "quantity": integer,
                    "confidence": float between 0 and 100,
                    "reasoning": "string explaining your decision considering current position"
                  }},
                  "TICKER2": {{
                    ...
                  }},
                  ...
                }}
              }}
              """,
            ),
        ]
    )

    # Generate the prompt
    prompt_data = {
        "signals_by_ticker": json.dumps(signals_by_ticker, indent=2),
        "current_prices": json.dumps(current_prices, indent=2),
        "max_shares": json.dumps(max_shares, indent=2),
        "portfolio_cash": f"{portfolio.get('cash', 0):.2f}",
        "portfolio_positions": json.dumps(portfolio.get("positions", {}), indent=2),
        "margin_requirement": f"{portfolio.get('margin_requirement', 0):.2f}",
        "total_margin_used": f"{portfolio.get('margin_used', 0):.2f}",
    }
    
    prompt = template.invoke(prompt_data)

    # Create default factory for PortfolioManagerOutput
    def create_default_portfolio_output():
        return PortfolioManagerOutput(decisions={ticker: PortfolioDecision(action="hold", quantity=0, confidence=0.0, reasoning="Default decision: hold") for ticker in tickers})

    return call_llm(
        prompt=prompt,
        pydantic_model=PortfolioManagerOutput,
        agent_name=agent_id,
        state=state,
        default_factory=create_default_portfolio_output,
    )
