from src.graph.state import AgentState, show_agent_reasoning
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
from typing_extensions import Literal
from src.tools.api import get_financial_metrics, get_market_cap, search_line_items
from src.utils.llm import call_llm
from src.utils.progress import progress
from src.utils.api_key import get_api_key_from_state
from src.agents.regime_detector import get_current_market_regime, AdaptiveThresholds
import logging

logger = logging.getLogger(__name__)


class WarrenBuffettSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


def warren_buffett_agent(state: AgentState, agent_id: str = "warren_buffett_agent"):
    """Analyzes stocks using Buffett's principles with adaptive, regime-aware thresholds."""
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")
    
    # Get current market regime for adaptive thresholds
    try:
        current_regime = get_current_market_regime(end_date, api_key)
        logger.info(f"Market regime detected: {current_regime.market_state}/{current_regime.volatility}/{current_regime.structure} (confidence: {current_regime.confidence:.2f})")
    except Exception as e:
        logger.warning(f"Failed to detect market regime: {e}. Using fallback neutral regime.")
        from src.agents.regime_detector import MarketRegime
        from datetime import datetime
        current_regime = MarketRegime(
            market_state="neutral",
            volatility="low", 
            structure="mean_reverting",
            confidence=0.3,
            timestamp=datetime.now()
        )
    
    # Collect all analysis for LLM reasoning
    analysis_data = {}
    buffett_analysis = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Fetching financial metrics")
        # Fetch required data - request more periods for better trend analysis
        metrics = get_financial_metrics(ticker, end_date, period="ttm", limit=10, api_key=api_key)

        progress.update_status(agent_id, ticker, "Gathering financial line items")
        financial_line_items = search_line_items(
            ticker,
            [
                "capital_expenditure",
                "depreciation_and_amortization",
                "net_income",
                "outstanding_shares",
                "total_assets",
                "total_liabilities",
                "shareholders_equity",
                "dividends_and_other_cash_distributions",
                "issuance_or_purchase_of_equity_shares",
                "gross_profit",
                "revenue",
                "free_cash_flow",
            ],
            end_date,
            period="ttm",
            limit=10,
            api_key=api_key,
        )

        progress.update_status(agent_id, ticker, "Getting market cap")
        # Get current market cap
        market_cap = get_market_cap(ticker, end_date, api_key=api_key)

        progress.update_status(agent_id, ticker, "Analyzing fundamentals with adaptive thresholds")
        # Analyze fundamentals with adaptive thresholds
        fundamental_analysis = analyze_fundamentals_adaptive(metrics, current_regime, ticker)

        progress.update_status(agent_id, ticker, "Analyzing consistency")
        consistency_analysis = analyze_consistency(financial_line_items)

        progress.update_status(agent_id, ticker, "Analyzing competitive moat")
        moat_analysis = analyze_moat(metrics)

        progress.update_status(agent_id, ticker, "Analyzing pricing power")
        pricing_power_analysis = analyze_pricing_power_adaptive(financial_line_items, metrics, current_regime)

        progress.update_status(agent_id, ticker, "Analyzing book value growth")
        book_value_analysis = analyze_book_value_growth(financial_line_items)

        progress.update_status(agent_id, ticker, "Analyzing management quality")
        mgmt_analysis = analyze_management_quality(financial_line_items)

        progress.update_status(agent_id, ticker, "Calculating intrinsic value with adaptive parameters")
        intrinsic_value_analysis = calculate_intrinsic_value_adaptive(financial_line_items, current_regime)

        # Calculate total score without circle of competence (LLM will handle that)
        total_score = (
            fundamental_analysis["score"] + 
            consistency_analysis["score"] + 
            moat_analysis["score"] + 
            mgmt_analysis["score"] +
            pricing_power_analysis["score"] + 
            book_value_analysis["score"]
        )
        
        # Update max possible score calculation
        max_possible_score = (
            10 +  # fundamental_analysis (ROE, debt, margins, current ratio)
            moat_analysis["max_score"] + 
            mgmt_analysis["max_score"] +
            5 +   # pricing_power (0-5)
            5     # book_value_growth (0-5)
        )

        # Add margin of safety analysis with adaptive thresholds
        margin_of_safety = None
        intrinsic_value = intrinsic_value_analysis["intrinsic_value"]
        if intrinsic_value and market_cap:
            margin_of_safety = (intrinsic_value - market_cap) / market_cap

        # Get adaptive thresholds for this analysis
        adaptive_thresholds = AdaptiveThresholds.get_adaptive_threshold("margin_of_safety", current_regime)
        
        # Combine all analysis results for LLM evaluation
        analysis_data[ticker] = {
            "ticker": ticker,
            "score": total_score,
            "max_score": max_possible_score,
            "fundamental_analysis": fundamental_analysis,
            "consistency_analysis": consistency_analysis,
            "moat_analysis": moat_analysis,
            "pricing_power_analysis": pricing_power_analysis,
            "book_value_analysis": book_value_analysis,
            "management_analysis": mgmt_analysis,
            "intrinsic_value_analysis": intrinsic_value_analysis,
            "market_cap": market_cap,
            "margin_of_safety": margin_of_safety,
            "market_regime": {
                "state": current_regime.market_state,
                "volatility": current_regime.volatility,
                "structure": current_regime.structure,
                "confidence": current_regime.confidence
            },
            "adaptive_thresholds": {
                "margin_of_safety_required": adaptive_thresholds,
                "discount_rate_used": intrinsic_value_analysis.get("assumptions", {}).get("discount_rate"),
                "regime_adjustments": f"Using {current_regime.market_state} market/{current_regime.volatility} volatility parameters"
            }
        }

        progress.update_status(agent_id, ticker, "Generating adaptive Warren Buffett analysis")
        buffett_output = generate_buffett_output_adaptive(
            ticker=ticker,
            analysis_data=analysis_data,
            current_regime=current_regime,
            state=state,
            agent_id=agent_id,
        )

        # Store analysis in consistent format with other agents
        buffett_analysis[ticker] = {
            "signal": buffett_output.signal,
            "confidence": buffett_output.confidence,
            "reasoning": buffett_output.reasoning,
        }

        progress.update_status(agent_id, ticker, "Done", analysis=buffett_output.reasoning)

    # Create the message
    message = HumanMessage(content=json.dumps(buffett_analysis), name=agent_id)

    # Show reasoning if requested
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(buffett_analysis, agent_id)

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"][agent_id] = buffett_analysis

    progress.update_status(agent_id, None, "Done")

    return {"messages": [message], "data": state["data"]}


def analyze_fundamentals_adaptive(metrics: list, regime, ticker: str, sector: str = None) -> dict[str, any]:
    """Analyze company fundamentals with adaptive, regime-aware thresholds."""
    if not metrics:
        return {"score": 0, "details": "Insufficient fundamental data", "regime_info": "No regime adjustments applied"}

    latest_metrics = metrics[0]
    score = 0
    reasoning = []

    # Get adaptive thresholds based on market regime
    roe_threshold = AdaptiveThresholds.get_adaptive_threshold("roe_threshold", regime, sector)
    debt_threshold = AdaptiveThresholds.get_adaptive_threshold("debt_to_equity_threshold", regime, sector)
    margin_threshold = AdaptiveThresholds.get_adaptive_threshold("operating_margin_threshold", regime, sector)
    current_ratio_threshold = AdaptiveThresholds.get_adaptive_threshold("current_ratio_threshold", regime, sector)
    
    # Log the adaptive adjustments
    reasoning.append(f"REGIME-ADAPTIVE ANALYSIS ({regime.market_state}/{regime.volatility})")
    
    # Check ROE with adaptive threshold
    if latest_metrics.return_on_equity and latest_metrics.return_on_equity > roe_threshold:
        score += 2
        reasoning.append(f"Strong ROE of {latest_metrics.return_on_equity:.1%} (threshold: {roe_threshold:.1%})")
    elif latest_metrics.return_on_equity:
        reasoning.append(f"Weak ROE of {latest_metrics.return_on_equity:.1%} (threshold: {roe_threshold:.1%})")
    else:
        reasoning.append("ROE data not available")

    # Check Debt to Equity with adaptive threshold
    if latest_metrics.debt_to_equity and latest_metrics.debt_to_equity < debt_threshold:
        score += 2
        reasoning.append(f"Conservative debt levels {latest_metrics.debt_to_equity:.1f} (threshold: <{debt_threshold:.1f})")
    elif latest_metrics.debt_to_equity:
        reasoning.append(f"High debt to equity ratio of {latest_metrics.debt_to_equity:.1f} (threshold: <{debt_threshold:.1f})")
    else:
        reasoning.append("Debt to equity data not available")

    # Check Operating Margin with adaptive threshold
    if latest_metrics.operating_margin and latest_metrics.operating_margin > margin_threshold:
        score += 2
        reasoning.append(f"Strong operating margins {latest_metrics.operating_margin:.1%} (threshold: >{margin_threshold:.1%})")
    elif latest_metrics.operating_margin:
        reasoning.append(f"Weak operating margin of {latest_metrics.operating_margin:.1%} (threshold: >{margin_threshold:.1%})")
    else:
        reasoning.append("Operating margin data not available")

    # Check Current Ratio with adaptive threshold
    if latest_metrics.current_ratio and latest_metrics.current_ratio > current_ratio_threshold:
        score += 1
        reasoning.append(f"Good liquidity position {latest_metrics.current_ratio:.1f} (threshold: >{current_ratio_threshold:.1f})")
    elif latest_metrics.current_ratio:
        reasoning.append(f"Weak liquidity with current ratio of {latest_metrics.current_ratio:.1f} (threshold: >{current_ratio_threshold:.1f})")
    else:
        reasoning.append("Current ratio data not available")

    # Add regime context
    regime_explanation = f"Thresholds adjusted for {regime.market_state} market conditions"
    if regime.volatility == "high":
        regime_explanation += " with heightened volatility requirements"
    
    return {
        "score": score, 
        "details": "; ".join(reasoning), 
        "metrics": latest_metrics.model_dump(),
        "regime_adjustments": regime_explanation,
        "adaptive_thresholds": {
            "roe": roe_threshold,
            "debt_to_equity": debt_threshold,
            "operating_margin": margin_threshold,
            "current_ratio": current_ratio_threshold
        }
    }


def analyze_pricing_power_adaptive(financial_line_items: list, metrics: list, regime) -> dict[str, any]:
    """
    Analyze pricing power with adaptive thresholds based on market regime.
    In volatile markets, require more consistent pricing power evidence.
    """
    if not financial_line_items or not metrics:
        return {"score": 0, "details": "Insufficient data for pricing power analysis"}
    
    score = 0
    reasoning = []
    
    # Adaptive thresholds based on regime
    margin_improvement_threshold = 0.02 if regime.volatility == "low" else 0.03  # Higher bar in volatile markets
    high_margin_threshold = 0.5 if regime.market_state == "bear" else 0.45  # Stricter in bear markets
    good_margin_threshold = 0.3 if regime.market_state == "bear" else 0.25
    
    # Check gross margin trends
    gross_margins = []
    for item in financial_line_items:
        if hasattr(item, 'gross_margin') and item.gross_margin is not None:
            gross_margins.append(item.gross_margin)
    
    if len(gross_margins) >= 3:
        # Check margin stability/improvement with adaptive thresholds
        recent_avg = sum(gross_margins[:2]) / 2 if len(gross_margins) >= 2 else gross_margins[0]
        older_avg = sum(gross_margins[-2:]) / 2 if len(gross_margins) >= 2 else gross_margins[-1]
        
        margin_change = recent_avg - older_avg
        
        if margin_change > margin_improvement_threshold:
            score += 3
            reasoning.append(f"Expanding gross margins (+{margin_change:.1%}) indicate strong pricing power (regime-adjusted threshold: >{margin_improvement_threshold:.1%})")
        elif margin_change > 0:
            score += 2
            reasoning.append(f"Improving gross margins (+{margin_change:.1%}) suggest good pricing power")
        elif abs(margin_change) < 0.01:  # Stable within 1%
            score += 1
            reasoning.append(f"Stable gross margins during {regime.market_state} market uncertainty")
        else:
            reasoning.append(f"Declining gross margins ({margin_change:.1%}) may indicate pricing pressure - concerning in {regime.market_state} environment")
    
    # Check if company has been able to maintain high margins consistently (stricter in bear markets)
    if gross_margins:
        avg_margin = sum(gross_margins) / len(gross_margins)
        if avg_margin > high_margin_threshold:
            score += 2
            reasoning.append(f"Consistently high gross margins ({avg_margin:.1%}) indicate strong pricing power (regime-adjusted threshold: >{high_margin_threshold:.1%})")
        elif avg_margin > good_margin_threshold:
            score += 1
            reasoning.append(f"Good gross margins ({avg_margin:.1%}) suggest decent pricing power (regime-adjusted threshold: >{good_margin_threshold:.1%})")
    
    regime_context = f"Analysis adjusted for {regime.market_state} market with {regime.volatility} volatility"
    
    return {
        "score": score,
        "details": "; ".join(reasoning) if reasoning else "Limited pricing power analysis available",
        "regime_adjustments": regime_context,
        "adaptive_thresholds": {
            "margin_improvement_required": margin_improvement_threshold,
            "high_margin_threshold": high_margin_threshold,
            "good_margin_threshold": good_margin_threshold
        }
    }


def calculate_intrinsic_value_adaptive(financial_line_items: list, regime) -> dict[str, any]:
    """
    Calculate intrinsic value with adaptive discount rates and growth assumptions based on market regime.
    """
    if not financial_line_items or len(financial_line_items) < 3:
        return {"intrinsic_value": None, "details": ["Insufficient data for reliable valuation"]}

    # Calculate owner earnings with better methodology
    earnings_data = calculate_owner_earnings(financial_line_items)
    if not earnings_data["owner_earnings"]:
        return {"intrinsic_value": None, "details": earnings_data["details"]}

    owner_earnings = earnings_data["owner_earnings"]
    latest_financial_line_items = financial_line_items[0]
    shares_outstanding = latest_financial_line_items.outstanding_shares

    if not shares_outstanding or shares_outstanding <= 0:
        return {"intrinsic_value": None, "details": ["Missing or invalid shares outstanding data"]}

    # Enhanced DCF with regime-adaptive assumptions
    details = []
    
    # Estimate growth rate based on historical performance (more conservative)
    historical_earnings = []
    for item in financial_line_items[:5]:  # Last 5 years
        if hasattr(item, 'net_income') and item.net_income:
            historical_earnings.append(item.net_income)
    
    # Calculate historical growth rate
    if len(historical_earnings) >= 3:
        oldest_earnings = historical_earnings[-1]
        latest_earnings = historical_earnings[0]
        years = len(historical_earnings) - 1
        
        if oldest_earnings > 0:
            historical_growth = ((latest_earnings / oldest_earnings) ** (1/years)) - 1
            # Conservative adjustment - cap growth and apply haircut
            historical_growth = max(-0.05, min(historical_growth, 0.15))  # Cap between -5% and 15%
            
            # Regime-based growth adjustment
            regime_growth_multiplier = {
                'bull': 1.1,    # Slightly higher growth expectations in bull markets
                'bear': 0.7,    # Much more conservative in bear markets
                'neutral': 0.85  # Moderately conservative in neutral markets
            }.get(regime.market_state, 0.85)
            
            conservative_growth = historical_growth * regime_growth_multiplier
        else:
            conservative_growth = 0.02 if regime.market_state == 'bear' else 0.03  # Lower default in bear markets
    else:
        conservative_growth = 0.02 if regime.market_state == 'bear' else 0.03
    
    # Adaptive assumptions based on market regime
    stage1_growth = min(conservative_growth, 0.08 if regime.market_state != 'bear' else 0.06)
    stage2_growth = min(conservative_growth * 0.5, 0.04 if regime.market_state != 'bear' else 0.03)
    terminal_growth = 0.025 if regime.market_state != 'bear' else 0.02
    
    # Regime-adaptive discount rate
    base_discount_rate = AdaptiveThresholds.get_adaptive_threshold("discount_rate", regime)
    
    # Additional volatility premium in high volatility environments
    volatility_premium = 0.01 if regime.volatility == "high" else 0
    discount_rate = base_discount_rate + volatility_premium
    
    # Three-stage DCF model
    stage1_years = 5 if regime.market_state != 'bear' else 4  # Shorter projection in bear markets
    stage2_years = 5
    
    present_value = 0
    details.append(f"Regime-adjusted DCF: {regime.market_state} market, {regime.volatility} volatility")
    details.append(f"Stage 1 ({stage1_growth:.1%}, {stage1_years}y), Stage 2 ({stage2_growth:.1%}, {stage2_years}y), Terminal ({terminal_growth:.1%})")
    
    # Stage 1: Higher growth
    stage1_pv = 0
    for year in range(1, stage1_years + 1):
        future_earnings = owner_earnings * (1 + stage1_growth) ** year
        pv = future_earnings / (1 + discount_rate) ** year
        stage1_pv += pv
    
    # Stage 2: Transition growth
    stage2_pv = 0
    stage1_final_earnings = owner_earnings * (1 + stage1_growth) ** stage1_years
    for year in range(1, stage2_years + 1):
        future_earnings = stage1_final_earnings * (1 + stage2_growth) ** year
        pv = future_earnings / (1 + discount_rate) ** (stage1_years + year)
        stage2_pv += pv
    
    # Terminal value using Gordon Growth Model
    final_earnings = stage1_final_earnings * (1 + stage2_growth) ** stage2_years
    terminal_earnings = final_earnings * (1 + terminal_growth)
    terminal_value = terminal_earnings / (discount_rate - terminal_growth)
    terminal_pv = terminal_value / (1 + discount_rate) ** (stage1_years + stage2_years)
    
    # Total intrinsic value
    intrinsic_value = stage1_pv + stage2_pv + terminal_pv
    
    # Apply regime-adaptive margin of safety
    margin_of_safety_multiplier = {
        'bull': 0.90,    # 10% haircut in bull markets
        'bear': 0.75,    # 25% haircut in bear markets  
        'neutral': 0.85  # 15% haircut in neutral markets
    }.get(regime.market_state, 0.85)
    
    if regime.volatility == "high":
        margin_of_safety_multiplier *= 0.95  # Additional 5% haircut for high volatility
    
    conservative_intrinsic_value = intrinsic_value * margin_of_safety_multiplier
    
    details.extend([
        f"Regime-adjusted discount rate: {discount_rate:.1%} (base: {base_discount_rate:.1%})",
        f"Growth assumptions adjusted for {regime.market_state} market",
        f"Margin of safety: {(1-margin_of_safety_multiplier)*100:.0f}% (regime-adjusted)",
        f"Stage 1 PV: ${stage1_pv:,.0f}",
        f"Stage 2 PV: ${stage2_pv:,.0f}",
        f"Terminal PV: ${terminal_pv:,.0f}",
        f"Total IV: ${intrinsic_value:,.0f}",
        f"Conservative IV: ${conservative_intrinsic_value:,.0f}",
        f"Owner earnings: ${owner_earnings:,.0f}",
    ])

    return {
        "intrinsic_value": conservative_intrinsic_value,
        "raw_intrinsic_value": intrinsic_value,
        "owner_earnings": owner_earnings,
        "assumptions": {
            "stage1_growth": stage1_growth,
            "stage2_growth": stage2_growth,
            "terminal_growth": terminal_growth,
            "discount_rate": discount_rate,
            "base_discount_rate": base_discount_rate,
            "volatility_premium": volatility_premium,
            "stage1_years": stage1_years,
            "stage2_years": stage2_years,
            "historical_growth": conservative_growth,
            "regime_multiplier": margin_of_safety_multiplier,
            "market_regime": regime.market_state,
            "volatility_regime": regime.volatility
        },
        "details": details,
    }


def generate_buffett_output_adaptive(
    ticker: str,
    analysis_data: dict[str, any],
    current_regime,
    state: AgentState,
    agent_id: str = "warren_buffett_agent",
) -> WarrenBuffettSignal:
    """Get investment decision from LLM with adaptive Buffett principles based on market regime"""
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are Warren Buffett, the Oracle of Omaha, but now with ADAPTIVE market intelligence. Analyze investment opportunities using my proven methodology enhanced with dynamic parameter adjustment based on current market conditions.

                MY ADAPTIVE CORE PRINCIPLES:
                
                🎯 REGIME-AWARE CIRCLE OF COMPETENCE: "Risk comes from not knowing what you're doing" - but risk parameters change with market conditions:

                BULL MARKET ADAPTATION:
                - Slightly more flexible on valuation metrics (P/E can be higher)
                - Still require strong business fundamentals but can accept growth premiums
                - Focus on companies that can sustain performance in eventual downturns
                - "Be greedy when others are greedy, but not recklessly so"

                BEAR MARKET ADAPTATION:
                - Stricter valuation requirements and higher margins of safety
                - Emphasize balance sheet strength and cash generation even more
                - Look for companies with pricing power during economic stress
                - "Be greedy when others are fearful" - but verify the fear is justified

                HIGH VOLATILITY PERIODS:
                - Require wider margins of safety due to uncertainty
                - Emphasize companies with predictable earnings streams
                - Value balance sheet conservatism more highly
                - Shorter valuation projection periods due to uncertainty

                ADAPTIVE VALUATION FRAMEWORK:
                The analysis provided uses regime-adjusted parameters:
                - Discount rates adapt to volatility conditions
                - Growth assumptions become more conservative in bear markets  
                - Margin of safety requirements increase in uncertain times
                - Fundamental thresholds adjust for sector and market conditions

                MY DECISION FRAMEWORK REMAINS THE SAME:
                1. Circle of Competence - Do I understand this business? (Consider regime impact on business model)
                2. Economic Moats - Will competitive advantages persist through market cycles?
                3. Management Quality - How do they perform under different market conditions?
                4. Financial Strength - Can they survive and thrive in current regime?
                5. Adaptive Valuation - Am I paying the right price for current conditions?

                CONFIDENCE LEVELS (Regime-Adjusted):
                - 90-100%: Exceptional business, regime-appropriate valuation, fits market conditions perfectly
                - 70-89%: Good business with decent moat, appropriately valued for current regime
                - 50-69%: Mixed signals, regime creates uncertainty about business prospects
                - 30-49%: Outside my expertise OR concerning fundamentals for current market environment
                - 10-29%: Poor business OR significantly overvalued for current conditions

                Remember: "Our favorite holding period is forever" - but "forever" must account for how businesses perform across different market regimes. The key is finding wonderful businesses at regime-appropriate prices.
                """,
            ),
            (
                "human",
                """Analyze this investment opportunity for {ticker} with current market conditions:

                CURRENT MARKET REGIME: {regime_info}

                COMPREHENSIVE ANALYSIS DATA WITH ADAPTIVE PARAMETERS:
                {analysis_data}

                🎯 KEY REGIME CONSIDERATIONS:
                - Market State: {market_state} (affects growth expectations and risk tolerance)
                - Volatility: {volatility} (affects margin of safety requirements)  
                - Structure: {structure} (affects business model sustainability)
                - Regime Confidence: {regime_confidence:.1%}

                Please provide your investment decision in exactly this JSON format:
                {{
                  "signal": "bullish" | "bearish" | "neutral",
                  "confidence": float between 0 and 100,
                  "reasoning": "string with your detailed Warren Buffett-style analysis incorporating regime adaptations"
                }}

                In your reasoning, be specific about:
                1. How the current {market_state}/{volatility} regime affects this investment (CRITICAL)
                2. Whether this falls within your circle of competence for current market conditions
                3. Your assessment of the business's competitive moat during this regime
                4. How management/capital allocation looks under current market stress/optimism
                5. Financial health relative to current economic environment
                6. Valuation using the regime-adjusted parameters provided
                7. Long-term prospects considering market cycle changes
                8. Why the adaptive analysis changes (or doesn't change) your view vs static analysis

                Speak as Warren Buffett would - with conviction, folksy wisdom, and specific references to how the current market environment affects your traditional criteria. Acknowledge when regime-adjusted parameters lead to different conclusions than static analysis would suggest.
                """,
            ),
        ]
    )

    regime_info = f"{current_regime.market_state.title()} Market / {current_regime.volatility.title()} Volatility / {current_regime.structure.replace('_', ' ').title()} Structure"
    
    prompt = template.invoke({
        "analysis_data": json.dumps(analysis_data, indent=2), 
        "ticker": ticker,
        "regime_info": regime_info,
        "market_state": current_regime.market_state,
        "volatility": current_regime.volatility, 
        "structure": current_regime.structure,
        "regime_confidence": current_regime.confidence
    })

    # Default fallback signal in case parsing fails
    def create_default_warren_buffett_signal():
        return WarrenBuffettSignal(
            signal="neutral", 
            confidence=30.0, 
            reasoning=f"Error in regime-adaptive analysis for {regime_info} conditions. Defaulting to neutral with regime-appropriate caution."
        )

    return call_llm(
        prompt=prompt,
        pydantic_model=WarrenBuffettSignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default_warren_buffett_signal,
    )


# Preserve original functions that don't need modification
def analyze_consistency(financial_line_items: list) -> dict[str, any]:
    """Analyze earnings consistency and growth."""
    if len(financial_line_items) < 4:  # Need at least 4 periods for trend analysis
        return {"score": 0, "details": "Insufficient historical data"}

    score = 0
    reasoning = []

    # Check earnings growth trend
    earnings_values = [item.net_income for item in financial_line_items if item.net_income]
    if len(earnings_values) >= 4:
        # Simple check: is each period's earnings bigger than the next?
        earnings_growth = all(earnings_values[i] > earnings_values[i + 1] for i in range(len(earnings_values) - 1))

        if earnings_growth:
            score += 3
            reasoning.append("Consistent earnings growth over past periods")
        else:
            reasoning.append("Inconsistent earnings growth pattern")

        # Calculate total growth rate from oldest to latest
        if len(earnings_values) >= 2 and earnings_values[-1] != 0:
            growth_rate = (earnings_values[0] - earnings_values[-1]) / abs(earnings_values[-1])
            reasoning.append(f"Total earnings growth of {growth_rate:.1%} over past {len(earnings_values)} periods")
    else:
        reasoning.append("Insufficient earnings data for trend analysis")

    return {
        "score": score,
        "details": "; ".join(reasoning),
    }


def analyze_moat(metrics: list) -> dict[str, any]:
    """
    Evaluate whether the company likely has a durable competitive advantage (moat).
    Enhanced to include multiple moat indicators that Buffett actually looks for:
    1. Consistent high returns on capital
    2. Pricing power (stable/growing margins)
    3. Scale advantages (improving metrics with size)
    4. Brand strength (inferred from margins and consistency)
    5. Switching costs (inferred from customer retention)
    """
    if not metrics or len(metrics) < 5:  # Need more data for proper moat analysis
        return {"score": 0, "max_score": 5, "details": "Insufficient data for comprehensive moat analysis"}

    reasoning = []
    moat_score = 0
    max_score = 5

    # 1. Return on Capital Consistency (Buffett's favorite moat indicator)
    historical_roes = [m.return_on_equity for m in metrics if m.return_on_equity is not None]
    historical_roics = [m.return_on_invested_capital for m in metrics if hasattr(m, 'return_on_invested_capital') and m.return_on_invested_capital is not None]
    
    if len(historical_roes) >= 5:
        # Check for consistently high ROE (>15% for most periods)
        high_roe_periods = sum(1 for roe in historical_roes if roe > 0.15)
        roe_consistency = high_roe_periods / len(historical_roes)
        
        if roe_consistency >= 0.8:  # 80%+ of periods with ROE > 15%
            moat_score += 2
            avg_roe = sum(historical_roes) / len(historical_roes)
            reasoning.append(f"Excellent ROE consistency: {high_roe_periods}/{len(historical_roes)} periods >15% (avg: {avg_roe:.1%}) - indicates durable competitive advantage")
        elif roe_consistency >= 0.6:
            moat_score += 1
            reasoning.append(f"Good ROE performance: {high_roe_periods}/{len(historical_roes)} periods >15%")
        else:
            reasoning.append(f"Inconsistent ROE: only {high_roe_periods}/{len(historical_roes)} periods >15%")
    else:
        reasoning.append("Insufficient ROE history for moat analysis")

    # 2. Operating Margin Stability (Pricing Power Indicator)
    historical_margins = [m.operating_margin for m in metrics if m.operating_margin is not None]
    if len(historical_margins) >= 5:
        # Check for stable or improving margins (sign of pricing power)
        avg_margin = sum(historical_margins) / len(historical_margins)
        recent_margins = historical_margins[:3]  # Last 3 periods
        older_margins = historical_margins[-3:]  # First 3 periods
        
        recent_avg = sum(recent_margins) / len(recent_margins)
        older_avg = sum(older_margins) / len(older_margins)
        
        if avg_margin > 0.2 and recent_avg >= older_avg:  # 20%+ margins and stable/improving
            moat_score += 1
            reasoning.append(f"Strong and stable operating margins (avg: {avg_margin:.1%}) indicate pricing power moat")
        elif avg_margin > 0.15:  # At least decent margins
            reasoning.append(f"Decent operating margins (avg: {avg_margin:.1%}) suggest some competitive advantage")
        else:
            reasoning.append(f"Low operating margins (avg: {avg_margin:.1%}) suggest limited pricing power")
    
    # 3. Asset Efficiency and Scale Advantages
    if len(metrics) >= 5:
        # Check asset turnover trends (revenue efficiency)
        asset_turnovers = []
        for m in metrics:
            if hasattr(m, 'asset_turnover') and m.asset_turnover is not None:
                asset_turnovers.append(m.asset_turnover)
        
        if len(asset_turnovers) >= 3:
            if any(turnover > 1.0 for turnover in asset_turnovers):  # Efficient asset use
                moat_score += 1
                reasoning.append("Efficient asset utilization suggests operational moat")
    
    # 4. Competitive Position Strength (inferred from trend stability)
    if len(historical_roes) >= 5 and len(historical_margins) >= 5:
        # Calculate coefficient of variation (stability measure)
        roe_avg = sum(historical_roes) / len(historical_roes)
        roe_variance = sum((roe - roe_avg) ** 2 for roe in historical_roes) / len(historical_roes)
        roe_stability = 1 - (roe_variance ** 0.5) / roe_avg if roe_avg > 0 else 0
        
        margin_avg = sum(historical_margins) / len(historical_margins)
        margin_variance = sum((margin - margin_avg) ** 2 for margin in historical_margins) / len(historical_margins)
        margin_stability = 1 - (margin_variance ** 0.5) / margin_avg if margin_avg > 0 else 0
        
        overall_stability = (roe_stability + margin_stability) / 2
        
        if overall_stability > 0.7:  # High stability indicates strong competitive position
            moat_score += 1
            reasoning.append(f"High performance stability ({overall_stability:.1%}) suggests strong competitive moat")
    
    # Cap the score at max_score
    moat_score = min(moat_score, max_score)

    return {
        "score": moat_score,
        "max_score": max_score,
        "details": "; ".join(reasoning) if reasoning else "Limited moat analysis available",
    }


def analyze_management_quality(financial_line_items: list) -> dict[str, any]:
    """
    Checks for share dilution or consistent buybacks, and some dividend track record.
    A simplified approach:
      - if there's net share repurchase or stable share count, it suggests management
        might be shareholder-friendly.
      - if there's a big new issuance, it might be a negative sign (dilution).
    """
    if not financial_line_items:
        return {"score": 0, "max_score": 2, "details": "Insufficient data for management analysis"}

    reasoning = []
    mgmt_score = 0

    latest = financial_line_items[0]
    if hasattr(latest, "issuance_or_purchase_of_equity_shares") and latest.issuance_or_purchase_of_equity_shares and latest.issuance_or_purchase_of_equity_shares < 0:
        # Negative means the company spent money on buybacks
        mgmt_score += 1
        reasoning.append("Company has been repurchasing shares (shareholder-friendly)")

    if hasattr(latest, "issuance_or_purchase_of_equity_shares") and latest.issuance_or_purchase_of_equity_shares and latest.issuance_or_purchase_of_equity_shares > 0:
        # Positive issuance means new shares => possible dilution
        reasoning.append("Recent common stock issuance (potential dilution)")
    else:
        reasoning.append("No significant new stock issuance detected")

    # Check for any dividends
    if hasattr(latest, "dividends_and_other_cash_distributions") and latest.dividends_and_other_cash_distributions and latest.dividends_and_other_cash_distributions < 0:
        mgmt_score += 1
        reasoning.append("Company has a track record of paying dividends")
    else:
        reasoning.append("No or minimal dividends paid")

    return {
        "score": mgmt_score,
        "max_score": 2,
        "details": "; ".join(reasoning),
    }


def calculate_owner_earnings(financial_line_items: list) -> dict[str, any]:
    """
    Calculate owner earnings (Buffett's preferred measure of true earnings power).
    Enhanced methodology: Net Income + Depreciation/Amortization - Maintenance CapEx - Working Capital Changes
    Uses multi-period analysis for better maintenance capex estimation.
    """
    if not financial_line_items or len(financial_line_items) < 2:
        return {"owner_earnings": None, "details": ["Insufficient data for owner earnings calculation"]}

    latest = financial_line_items[0]
    details = []

    # Core components
    net_income = latest.net_income
    depreciation = latest.depreciation_and_amortization
    capex = latest.capital_expenditure

    if not all([net_income is not None, depreciation is not None, capex is not None]):
        missing = []
        if net_income is None: missing.append("net income")
        if depreciation is None: missing.append("depreciation")
        if capex is None: missing.append("capital expenditure")
        return {"owner_earnings": None, "details": [f"Missing components: {', '.join(missing)}"]}

    # Enhanced maintenance capex estimation using historical analysis
    maintenance_capex = estimate_maintenance_capex(financial_line_items)
    
    # Working capital change analysis (if data available)
    working_capital_change = 0
    if len(financial_line_items) >= 2:
        try:
            current_assets_current = getattr(latest, 'current_assets', None)
            current_liab_current = getattr(latest, 'current_liabilities', None)
            
            previous = financial_line_items[1]
            current_assets_previous = getattr(previous, 'current_assets', None)
            current_liab_previous = getattr(previous, 'current_liabilities', None)
            
            if all([current_assets_current, current_liab_current, current_assets_previous, current_liab_previous]):
                wc_current = current_assets_current - current_liab_current
                wc_previous = current_assets_previous - current_liab_previous
                working_capital_change = wc_current - wc_previous
                details.append(f"Working capital change: ${working_capital_change:,.0f}")
        except:
            pass  # Skip working capital adjustment if data unavailable

    # Calculate owner earnings
    owner_earnings = net_income + depreciation - maintenance_capex - working_capital_change

    # Sanity checks
    if owner_earnings < net_income * 0.3:  # Owner earnings shouldn't be less than 30% of net income typically
        details.append("Warning: Owner earnings significantly below net income - high capex intensity")
    
    if maintenance_capex > depreciation * 2:  # Maintenance capex shouldn't typically exceed 2x depreciation
        details.append("Warning: Estimated maintenance capex seems high relative to depreciation")

    details.extend([
        f"Net income: ${net_income:,.0f}",
        f"Depreciation: ${depreciation:,.0f}",
        f"Estimated maintenance capex: ${maintenance_capex:,.0f}",
        f"Owner earnings: ${owner_earnings:,.0f}"
    ])

    return {
        "owner_earnings": owner_earnings,
        "components": {
            "net_income": net_income,
            "depreciation": depreciation,
            "maintenance_capex": maintenance_capex,
            "working_capital_change": working_capital_change,
            "total_capex": abs(capex) if capex else 0
        },
        "details": details,
    }


def estimate_maintenance_capex(financial_line_items: list) -> float:
    """
    Estimate maintenance capital expenditure using multiple approaches.
    Buffett considers this crucial for understanding true owner earnings.
    """
    if not financial_line_items:
        return 0
    
    # Approach 1: Historical average as % of revenue
    capex_ratios = []
    depreciation_values = []
    
    for item in financial_line_items[:5]:  # Last 5 periods
        if hasattr(item, 'capital_expenditure') and hasattr(item, 'revenue'):
            if item.capital_expenditure and item.revenue and item.revenue > 0:
                capex_ratio = abs(item.capital_expenditure) / item.revenue
                capex_ratios.append(capex_ratio)
        
        if hasattr(item, 'depreciation_and_amortization') and item.depreciation_and_amortization:
            depreciation_values.append(item.depreciation_and_amortization)
    
    # Approach 2: Percentage of depreciation (typically 80-120% for maintenance)
    latest_depreciation = financial_line_items[0].depreciation_and_amortization if financial_line_items[0].depreciation_and_amortization else 0
    
    # Approach 3: Industry-specific heuristics
    latest_capex = abs(financial_line_items[0].capital_expenditure) if financial_line_items[0].capital_expenditure else 0
    
    # Conservative estimate: Use the higher of:
    # 1. 85% of total capex (assuming 15% is growth capex)
    # 2. 100% of depreciation (replacement of worn-out assets)
    # 3. Historical average if stable
    
    method_1 = latest_capex * 0.85  # 85% of total capex
    method_2 = latest_depreciation  # 100% of depreciation
    
    # If we have historical data, use average capex ratio
    if len(capex_ratios) >= 3:
        avg_capex_ratio = sum(capex_ratios) / len(capex_ratios)
        latest_revenue = financial_line_items[0].revenue if hasattr(financial_line_items[0], 'revenue') and financial_line_items[0].revenue else 0
        method_3 = avg_capex_ratio * latest_revenue if latest_revenue else 0
        
        # Use the median of the three approaches for conservatism
        estimates = sorted([method_1, method_2, method_3])
        return estimates[1]  # Median
    else:
        # Use the higher of method 1 and 2
        return max(method_1, method_2)


def analyze_book_value_growth(financial_line_items: list) -> dict[str, any]:
    """Analyze book value per share growth - a key Buffett metric."""
    if len(financial_line_items) < 3:
        return {"score": 0, "details": "Insufficient data for book value analysis"}
    
    # Extract book values per share
    book_values = [
        item.shareholders_equity / item.outstanding_shares
        for item in financial_line_items
        if hasattr(item, 'shareholders_equity') and hasattr(item, 'outstanding_shares')
        and item.shareholders_equity and item.outstanding_shares
    ]
    
    if len(book_values) < 3:
        return {"score": 0, "details": "Insufficient book value data for growth analysis"}
    
    score = 0
    reasoning = []
    
    # Analyze growth consistency
    growth_periods = sum(1 for i in range(len(book_values) - 1) if book_values[i] > book_values[i + 1])
    growth_rate = growth_periods / (len(book_values) - 1)
    
    # Score based on consistency
    if growth_rate >= 0.8:
        score += 3
        reasoning.append("Consistent book value per share growth (Buffett's favorite metric)")
    elif growth_rate >= 0.6:
        score += 2
        reasoning.append("Good book value per share growth pattern")
    elif growth_rate >= 0.4:
        score += 1
        reasoning.append("Moderate book value per share growth")
    else:
        reasoning.append("Inconsistent book value per share growth")
    
    # Calculate and score CAGR
    cagr_score, cagr_reason = _calculate_book_value_cagr(book_values)
    score += cagr_score
    reasoning.append(cagr_reason)
    
    return {"score": score, "details": "; ".join(reasoning)}


def _calculate_book_value_cagr(book_values: list) -> tuple[int, str]:
    """Helper function to safely calculate book value CAGR and return score + reasoning."""
    if len(book_values) < 2:
        return 0, "Insufficient data for CAGR calculation"
    
    oldest_bv, latest_bv = book_values[-1], book_values[0]
    years = len(book_values) - 1
    
    # Handle different scenarios
    if oldest_bv > 0 and latest_bv > 0:
        cagr = ((latest_bv / oldest_bv) ** (1/years)) - 1
        if cagr > 0.15:
            return 2, f"Excellent book value CAGR: {cagr:.1%}"
        elif cagr > 0.1:
            return 1, f"Good book value CAGR: {cagr:.1%}"
        else:
            return 0, f"Book value CAGR: {cagr:.1%}"
    elif oldest_bv < 0 < latest_bv:
        return 3, "Excellent: Company improved from negative to positive book value"
    elif oldest_bv > 0 > latest_bv:
        return 0, "Warning: Company declined from positive to negative book value"
    else:
        return 0, "Unable to calculate meaningful book value CAGR due to negative values"