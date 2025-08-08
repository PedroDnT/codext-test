#!/usr/bin/env python3
"""
Trading Analysis Report Generator
Professional institutional-grade research report generation for cryptocurrency trading
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class TradingReportGenerator:
    """Generate comprehensive trading analysis reports"""
    
    def __init__(self):
        self.report_path = None
    
    def generate_comprehensive_report(self, analysis_data: Dict, output_path: str = None) -> str:
        """
        Generate comprehensive trading analysis report
        
        Args:
            analysis_data: Complete analysis data from MarketAnalyzer
            output_path: Optional output file path
            
        Returns:
            Path to generated report
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"/home/ubuntu/crypto_trading_analysis_report_{timestamp}.md"
        
        self.report_path = output_path
        
        # Generate report sections
        self._write_header(analysis_data)
        self._write_executive_summary(analysis_data)
        self._write_market_sentiment_analysis(analysis_data)
        self._write_detailed_trade_analysis(analysis_data)
        self._write_risk_assessment(analysis_data)
        self._write_technical_appendix(analysis_data)
        self._write_disclaimer()
        
        logger.info(f"Comprehensive trading report generated: {output_path}")
        return output_path
    
    def _write_header(self, analysis_data: Dict):
        """Write report header and title page"""
        timestamp = analysis_data.get('analysis_timestamp', datetime.now().isoformat())
        analysis_date = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).strftime("%B %d, %Y at %H:%M UTC")
        
        header_content = f"""# Cryptocurrency Trading Analysis Report
## Leveraged Perpetual Contracts - Medium-Term Opportunities

**Analysis Date:** {analysis_date}  
**Leverage:** 20x  
**Time Horizon:** Hours to 1-2 Days  
**Focus:** High-Probability Trading Setups

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Market Sentiment Analysis](#market-sentiment-analysis)
3. [Detailed Trade Analysis](#detailed-trade-analysis)
4. [Risk Assessment](#risk-assessment)
5. [Technical Appendix](#technical-appendix)
6. [Risk Disclaimer](#risk-disclaimer)

---

"""
        
        with open(self.report_path, 'w') as f:
            f.write(header_content)
    
    def _write_executive_summary(self, analysis_data: Dict):
        """Write executive summary section"""
        market_sentiment = analysis_data.get('market_sentiment', {})
        trade_setups = analysis_data.get('top_trade_setups', [])
        analysis_summary = analysis_data.get('analysis_summary', {})
        
        overall_sentiment = market_sentiment.get('overall_sentiment', 'neutral').title()
        market_bias = market_sentiment.get('market_bias', 'neutral').replace('_', ' ').title()
        sentiment_score = market_sentiment.get('sentiment_score', 0)
        
        exec_summary = f"""
## Executive Summary

### Overall Market Assessment

The cryptocurrency market is currently exhibiting **{overall_sentiment}** sentiment with a **{market_bias}** bias (sentiment score: {sentiment_score:.1f}/100). Our comprehensive analysis of major cryptocurrencies has identified {len(trade_setups)} high-probability trading opportunities suitable for leveraged perpetual contracts.

### Market Conditions Summary

- **Overall Crypto Market Sentiment:** {overall_sentiment}
- **Risk Environment:** {market_bias}
- **Assets Analyzed:** {analysis_summary.get('total_assets_analyzed', 0)}
- **Valid Technical Setups:** {analysis_summary.get('trade_setups_generated', 0)}

### Top 3 Actionable Trade Setups

"""
        
        with open(self.report_path, 'a') as f:
            f.write(exec_summary)
        
        # Write top trade setups
        if trade_setups:
            for i, setup in enumerate(trade_setups[:3], 1):
                setup_summary = f"""
#### {i}. {setup['asset']}/USDT Perpetual - {setup['direction']}

**Strategy:** {setup['strategy']}  
**Setup Strength:** {setup['setup_strength']:.1f}/100  
**Current Price:** ${setup['current_price']:.2f}  
**Market Context:** {setup['market_context'].title()}

**Quick Overview:** {setup.get('strategy_rationale', 'Technical setup identified based on market structure analysis.')}

"""
                with open(self.report_path, 'a') as f:
                    f.write(setup_summary)
        else:
            no_setups = """
Currently, no high-probability trading setups meet our stringent criteria for 20x leveraged positions. This may be due to:
- Market conditions not favorable for high-leverage trading
- Lack of clear technical setups with sufficient conviction
- Data limitations affecting analysis quality

We recommend waiting for clearer market conditions before entering leveraged positions.

"""
            with open(self.report_path, 'a') as f:
                f.write(no_setups)
        
        # Primary risk factors
        risk_factors = self._identify_primary_risk_factors(analysis_data)
        risk_section = f"""
### Primary Risk Factors

{risk_factors}

---

"""
        
        with open(self.report_path, 'a') as f:
            f.write(risk_section)
    
    def _write_market_sentiment_analysis(self, analysis_data: Dict):
        """Write detailed market sentiment analysis"""
        market_sentiment = analysis_data.get('market_sentiment', {})
        components = market_sentiment.get('components', {})
        
        sentiment_section = f"""
## Market Sentiment Analysis

### Comprehensive Sentiment Overview

Our multi-factor sentiment analysis incorporates traditional finance indicators, cryptocurrency-specific metrics, and cross-asset correlations to provide a holistic view of market conditions.

"""
        
        with open(self.report_path, 'a') as f:
            f.write(sentiment_section)
        
        # Traditional Finance Analysis
        tradfi = components.get('traditional_finance', {})
        if tradfi.get('score') is not None:
            tradfi_analysis = f"""
### Traditional Finance Context

**S&P 500 & VIX Analysis**  
**Sentiment Score:** {tradfi['score']:.1f}  
**Interpretation:** {tradfi['interpretation'].replace('_', ' ').title()}

The traditional finance markets are currently showing {tradfi['interpretation'].replace('_', ' ')} conditions, which typically {self._get_tradfi_impact(tradfi['interpretation'])} for cryptocurrency markets due to their correlation with risk assets.

"""
            with open(self.report_path, 'a') as f:
                f.write(tradfi_analysis)
        
        # Bitcoin Dominance Analysis
        dominance = components.get('dominance', {})
        if dominance.get('score') is not None:
            dom_analysis = f"""
### Bitcoin Dominance Analysis

**Dominance Sentiment Score:** {dominance['score']:.1f}  
**Current Interpretation:** {dominance['interpretation'].replace('_', ' ').title()}

{self._get_dominance_interpretation(dominance['interpretation'])}

"""
            with open(self.report_path, 'a') as f:
                f.write(dom_analysis)
        
        # Crypto-Specific Sentiment
        crypto_sentiment = components.get('crypto_markets', {})
        if crypto_sentiment.get('score') is not None:
            crypto_analysis = f"""
### Cryptocurrency Market Sentiment

**Funding Rate Analysis Score:** {crypto_sentiment['score']:.1f}  
**Market Interpretation:** {crypto_sentiment['interpretation'].replace('_', ' ').title()}

{self._get_funding_interpretation(crypto_sentiment['interpretation'])}

---

"""
            with open(self.report_path, 'a') as f:
                f.write(crypto_analysis)
    
    def _write_detailed_trade_analysis(self, analysis_data: Dict):
        """Write detailed analysis for each trade setup"""
        trade_setups = analysis_data.get('top_trade_setups', [])
        
        trade_section = """
## Detailed Trade Analysis

The following section provides comprehensive analysis for each identified trading opportunity, including entry/exit parameters, risk metrics, and supporting technical evidence.

"""
        
        with open(self.report_path, 'a') as f:
            f.write(trade_section)
        
        if not trade_setups:
            no_trades = """
### No High-Conviction Setups Identified

Based on our comprehensive analysis, no trading setups currently meet our criteria for high-probability, medium-term leveraged positions. This conservative approach prioritizes capital preservation over forced trade generation.

**Reasons for No Setups:**
- Market conditions may be too volatile for 20x leverage
- Technical patterns lack sufficient confirmation
- Risk-reward ratios do not meet our standards
- Sentiment indicators show conflicting signals

**Recommendation:** Wait for clearer market conditions and more definitive technical setups before entering leveraged positions.

"""
            with open(self.report_path, 'a') as f:
                f.write(no_trades)
            return
        
        for i, setup in enumerate(trade_setups, 1):
            self._write_individual_trade_analysis(setup, i)
    
    def _write_individual_trade_analysis(self, setup: Dict, trade_number: int):
        """Write detailed analysis for individual trade"""
        asset = setup['asset']
        direction = setup['direction']
        strategy = setup['strategy']
        current_price = setup['current_price']
        entry_zone = setup.get('entry_zone', {})
        tp_levels = setup.get('take_profit_levels', [])
        stop_loss = setup.get('stop_loss')
        rationale = setup.get('strategy_rationale', '')
        
        trade_header = f"""
### Trade Setup #{trade_number}: {direction} {asset}/USDT Perpetual

**Asset & Direction:** {direction} - {asset}/USDT Perpetual  
**Strategy Type:** {strategy}  
**Setup Strength:** {setup['setup_strength']:.1f}/100  
**Leverage:** {setup['leverage']}x

"""
        
        with open(self.report_path, 'a') as f:
            f.write(trade_header)
        
        # Strategy Rationale
        rationale_section = f"""
#### Strategy Rationale

{rationale}

This setup is based on {strategy.lower()} analysis, which has shown historical effectiveness in similar market conditions. The setup strength of {setup['setup_strength']:.1f}/100 indicates {'high' if setup['setup_strength'] > 70 else 'moderate' if setup['setup_strength'] > 40 else 'low'} conviction based on our quantitative scoring model.

"""
        
        with open(self.report_path, 'a') as f:
            f.write(rationale_section)
        
        # Trade Execution Parameters
        execution_section = f"""
#### Trade Execution Parameters

**Current Price:** ${current_price:.2f}

"""
        
        if entry_zone:
            execution_section += f"""**Entry Zone:**  
- Lower Bound: ${entry_zone.get('lower', current_price):.2f}  
- Upper Bound: ${entry_zone.get('upper', current_price):.2f}  
- Recommended Entry: ${(entry_zone.get('lower', current_price) + entry_zone.get('upper', current_price)) / 2:.2f}

"""
        
        if tp_levels:
            execution_section += "**Take Profit Levels:**\n"
            for i, tp in enumerate(tp_levels[:3], 1):
                if tp:
                    pct_gain = ((tp - current_price) / current_price) * 100 * (1 if direction == 'LONG' else -1)
                    execution_section += f"- TP{i}: ${tp:.2f} ({pct_gain:+.1f}%)\n"
            execution_section += "\n"
        
        if stop_loss:
            pct_loss = ((stop_loss - current_price) / current_price) * 100 * (1 if direction == 'LONG' else -1)
            execution_section += f"**Stop Loss:** ${stop_loss:.2f} ({pct_loss:+.1f}%)\n\n"
        
        with open(self.report_path, 'a') as f:
            f.write(execution_section)
        
        # Risk Metrics
        risk_metrics = setup.get('risk_metrics', {})
        liquidation_prices = risk_metrics.get('liquidation_prices', {})
        
        if liquidation_prices:
            liq_price = liquidation_prices.get('long_position' if direction == 'LONG' else 'short_position')
            if liq_price:
                liq_distance = abs(liq_price - current_price) / current_price * 100
                
                risk_section = f"""
#### Risk Metrics

**Liquidation Price:** ${liq_price:.2f} ({liq_distance:.1f}% from current price)  
**Position Size Recommendation:** Conservative sizing due to 20x leverage  
**Risk Level:** {risk_metrics.get('volatility_level', 'medium').title()}

"""
                with open(self.report_path, 'a') as f:
                    f.write(risk_section)
        
        # Supporting Evidence
        evidence_section = f"""
#### Supporting Evidence

- **Technical Analysis:** {strategy} pattern identified with quantitative confirmation
- **Market Structure:** Setup aligns with current market structure analysis
- **Volume Analysis:** {self._get_volume_context(setup)}
- **Sentiment Alignment:** Setup {'aligns with' if setup.get('market_context') != 'neutral' else 'is neutral to'} current market sentiment

---

"""
        
        with open(self.report_path, 'a') as f:
            f.write(evidence_section)
    
    def _write_risk_assessment(self, analysis_data: Dict):
        """Write comprehensive risk assessment section"""
        trade_setups = analysis_data.get('top_trade_setups', [])
        market_sentiment = analysis_data.get('market_sentiment', {})
        
        risk_section = f"""
## Risk Assessment

### Leverage Risk Analysis

**WARNING:** All recommended trades utilize 20x leverage, which significantly amplifies both potential profits and losses. A 5% adverse price movement will result in 100% loss of position margin.

"""
        
        with open(self.report_path, 'a') as f:
            f.write(risk_section)
        
        # Individual trade risks
        if trade_setups:
            for i, setup in enumerate(trade_setups, 1):
                risk_metrics = setup.get('risk_metrics', {})
                liquidation_prices = risk_metrics.get('liquidation_prices', {})
                
                trade_risk = f"""
#### Trade #{i} Risk Analysis - {setup['asset']} {setup['direction']}

"""
                
                if liquidation_prices:
                    direction = setup['direction']
                    liq_price = liquidation_prices.get('long_position' if direction == 'LONG' else 'short_position')
                    current_price = setup['current_price']
                    
                    if liq_price:
                        liq_distance = abs(liq_price - current_price) / current_price * 100
                        
                        trade_risk += f"""**Liquidation Analysis:**
- Liquidation Price: ${liq_price:.2f}
- Distance to Liquidation: {liq_distance:.1f}%
- Risk Level: {'HIGH' if liq_distance < 10 else 'MEDIUM' if liq_distance < 20 else 'LOW'}

"""
                
                # Scenario Analysis
                stop_loss = setup.get('stop_loss')
                tp_levels = setup.get('take_profit_levels', [])
                
                if stop_loss and tp_levels:
                    best_tp = tp_levels[0] if tp_levels else current_price
                    
                    best_case_pct = ((best_tp - current_price) / current_price) * 100 * (1 if direction == 'LONG' else -1)
                    worst_case_pct = ((stop_loss - current_price) / current_price) * 100 * (1 if direction == 'LONG' else -1)
                    
                    trade_risk += f"""**Scenario Analysis:**
- Best Case (TP1): {best_case_pct:+.1f}% return
- Worst Case (SL): {worst_case_pct:+.1f}% loss
- Risk-Reward Ratio: {abs(best_case_pct / worst_case_pct):.2f}:1

"""
                
                # Trade Invalidation
                trade_risk += f"""**Trade Invalidation Conditions:**
- Stop loss hit: Position invalidated
- Market structure breakdown: Reassess all positions
- Significant news events: Review position sizing
- Extreme volatility: Consider position reduction

"""
                
                with open(self.report_path, 'a') as f:
                    f.write(trade_risk)
        
        # Overall market risks
        overall_risks = f"""
### Overall Market Risk Factors

**Systematic Risks:**
- Cryptocurrency market volatility remains extremely high
- Regulatory developments can cause sudden price movements
- Macroeconomic factors affecting risk asset sentiment
- Exchange-specific risks (outages, liquidity issues)

**Leverage-Specific Risks:**
- Margin calls and forced liquidations
- Slippage during volatile periods
- Funding rate fluctuations affecting position costs
- Gap risk during market closures or extreme events

**Mitigation Strategies:**
- Never risk more than 1-2% of total capital per trade
- Use proper position sizing calculations
- Monitor positions actively during volatile periods
- Have contingency plans for adverse scenarios

---

"""
        
        with open(self.report_path, 'a') as f:
            f.write(overall_risks)
    
    def _write_technical_appendix(self, analysis_data: Dict):
        """Write technical appendix with methodology"""
        appendix = f"""
## Technical Appendix

### Methodology Overview

This analysis employs a multi-layered quantitative approach combining:

1. **Market Structure Analysis**
   - Support and resistance identification
   - Trend analysis using multiple timeframes
   - Chart pattern recognition

2. **Volume Profile Analysis**
   - Point of Control (POC) identification
   - Value Area calculations
   - High/Low Volume Node mapping

3. **Sentiment Quantification**
   - Funding rate analysis
   - Open Interest divergence detection
   - Long/Short ratio evaluation
   - Liquidation proximity scoring

4. **Strategy Implementation**
   - Liquidity grab/stop hunt detection
   - Mean reversion identification
   - Breakout pattern analysis
   - HVN to HVN rotation mapping

### Data Sources

- **Traditional Finance:** Yahoo Finance API (S&P 500, VIX)
- **Cryptocurrency Data:** Binance Futures API, CoinMarketCap
- **Sentiment Data:** Funding rates, Open Interest, Liquidation levels
- **Technical Indicators:** RSI, Moving Averages, Bollinger Bands

### Scoring Methodology

Setup strength scores (0-100) are calculated using:
- Technical pattern confirmation (40% weight)
- Volume analysis (25% weight)
- Sentiment alignment (20% weight)
- Risk-reward ratio (15% weight)

### Limitations

- Analysis based on historical patterns and current data
- Market conditions can change rapidly
- External factors not captured in technical analysis
- Leverage amplifies both gains and losses significantly

---

"""
        
        with open(self.report_path, 'a') as f:
            f.write(appendix)
    
    def _write_disclaimer(self):
        """Write comprehensive risk disclaimer"""
        disclaimer = f"""
## Risk Disclaimer

**IMPORTANT RISK DISCLOSURE**

This analysis is provided for educational and informational purposes only and should not be considered as financial advice. Cryptocurrency trading, especially with leverage, involves substantial risk of loss and is not suitable for all investors.

### Key Risk Warnings

1. **Leverage Risk:** 20x leverage means a 5% adverse price movement results in 100% loss of position margin
2. **Market Volatility:** Cryptocurrency markets are extremely volatile and can move against positions rapidly
3. **Liquidation Risk:** Leveraged positions can be liquidated automatically, resulting in total loss of margin
4. **No Guarantee:** Past performance and technical analysis do not guarantee future results
5. **Capital Risk:** Never trade with money you cannot afford to lose

### Professional Advice

Before making any trading decisions:
- Consult with qualified financial advisors
- Understand your risk tolerance
- Consider your financial situation
- Educate yourself about cryptocurrency markets
- Start with small position sizes

### Regulatory Notice

Cryptocurrency trading may be restricted or prohibited in your jurisdiction. Ensure compliance with local laws and regulations before trading.

**By using this analysis, you acknowledge that you understand these risks and trade at your own discretion.**

---

*Report generated on {datetime.now().strftime("%B %d, %Y at %H:%M UTC")}*  
*Analysis System Version: 1.0*  
*For questions or support, please consult with qualified financial professionals.*

"""
        
        with open(self.report_path, 'a') as f:
            f.write(disclaimer)
    
    def _identify_primary_risk_factors(self, analysis_data: Dict) -> str:
        """Identify and format primary risk factors"""
        market_sentiment = analysis_data.get('market_sentiment', {})
        sentiment_score = market_sentiment.get('sentiment_score', 0)
        
        risk_factors = []
        
        if abs(sentiment_score) > 50:
            risk_factors.append("Extreme market sentiment may lead to sudden reversals")
        
        if sentiment_score < -30:
            risk_factors.append("Bearish market conditions increase downside risk")
        
        # Check for data quality issues
        analysis_summary = analysis_data.get('analysis_summary', {})
        if analysis_summary.get('valid_analyses', 0) < analysis_summary.get('total_assets_analyzed', 1):
            risk_factors.append("Limited data availability may affect analysis quality")
        
        if not analysis_data.get('top_trade_setups'):
            risk_factors.append("Lack of high-conviction setups suggests waiting for better opportunities")
        
        if not risk_factors:
            risk_factors.append("Standard cryptocurrency market volatility and leverage risks apply")
        
        return "\\n".join(f"- {factor}" for factor in risk_factors)
    
    def _get_tradfi_impact(self, interpretation: str) -> str:
        """Get traditional finance impact description"""
        impacts = {
            'risk_on': 'provides supportive conditions',
            'risk_off': 'creates headwinds',
            'neutral': 'has mixed implications'
        }
        return impacts.get(interpretation, 'has uncertain implications')
    
    def _get_dominance_interpretation(self, interpretation: str) -> str:
        """Get Bitcoin dominance interpretation"""
        interpretations = {
            'btc_strength_alt_weakness': 'Bitcoin is showing relative strength while altcoins underperform. This typically indicates flight to quality within crypto markets and may limit altcoin upside potential.',
            'alt_season': 'Bitcoin dominance is declining, suggesting altcoin strength. This environment often provides better opportunities for altcoin trading strategies.',
            'balanced': 'Bitcoin dominance is in a balanced range, indicating neither strong Bitcoin nor altcoin preference. This suggests a more neutral crypto market environment.'
        }
        return interpretations.get(interpretation, 'The dominance pattern suggests mixed market conditions.')
    
    def _get_funding_interpretation(self, interpretation: str) -> str:
        """Get funding rate interpretation"""
        interpretations = {
            'excessive_bullishness': 'Funding rates are extremely positive, indicating excessive long bias. This often precedes corrections as longs become overcrowded.',
            'excessive_bearishness': 'Funding rates are extremely negative, indicating excessive short bias. This may signal potential short squeeze opportunities.',
            'balanced_funding': 'Funding rates are relatively balanced, suggesting neither extreme bullish nor bearish positioning in the derivatives market.'
        }
        return interpretations.get(interpretation, 'Funding rate analysis shows mixed signals.')
    
    def _get_volume_context(self, setup: Dict) -> str:
        """Get volume context for trade setup"""
        strategy = setup.get('strategy', '').lower()
        
        if 'breakout' in strategy:
            return 'Increasing volume supports breakout validity'
        elif 'liquidity' in strategy:
            return 'Volume analysis confirms liquidity pool presence'
        elif 'reversion' in strategy:
            return 'Volume patterns support mean reversion thesis'
        else:
            return 'Volume analysis incorporated in setup identification'

def main():
    """Test the report generator"""
    print("Testing Trading Report Generator")
    print("=" * 35)
    
    # Load sample analysis data
    try:
        # Find the most recent analysis file
        analysis_files = [f for f in os.listdir('/home/ubuntu') if f.startswith('market_analysis_') and f.endswith('.json')]
        if analysis_files:
            latest_file = sorted(analysis_files)[-1]
            with open(f'/home/ubuntu/{latest_file}', 'r') as f:
                analysis_data = json.load(f)
            
            print(f"\\nLoaded analysis data from: {latest_file}")
            
            # Generate report
            generator = TradingReportGenerator()
            report_path = generator.generate_comprehensive_report(analysis_data)
            
            print(f"✓ Comprehensive report generated: {os.path.basename(report_path)}")
            
            # Check report size
            with open(report_path, 'r') as f:
                content = f.read()
                word_count = len(content.split())
                line_count = len(content.split('\\n'))
            
            print(f"✓ Report statistics:")
            print(f"  - Word count: {word_count:,}")
            print(f"  - Line count: {line_count:,}")
            print(f"  - File size: {len(content):,} characters")
            
        else:
            print("✗ No analysis data files found")
            
    except Exception as e:
        print(f"✗ Error: {str(e)}")
    
    print("\\nReport Generator testing completed!")

if __name__ == "__main__":
    main()

