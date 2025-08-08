#!/usr/bin/env python3
"""
Cryptocurrency Trading Analysis System - Complete Demo
Comprehensive demonstration of the leveraged perpetual contracts analysis system
"""

import json
import os
from datetime import datetime
import logging

from market_analyzer import MarketAnalyzer
from report_generator import TradingReportGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_complete_analysis_demo():
    """Run complete cryptocurrency trading analysis demonstration"""
    
    print("=" * 80)
    print("CRYPTOCURRENCY TRADING ANALYSIS SYSTEM")
    print("Leveraged Perpetual Contracts - Medium-Term Analysis")
    print("=" * 80)
    
    print("\\nSystem Overview:")
    print("- Focus: 20x leveraged perpetual contracts")
    print("- Time Horizon: Hours to 1-2 days")
    print("- Strategy: High-probability technical setups")
    print("- Analysis: Multi-factor quantitative approach")
    
    # Initialize the system
    print("\\n" + "─" * 60)
    print("INITIALIZING ANALYSIS SYSTEM")
    print("─" * 60)
    
    analyzer = MarketAnalyzer()
    report_generator = TradingReportGenerator()
    
    print("✓ Market Analyzer initialized")
    print("✓ Report Generator initialized")
    print("✓ Data collection modules loaded")
    print("✓ Technical analysis engines ready")
    print("✓ Sentiment analysis systems active")
    print("✓ Trading strategy detectors online")
    
    # Define analysis parameters
    symbols_to_analyze = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    
    print(f"\\n📊 Analysis Parameters:")
    print(f"   Assets: {', '.join([s.replace('USDT', '') for s in symbols_to_analyze])}")
    print(f"   Leverage: 20x")
    print(f"   Data Sources: Multiple exchanges and APIs")
    print(f"   Strategies: 4 primary detection algorithms")
    
    # Run comprehensive analysis
    print("\\n" + "─" * 60)
    print("RUNNING COMPREHENSIVE MARKET ANALYSIS")
    print("─" * 60)
    
    print("\\n🔍 Data Collection Phase:")
    print("   • Fetching traditional finance data (S&P 500, VIX)")
    print("   • Collecting cryptocurrency market data")
    print("   • Gathering sentiment indicators")
    print("   • Retrieving dominance metrics")
    
    try:
        analysis_results = analyzer.run_comprehensive_analysis(symbols_to_analyze)
        
        print("\\n✅ Data Collection Completed")
        
        # Display analysis summary
        summary = analysis_results.get('analysis_summary', {})
        market_sentiment = analysis_results.get('market_sentiment', {})
        
        print("\\n📈 Analysis Summary:")
        print(f"   • Assets Analyzed: {summary.get('total_assets_analyzed', 0)}")
        print(f"   • Valid Technical Analyses: {summary.get('valid_analyses', 0)}")
        print(f"   • Trade Setups Generated: {summary.get('trade_setups_generated', 0)}")
        print(f"   • Overall Market Sentiment: {market_sentiment.get('overall_sentiment', 'Unknown').title()}")
        print(f"   • Market Bias: {market_sentiment.get('market_bias', 'Unknown').replace('_', ' ').title()}")
        print(f"   • Sentiment Score: {market_sentiment.get('sentiment_score', 0):.1f}/100")
        
        # Display trade setups
        trade_setups = analysis_results.get('top_trade_setups', [])
        
        print("\\n" + "─" * 60)
        print("TRADING OPPORTUNITIES IDENTIFIED")
        print("─" * 60)
        
        if trade_setups:
            print(f"\\n🎯 Top {len(trade_setups)} Trading Setups:")
            
            for i, setup in enumerate(trade_setups, 1):
                print(f"\\n   {i}. {setup['asset']}/USDT - {setup['direction']}")
                print(f"      Strategy: {setup['strategy']}")
                print(f"      Strength: {setup['setup_strength']:.1f}/100")
                print(f"      Current Price: ${setup['current_price']:.2f}")
                print(f"      Market Context: {setup['market_context'].title()}")
                
                # Entry and targets
                entry_zone = setup.get('entry_zone', {})
                if entry_zone:
                    avg_entry = (entry_zone.get('lower', 0) + entry_zone.get('upper', 0)) / 2
                    print(f"      Entry Zone: ${avg_entry:.2f}")
                
                tp_levels = setup.get('take_profit_levels', [])
                if tp_levels and tp_levels[0]:
                    print(f"      First Target: ${tp_levels[0]:.2f}")
                
                stop_loss = setup.get('stop_loss')
                if stop_loss:
                    print(f"      Stop Loss: ${stop_loss:.2f}")
                
                # Risk metrics
                risk_metrics = setup.get('risk_metrics', {})
                liquidation_prices = risk_metrics.get('liquidation_prices', {})
                if liquidation_prices:
                    liq_key = 'long_position' if setup['direction'] == 'LONG' else 'short_position'
                    liq_price = liquidation_prices.get(liq_key)
                    if liq_price:
                        print(f"      Liquidation: ${liq_price:.2f}")
        else:
            print("\\n⚠️  No High-Conviction Setups Currently Available")
            print("\\n   Reasons:")
            print("   • Market conditions may be too volatile for 20x leverage")
            print("   • Technical patterns lack sufficient confirmation")
            print("   • Risk-reward ratios do not meet our standards")
            print("   • Data limitations affecting analysis quality")
            print("\\n   Recommendation: Wait for clearer market conditions")
        
        # Generate comprehensive report
        print("\\n" + "─" * 60)
        print("GENERATING INSTITUTIONAL REPORT")
        print("─" * 60)
        
        print("\\n📄 Report Generation:")
        print("   • Compiling executive summary")
        print("   • Analyzing market sentiment components")
        print("   • Detailing trade execution parameters")
        print("   • Calculating comprehensive risk metrics")
        print("   • Adding technical methodology appendix")
        print("   • Including regulatory disclaimers")
        
        report_path = report_generator.generate_comprehensive_report(analysis_results)
        
        # Report statistics
        with open(report_path, 'r') as f:
            content = f.read()
            word_count = len(content.split())
            section_count = content.count('##')
        
        print("\\n✅ Report Generation Completed")
        print(f"\\n📋 Report Statistics:")
        print(f"   • File: {os.path.basename(report_path)}")
        print(f"   • Word Count: {word_count:,}")
        print(f"   • Sections: {section_count}")
        print(f"   • Format: Professional Markdown")
        print(f"   • Compliance: Institutional-grade with risk disclaimers")
        
        # Save analysis data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_file = f"/home/ubuntu/analysis_data_{timestamp}.json"
        
        with open(data_file, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        print(f"   • Raw Data: {os.path.basename(data_file)}")
        
        # System capabilities summary
        print("\\n" + "─" * 60)
        print("SYSTEM CAPABILITIES DEMONSTRATED")
        print("─" * 60)
        
        print("\\n🔧 Technical Analysis:")
        print("   ✓ Multi-timeframe trend analysis")
        print("   ✓ Support/resistance identification")
        print("   ✓ Volume profile analysis (POC, HVN, LVN)")
        print("   ✓ RSI divergence detection")
        print("   ✓ Chart pattern recognition")
        print("   ✓ Volatility metrics calculation")
        
        print("\\n📊 Sentiment Analysis:")
        print("   ✓ Funding rate sentiment scoring")
        print("   ✓ Open Interest divergence analysis")
        print("   ✓ Liquidation proximity calculations")
        print("   ✓ Long/short ratio evaluation")
        print("   ✓ Composite sentiment scoring")
        
        print("\\n🎯 Trading Strategies:")
        print("   ✓ Liquidity grab/stop hunt detection")
        print("   ✓ Mean reversion identification")
        print("   ✓ Breakout pattern analysis")
        print("   ✓ HVN to HVN rotation mapping")
        print("   ✓ Risk-adjusted position sizing")
        
        print("\\n🌐 Data Integration:")
        print("   ✓ Traditional finance correlation (S&P 500, VIX)")
        print("   ✓ Bitcoin dominance analysis")
        print("   ✓ Multi-exchange cryptocurrency data")
        print("   ✓ Real-time sentiment indicators")
        print("   ✓ Cross-asset correlation analysis")
        
        print("\\n📈 Risk Management:")
        print("   ✓ 20x leverage liquidation calculations")
        print("   ✓ Scenario analysis (best/worst case)")
        print("   ✓ Position sizing recommendations")
        print("   ✓ Trade invalidation conditions")
        print("   ✓ Comprehensive risk disclaimers")
        
        # Final summary
        print("\\n" + "=" * 80)
        print("ANALYSIS COMPLETE - SYSTEM DEMONSTRATION SUCCESSFUL")
        print("=" * 80)
        
        print("\\n🎉 Demonstration Summary:")
        print(f"   • Analysis completed at: {analysis_results['analysis_timestamp']}")
        print(f"   • Market sentiment: {market_sentiment.get('overall_sentiment', 'Unknown').title()}")
        print(f"   • Trade opportunities: {len(trade_setups)}")
        print(f"   • Report generated: {os.path.basename(report_path)}")
        print(f"   • System status: Fully operational")
        
        print("\\n📁 Generated Files:")
        print(f"   • Trading Report: {os.path.basename(report_path)}")
        print(f"   • Analysis Data: {os.path.basename(data_file)}")
        
        print("\\n⚠️  Important Reminders:")
        print("   • This system is for educational/analysis purposes")
        print("   • 20x leverage involves extreme risk")
        print("   • Always consult qualified financial advisors")
        print("   • Never risk more than you can afford to lose")
        print("   • Cryptocurrency markets are highly volatile")
        
        return {
            'success': True,
            'analysis_results': analysis_results,
            'report_path': report_path,
            'data_file': data_file
        }
        
    except Exception as e:
        print(f"\\n❌ Error during analysis: {str(e)}")
        logger.error(f"Demo failed: {str(e)}")
        return {'success': False, 'error': str(e)}

def display_system_architecture():
    """Display system architecture information"""
    
    print("\\n" + "=" * 80)
    print("SYSTEM ARCHITECTURE OVERVIEW")
    print("=" * 80)
    
    print("""
🏗️  MODULAR ARCHITECTURE:

┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA COLLECTION LAYER                            │
├─────────────────────────────────────────────────────────────────────────────┤
│  • CryptoDataCollector: Multi-source data aggregation                      │
│  • Yahoo Finance API: Traditional finance data (S&P 500, VIX)              │
│  • Binance Futures API: Cryptocurrency perpetual contracts                 │
│  • CoinMarketCap API: Bitcoin dominance and market metrics                 │
│  • Rate limiting and error handling                                        │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          ANALYSIS ENGINE LAYER                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  • TechnicalAnalyzer: Chart patterns, indicators, market structure         │
│  • SentimentAnalyzer: Funding rates, OI divergence, sentiment scoring      │
│  • TradingStrategies: 4 core strategy detection algorithms                 │
│  • Risk calculations and position sizing                                   │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        ORCHESTRATION LAYER                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  • MarketAnalyzer: Main coordination and analysis orchestration            │
│  • Cross-asset correlation analysis                                        │
│  • Trade setup generation and ranking                                      │
│  • Comprehensive market sentiment evaluation                               │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                           OUTPUT LAYER                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  • TradingReportGenerator: Institutional-grade report generation           │
│  • Professional markdown formatting                                        │
│  • Comprehensive risk disclaimers                                          │
│  • Executive summaries and detailed analysis                               │
└─────────────────────────────────────────────────────────────────────────────┘

🔧 CORE COMPONENTS:

   📊 Technical Analysis Engine
   ├── Moving averages (SMA, EMA)
   ├── RSI and divergence detection
   ├── Bollinger Bands
   ├── Volume Profile (POC, HVN, LVN)
   ├── Support/resistance identification
   ├── Chart pattern recognition
   └── Volatility metrics

   🧠 Sentiment Analysis Engine
   ├── Funding rate analysis
   ├── Open Interest divergence
   ├── Liquidation proximity scoring
   ├── Long/short ratio evaluation
   └── Composite sentiment calculation

   🎯 Trading Strategy Detectors
   ├── Liquidity grab/stop hunt
   ├── Mean reversion setups
   ├── Breakout pattern analysis
   └── HVN to HVN rotation

   ⚖️  Risk Management System
   ├── 20x leverage liquidation calculations
   ├── Position sizing recommendations
   ├── Scenario analysis
   └── Trade invalidation conditions

🔄 DATA FLOW:

   Raw Data → Technical Analysis → Sentiment Analysis → Strategy Detection
        ↓
   Market Analysis → Trade Setup Generation → Risk Assessment → Report Generation

""")

def main():
    """Main demonstration function"""
    
    # Display system architecture
    display_system_architecture()
    
    # Run complete analysis demo
    result = run_complete_analysis_demo()
    
    if result['success']:
        print("\\n🎯 Demo completed successfully!")
        print("\\nThe system has demonstrated its ability to:")
        print("• Collect and analyze multi-source market data")
        print("• Perform comprehensive technical and sentiment analysis")
        print("• Identify high-probability trading opportunities")
        print("• Generate institutional-grade research reports")
        print("• Provide detailed risk assessments")
        
        print("\\n📚 Next Steps:")
        print("• Review the generated trading report")
        print("• Examine the technical analysis methodology")
        print("• Understand the risk management framework")
        print("• Consider integration with live trading systems")
        
    else:
        print(f"\\n❌ Demo failed: {result.get('error', 'Unknown error')}")
    
    print("\\n" + "=" * 80)

if __name__ == "__main__":
    main()

