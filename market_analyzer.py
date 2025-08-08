#!/usr/bin/env python3
"""
Market Analyzer - Main orchestration module for cryptocurrency trading analysis
Comprehensive analysis system for leveraged perpetual contracts
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta

from data_collector import CryptoDataCollector
from technical_analysis import TechnicalAnalyzer
from sentiment_analyzer import SentimentAnalyzer
from trading_strategies import TradingStrategies

logger = logging.getLogger(__name__)

class MarketAnalyzer:
    """Main market analyzer for comprehensive cryptocurrency trading analysis"""
    
    def __init__(self):
        self.data_collector = CryptoDataCollector()
        self.technical_analyzer = TechnicalAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.trading_strategies = TradingStrategies()
        self.leverage = 20
    
    def analyze_market_sentiment(self, collected_data: Dict) -> Dict:
        """
        Analyze overall market sentiment from collected data
        
        Args:
            collected_data: Data collected from various sources
            
        Returns:
            Dictionary containing market sentiment analysis
        """
        sentiment_analysis = {
            'overall_sentiment': 'neutral',
            'sentiment_score': 0,
            'risk_factors': [],
            'market_bias': 'neutral'
        }
        
        try:
            # Traditional finance sentiment (S&P 500, VIX)
            tradfi_sentiment = self._analyze_tradfi_sentiment(collected_data.get('traditional_finance', {}))
            
            # Bitcoin dominance analysis
            dominance_sentiment = self._analyze_dominance_sentiment(collected_data.get('dominance_data', {}))
            
            # Crypto-specific sentiment
            crypto_sentiment = self._analyze_crypto_sentiment(collected_data.get('crypto_data', {}))
            
            # Combine sentiments
            sentiment_components = [tradfi_sentiment, dominance_sentiment, crypto_sentiment]
            valid_components = [s for s in sentiment_components if s['score'] is not None]
            
            if valid_components:
                overall_score = sum(s['score'] for s in valid_components) / len(valid_components)
                sentiment_analysis['sentiment_score'] = overall_score
                
                if overall_score > 30:
                    sentiment_analysis['overall_sentiment'] = 'bullish'
                    sentiment_analysis['market_bias'] = 'risk_on'
                elif overall_score < -30:
                    sentiment_analysis['overall_sentiment'] = 'bearish'
                    sentiment_analysis['market_bias'] = 'risk_off'
                else:
                    sentiment_analysis['overall_sentiment'] = 'neutral'
                    sentiment_analysis['market_bias'] = 'mixed'
            
            sentiment_analysis['components'] = {
                'traditional_finance': tradfi_sentiment,
                'dominance': dominance_sentiment,
                'crypto_markets': crypto_sentiment
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market sentiment: {str(e)}")
            sentiment_analysis['error'] = str(e)
        
        return sentiment_analysis
    
    def analyze_asset(self, symbol: str, asset_data: Dict) -> Dict:
        """
        Comprehensive analysis of a single cryptocurrency asset
        
        Args:
            symbol: Asset symbol (e.g., 'BTCUSDT')
            asset_data: Asset-specific data
            
        Returns:
            Dictionary containing comprehensive asset analysis
        """
        if not asset_data.get('success', False):
            return {'error': f'No valid data for {symbol}'}
        
        try:
            # Extract price data from klines
            kline_data = asset_data.get('kline_data', [])
            if not kline_data:
                return {'error': f'No kline data for {symbol}'}
            
            # Parse kline data [timestamp, open, high, low, close, volume, ...]
            timestamps = [int(k[0]) for k in kline_data]
            opens = [float(k[1]) for k in kline_data]
            highs = [float(k[2]) for k in kline_data]
            lows = [float(k[3]) for k in kline_data]
            closes = [float(k[4]) for k in kline_data]
            volumes = [float(k[5]) for k in kline_data]
            
            current_price = closes[-1]
            
            # Technical analysis
            technical_analysis = self.technical_analyzer.analyze_market_structure(highs, lows, closes)
            
            # Volume profile analysis
            volume_profile = self.technical_analyzer.calculate_volume_profile(closes, volumes)
            
            # Sentiment analysis (using available data)
            funding_data = asset_data.get('funding_data', [])
            funding_rates = [float(f['fundingRate']) for f in funding_data] if funding_data else []
            
            sentiment_data = {}
            if funding_rates:
                sentiment_data['funding'] = self.sentiment_analyzer.calculate_funding_rate_sentiment(funding_rates)
            
            # Open Interest analysis
            oi_data = asset_data.get('open_interest', {})
            if oi_data:
                # For OI divergence, we need historical OI data
                # For now, we'll use current OI info
                sentiment_data['oi_info'] = {
                    'current_oi': float(oi_data.get('openInterest', 0)),
                    'symbol': oi_data.get('symbol', symbol)
                }
            
            # Trading strategy analysis
            strategy_analysis = self._analyze_trading_strategies(highs, lows, closes, volumes)
            
            # Risk metrics
            risk_metrics = self._calculate_risk_metrics(closes, technical_analysis)
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'technical_analysis': technical_analysis,
                'volume_profile': volume_profile,
                'sentiment_data': sentiment_data,
                'strategy_analysis': strategy_analysis,
                'risk_metrics': risk_metrics,
                'market_data': {
                    'ticker_data': asset_data.get('ticker_data', {}),
                    'funding_data': funding_data,
                    'open_interest': oi_data
                },
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing asset {symbol}: {str(e)}")
            return {'error': f'Analysis failed for {symbol}: {str(e)}'}
    
    def generate_trade_setups(self, asset_analyses: Dict, market_sentiment: Dict) -> List[Dict]:
        """
        Generate top trading setups based on analysis
        
        Args:
            asset_analyses: Dictionary of asset analyses
            market_sentiment: Overall market sentiment
            
        Returns:
            List of top trading setups
        """
        all_setups = []
        
        for symbol, analysis in asset_analyses.items():
            if 'error' in analysis:
                continue
            
            try:
                strategy_analysis = analysis.get('strategy_analysis', {})
                current_price = analysis.get('current_price', 0)
                
                # Extract best setups from each strategy
                for strategy_name, strategy_data in strategy_analysis.items():
                    if strategy_data.get('setup_type', 'none') != 'none':
                        setup = self._create_trade_setup(
                            symbol, strategy_name, strategy_data, analysis, market_sentiment
                        )
                        if setup:
                            all_setups.append(setup)
            
            except Exception as e:
                logger.error(f"Error generating setups for {symbol}: {str(e)}")
        
        # Sort setups by strength and filter top 3
        all_setups.sort(key=lambda x: x.get('setup_strength', 0), reverse=True)
        return all_setups[:3]
    
    def _analyze_tradfi_sentiment(self, tradfi_data: Dict) -> Dict:
        """Analyze traditional finance sentiment"""
        sentiment = {'score': None, 'interpretation': 'neutral'}
        
        try:
            spy_data = tradfi_data.get('SPY', {})
            vix_data = tradfi_data.get('VIX', {})
            
            if spy_data.get('success') and 'meta' in spy_data:
                spy_meta = spy_data['meta']
                price_change_pct = float(spy_meta.get('regularMarketChangePercent', 0))
                
                # VIX analysis
                vix_score = 0
                if vix_data.get('success') and 'meta' in vix_data:
                    vix_price = float(vix_data['meta'].get('regularMarketPrice', 20))
                    if vix_price > 30:
                        vix_score = -30  # High VIX = fear
                    elif vix_price < 15:
                        vix_score = 20   # Low VIX = complacency
                
                # Combine SPY and VIX
                sentiment['score'] = price_change_pct * 10 + vix_score
                
                if sentiment['score'] > 10:
                    sentiment['interpretation'] = 'risk_on'
                elif sentiment['score'] < -10:
                    sentiment['interpretation'] = 'risk_off'
                
        except Exception as e:
            logger.error(f"Error analyzing TradFi sentiment: {str(e)}")
        
        return sentiment
    
    def _analyze_dominance_sentiment(self, dominance_data: Dict) -> Dict:
        """Analyze Bitcoin dominance sentiment"""
        sentiment = {'score': None, 'interpretation': 'neutral'}
        
        try:
            if dominance_data.get('success'):
                btc_dominance = dominance_data.get('btc_dominance', 50)
                
                # BTC dominance interpretation
                if btc_dominance > 65:
                    sentiment['score'] = -20  # High dominance = altcoin weakness
                    sentiment['interpretation'] = 'btc_strength_alt_weakness'
                elif btc_dominance < 40:
                    sentiment['score'] = 20   # Low dominance = altcoin strength
                    sentiment['interpretation'] = 'alt_season'
                else:
                    sentiment['score'] = 0
                    sentiment['interpretation'] = 'balanced'
                
        except Exception as e:
            logger.error(f"Error analyzing dominance sentiment: {str(e)}")
        
        return sentiment
    
    def _analyze_crypto_sentiment(self, crypto_data: Dict) -> Dict:
        """Analyze cryptocurrency-specific sentiment"""
        sentiment = {'score': None, 'interpretation': 'neutral'}
        
        try:
            # Analyze funding rates across assets
            funding_scores = []
            
            for symbol, data in crypto_data.items():
                if data.get('success') and 'funding_data' in data:
                    funding_data = data['funding_data']
                    if funding_data:
                        latest_funding = float(funding_data[0]['fundingRate'])
                        funding_scores.append(latest_funding * 10000)  # Scale to readable numbers
            
            if funding_scores:
                avg_funding = sum(funding_scores) / len(funding_scores)
                sentiment['score'] = avg_funding
                
                if avg_funding > 5:
                    sentiment['interpretation'] = 'excessive_bullishness'
                elif avg_funding < -5:
                    sentiment['interpretation'] = 'excessive_bearishness'
                else:
                    sentiment['interpretation'] = 'balanced_funding'
                
        except Exception as e:
            logger.error(f"Error analyzing crypto sentiment: {str(e)}")
        
        return sentiment
    
    def _analyze_trading_strategies(self, highs: List[float], lows: List[float], 
                                  closes: List[float], volumes: List[float]) -> Dict:
        """Analyze all trading strategies for an asset"""
        strategies = {}
        
        try:
            # Liquidity grab analysis
            strategies['liquidity_grab'] = self.trading_strategies.detect_liquidity_grab_setup(
                highs, lows, closes, volumes
            )
            
            # Mean reversion analysis
            strategies['mean_reversion'] = self.trading_strategies.detect_mean_reversion_setup(
                closes, volumes
            )
            
            # Breakout analysis
            strategies['breakout'] = self.trading_strategies.detect_breakout_setup(
                highs, lows, closes, volumes
            )
            
            # HVN rotation analysis
            strategies['hvn_rotation'] = self.trading_strategies.detect_hvn_rotation_setup(
                closes, volumes
            )
            
        except Exception as e:
            logger.error(f"Error analyzing trading strategies: {str(e)}")
            strategies['error'] = str(e)
        
        return strategies
    
    def _calculate_risk_metrics(self, closes: List[float], technical_analysis: Dict) -> Dict:
        """Calculate risk metrics for an asset"""
        risk_metrics = {}
        
        try:
            current_price = closes[-1]
            
            # Volatility metrics
            volatility = technical_analysis.get('volatility_metrics', {})
            current_vol = volatility.get('current_volatility', 0)
            
            # Risk classification
            if current_vol > 100:  # High volatility
                risk_level = 'high'
            elif current_vol > 50:
                risk_level = 'medium'
            else:
                risk_level = 'low'
            
            # Liquidation prices for 20x leverage
            long_liquidation = self.trading_strategies.calculate_liquidation_price(
                current_price, 'long', self.leverage
            )
            short_liquidation = self.trading_strategies.calculate_liquidation_price(
                current_price, 'short', self.leverage
            )
            
            risk_metrics = {
                'volatility_level': risk_level,
                'current_volatility': current_vol,
                'liquidation_prices': {
                    'long_position': long_liquidation,
                    'short_position': short_liquidation
                },
                'distance_to_liquidation': {
                    'long_pct': (current_price - long_liquidation) / current_price * 100,
                    'short_pct': (short_liquidation - current_price) / current_price * 100
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")
            risk_metrics['error'] = str(e)
        
        return risk_metrics
    
    def _create_trade_setup(self, symbol: str, strategy_name: str, strategy_data: Dict,
                          analysis: Dict, market_sentiment: Dict) -> Optional[Dict]:
        """Create a formatted trade setup"""
        try:
            setup_strength = strategy_data.get('setup_strength', 0)
            if setup_strength < 20:  # Filter weak setups
                return None
            
            current_price = analysis.get('current_price', 0)
            risk_metrics = analysis.get('risk_metrics', {})
            
            # Adjust strength based on market sentiment
            sentiment_score = market_sentiment.get('sentiment_score', 0)
            direction = strategy_data.get('setup_direction', 'neutral')
            
            # Boost strength if setup aligns with market sentiment
            if direction == 'long' and sentiment_score > 0:
                setup_strength *= 1.2
            elif direction == 'short' and sentiment_score < 0:
                setup_strength *= 1.2
            elif direction == 'long' and sentiment_score < -20:
                setup_strength *= 0.8  # Reduce strength if against strong bearish sentiment
            elif direction == 'short' and sentiment_score > 20:
                setup_strength *= 0.8  # Reduce strength if against strong bullish sentiment
            
            # Create trade setup
            trade_setup = {
                'asset': symbol.replace('USDT', ''),
                'direction': direction.upper(),
                'strategy': strategy_name.replace('_', ' ').title(),
                'setup_strength': min(100, setup_strength),
                'current_price': current_price,
                'entry_zone': strategy_data.get('entry_zone'),
                'take_profit_levels': strategy_data.get('take_profit_levels', []),
                'stop_loss': strategy_data.get('stop_loss'),
                'leverage': self.leverage,
                'risk_metrics': risk_metrics,
                'strategy_rationale': self._generate_rationale(strategy_name, strategy_data, analysis),
                'market_context': market_sentiment.get('overall_sentiment', 'neutral')
            }
            
            return trade_setup
            
        except Exception as e:
            logger.error(f"Error creating trade setup: {str(e)}")
            return None
    
    def _generate_rationale(self, strategy_name: str, strategy_data: Dict, analysis: Dict) -> str:
        """Generate trading rationale for a setup"""
        rationales = {
            'liquidity_grab': f"Liquidity grab setup targeting {strategy_data.get('setup_direction', 'neutral')} side. "
                            f"Price approaching key level at ${strategy_data.get('target_level', 0):.2f}.",
            
            'mean_reversion': f"Mean reversion setup with price showing extreme deviation from "
                            f"${strategy_data.get('mean_target', 0):.2f} target level.",
            
            'breakout': f"Breakout setup from consolidation pattern. Volume increasing with "
                       f"breakout level at ${strategy_data.get('breakout_level', 0):.2f}.",
            
            'hvn_rotation': f"Volume profile rotation from LVN to HVN at "
                          f"${strategy_data.get('target_hvn', 0):.2f}."
        }
        
        base_rationale = rationales.get(strategy_name, "Technical setup identified.")
        
        # Add technical context
        technical = analysis.get('technical_analysis', {})
        current_rsi = technical.get('current_rsi', 50)
        
        if current_rsi > 70:
            base_rationale += " RSI showing overbought conditions."
        elif current_rsi < 30:
            base_rationale += " RSI showing oversold conditions."
        
        return base_rationale
    
    def run_comprehensive_analysis(self, symbols: List[str] = None) -> Dict:
        """
        Run comprehensive market analysis
        
        Args:
            symbols: List of symbols to analyze
            
        Returns:
            Dictionary containing complete analysis
        """
        if symbols is None:
            symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
        
        logger.info("Starting comprehensive market analysis")
        
        # Collect data
        collected_data = self.data_collector.collect_comprehensive_data(symbols)
        
        # Analyze market sentiment
        market_sentiment = self.analyze_market_sentiment(collected_data)
        
        # Analyze individual assets
        asset_analyses = {}
        for symbol in symbols:
            if symbol in collected_data['crypto_data']:
                asset_analyses[symbol] = self.analyze_asset(symbol, collected_data['crypto_data'][symbol])
        
        # Generate trade setups
        trade_setups = self.generate_trade_setups(asset_analyses, market_sentiment)
        
        # Compile comprehensive analysis
        comprehensive_analysis = {
            'analysis_timestamp': datetime.now().isoformat(),
            'market_sentiment': market_sentiment,
            'asset_analyses': asset_analyses,
            'top_trade_setups': trade_setups,
            'raw_data': collected_data,
            'analysis_summary': {
                'total_assets_analyzed': len(asset_analyses),
                'valid_analyses': len([a for a in asset_analyses.values() if 'error' not in a]),
                'trade_setups_generated': len(trade_setups),
                'overall_market_bias': market_sentiment.get('market_bias', 'neutral')
            }
        }
        
        logger.info("Comprehensive market analysis completed")
        return comprehensive_analysis

def main():
    """Test the market analyzer"""
    print("Testing Market Analyzer")
    print("=" * 30)
    
    analyzer = MarketAnalyzer()
    
    print("\\nRunning comprehensive analysis...")
    analysis = analyzer.run_comprehensive_analysis(['BTCUSDT', 'ETHUSDT'])
    
    print(f"\\n✓ Analysis completed at: {analysis['analysis_timestamp']}")
    print(f"✓ Market sentiment: {analysis['market_sentiment']['overall_sentiment']}")
    print(f"✓ Assets analyzed: {analysis['analysis_summary']['total_assets_analyzed']}")
    print(f"✓ Trade setups generated: {analysis['analysis_summary']['trade_setups_generated']}")
    
    # Display top trade setups
    if analysis['top_trade_setups']:
        print("\\nTop Trade Setups:")
        for i, setup in enumerate(analysis['top_trade_setups'], 1):
            print(f"  {i}. {setup['asset']} {setup['direction']} - {setup['strategy']}")
            print(f"     Strength: {setup['setup_strength']:.1f}/100")
            print(f"     Current Price: ${setup['current_price']:.2f}")
    
    # Save analysis to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"market_analysis_{timestamp}.json"
    
    with open(f"/home/ubuntu/{filename}", 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    
    print(f"\\n✓ Analysis saved to: {filename}")
    print("\\nMarket Analyzer testing completed!")

if __name__ == "__main__":
    main()

