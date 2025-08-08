#!/usr/bin/env python3
"""
Sentiment Analysis Module for Cryptocurrency Trading
Quantitative sentiment indicators and market bias calculations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Sentiment analysis and quantitative metrics for crypto trading"""
    
    def __init__(self):
        self.sentiment_data = {}
    
    def calculate_funding_rate_sentiment(self, funding_rates: List[float], 
                                       timestamps: List[str] = None) -> Dict:
        """
        Calculate sentiment based on funding rates
        
        Args:
            funding_rates: List of funding rate values
            timestamps: Optional timestamps for the funding rates
            
        Returns:
            Dictionary containing funding rate sentiment analysis
        """
        if not funding_rates:
            return {'error': 'No funding rate data provided'}
        
        # Convert to numpy array for calculations
        rates = np.array(funding_rates)
        
        # Current funding rate
        current_rate = rates[-1] if len(rates) > 0 else 0
        
        # Average funding rate over period
        avg_rate = np.mean(rates)
        
        # Funding rate trend (slope of recent rates)
        if len(rates) >= 5:
            recent_rates = rates[-5:]
            x = np.arange(len(recent_rates))
            trend_slope = np.polyfit(x, recent_rates, 1)[0]
        else:
            trend_slope = 0
        
        # Sentiment classification based on funding rates
        # Positive funding = longs pay shorts (bullish bias)
        # Negative funding = shorts pay longs (bearish bias)
        
        sentiment_score = 0
        sentiment_label = 'neutral'
        
        if current_rate > 0.01:  # 1% funding rate
            sentiment_score = min(100, current_rate * 10000)  # Scale to 0-100
            sentiment_label = 'extremely_bullish'
        elif current_rate > 0.005:  # 0.5% funding rate
            sentiment_score = current_rate * 10000
            sentiment_label = 'very_bullish'
        elif current_rate > 0.001:  # 0.1% funding rate
            sentiment_score = current_rate * 5000
            sentiment_label = 'bullish'
        elif current_rate < -0.01:  # -1% funding rate
            sentiment_score = max(-100, current_rate * 10000)
            sentiment_label = 'extremely_bearish'
        elif current_rate < -0.005:  # -0.5% funding rate
            sentiment_score = current_rate * 10000
            sentiment_label = 'very_bearish'
        elif current_rate < -0.001:  # -0.1% funding rate
            sentiment_score = current_rate * 5000
            sentiment_label = 'bearish'
        
        # Persistence score (how long has sentiment been in this direction)
        persistence_score = 0
        if len(rates) >= 10:
            recent_10 = rates[-10:]
            if current_rate > 0:
                persistence_score = sum(1 for r in recent_10 if r > 0) / len(recent_10)
            else:
                persistence_score = sum(1 for r in recent_10 if r < 0) / len(recent_10)
        
        return {
            'current_funding_rate': current_rate,
            'average_funding_rate': avg_rate,
            'funding_trend_slope': trend_slope,
            'sentiment_score': sentiment_score,
            'sentiment_label': sentiment_label,
            'persistence_score': persistence_score,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def calculate_oi_divergence_score(self, prices: List[float], 
                                    open_interest: List[float]) -> Dict:
        """
        Calculate Open Interest divergence with price
        
        Args:
            prices: List of price values
            open_interest: List of open interest values
            
        Returns:
            Dictionary containing OI divergence analysis
        """
        if len(prices) != len(open_interest) or len(prices) < 10:
            return {'error': 'Insufficient or mismatched data'}
        
        # Calculate price and OI changes
        price_changes = np.diff(prices)
        oi_changes = np.diff(open_interest)
        
        # Calculate correlation between price and OI changes
        if len(price_changes) > 1:
            correlation = np.corrcoef(price_changes, oi_changes)[0, 1]
        else:
            correlation = 0
        
        # Recent divergence analysis (last 10 periods)
        recent_periods = min(10, len(price_changes))
        recent_price_changes = price_changes[-recent_periods:]
        recent_oi_changes = oi_changes[-recent_periods:]
        
        # Count divergent periods
        divergent_periods = 0
        for i in range(len(recent_price_changes)):
            price_direction = 1 if recent_price_changes[i] > 0 else -1
            oi_direction = 1 if recent_oi_changes[i] > 0 else -1
            
            if price_direction != oi_direction:
                divergent_periods += 1
        
        divergence_ratio = divergent_periods / len(recent_price_changes)
        
        # Divergence strength
        if divergence_ratio > 0.7:
            divergence_strength = 'strong'
        elif divergence_ratio > 0.5:
            divergence_strength = 'moderate'
        elif divergence_ratio > 0.3:
            divergence_strength = 'weak'
        else:
            divergence_strength = 'none'
        
        # Current trend analysis
        current_price_trend = 'up' if recent_price_changes[-1] > 0 else 'down'
        current_oi_trend = 'up' if recent_oi_changes[-1] > 0 else 'down'
        
        # Interpretation
        interpretation = 'neutral'
        if current_price_trend == 'up' and current_oi_trend == 'down':
            interpretation = 'bearish_divergence'  # Price up, OI down = weak rally
        elif current_price_trend == 'down' and current_oi_trend == 'up':
            interpretation = 'bullish_divergence'  # Price down, OI up = potential reversal
        elif current_price_trend == 'up' and current_oi_trend == 'up':
            interpretation = 'bullish_confirmation'  # Both up = strong trend
        elif current_price_trend == 'down' and current_oi_trend == 'down':
            interpretation = 'bearish_confirmation'  # Both down = strong downtrend
        
        return {
            'correlation': correlation,
            'divergence_ratio': divergence_ratio,
            'divergence_strength': divergence_strength,
            'current_price_trend': current_price_trend,
            'current_oi_trend': current_oi_trend,
            'interpretation': interpretation,
            'divergent_periods': divergent_periods,
            'total_periods_analyzed': len(recent_price_changes)
        }
    
    def calculate_liquidation_proximity_score(self, current_price: float,
                                            liquidation_levels: List[Dict]) -> Dict:
        """
        Calculate proximity to liquidation clusters
        
        Args:
            current_price: Current asset price
            liquidation_levels: List of liquidation level dictionaries
                                Each dict should have 'price' and 'volume' keys
            
        Returns:
            Dictionary containing liquidation proximity analysis
        """
        if not liquidation_levels:
            return {'error': 'No liquidation data provided'}
        
        # Calculate distances to liquidation levels
        liquidation_analysis = []
        
        for level in liquidation_levels:
            if 'price' not in level or 'volume' not in level:
                continue
            
            liq_price = level['price']
            liq_volume = level['volume']
            
            # Distance to liquidation level
            distance = abs(current_price - liq_price) / current_price
            direction = 'above' if liq_price > current_price else 'below'
            
            # Magnet strength based on volume and proximity
            magnet_strength = liq_volume / (1 + distance * 100)  # Higher volume and closer = stronger magnet
            
            liquidation_analysis.append({
                'price': liq_price,
                'volume': liq_volume,
                'distance_pct': distance * 100,
                'direction': direction,
                'magnet_strength': magnet_strength
            })
        
        # Sort by magnet strength
        liquidation_analysis.sort(key=lambda x: x['magnet_strength'], reverse=True)
        
        # Find closest significant liquidation levels
        close_liquidations = [liq for liq in liquidation_analysis if liq['distance_pct'] < 5.0]  # Within 5%
        
        # Calculate overall proximity score
        if close_liquidations:
            proximity_score = sum(liq['magnet_strength'] for liq in close_liquidations[:3])  # Top 3
            nearest_liquidation = close_liquidations[0]
        else:
            proximity_score = 0
            nearest_liquidation = liquidation_analysis[0] if liquidation_analysis else None
        
        return {
            'proximity_score': proximity_score,
            'nearest_liquidation': nearest_liquidation,
            'close_liquidations': close_liquidations,
            'all_liquidations': liquidation_analysis[:10],  # Top 10 by strength
            'total_liquidation_levels': len(liquidation_levels)
        }
    
    def calculate_long_short_ratio_sentiment(self, long_positions: float,
                                           short_positions: float,
                                           historical_ratios: List[float] = None) -> Dict:
        """
        Calculate sentiment based on long/short ratios
        
        Args:
            long_positions: Current long position percentage
            short_positions: Current short position percentage
            historical_ratios: Optional historical long/short ratios
            
        Returns:
            Dictionary containing long/short ratio sentiment
        """
        if long_positions + short_positions == 0:
            return {'error': 'Invalid position data'}
        
        # Current long/short ratio
        total_positions = long_positions + short_positions
        long_ratio = long_positions / total_positions
        short_ratio = short_positions / total_positions
        
        # Sentiment based on ratio
        sentiment_score = (long_ratio - 0.5) * 200  # Scale to -100 to +100
        
        if sentiment_score > 30:
            sentiment_label = 'very_bullish'
        elif sentiment_score > 10:
            sentiment_label = 'bullish'
        elif sentiment_score > -10:
            sentiment_label = 'neutral'
        elif sentiment_score > -30:
            sentiment_label = 'bearish'
        else:
            sentiment_label = 'very_bearish'
        
        # Contrarian indicator (extreme ratios often signal reversals)
        contrarian_signal = 'none'
        if long_ratio > 0.8:
            contrarian_signal = 'bearish_extreme'  # Too many longs = potential reversal down
        elif long_ratio < 0.2:
            contrarian_signal = 'bullish_extreme'  # Too many shorts = potential reversal up
        
        # Historical comparison
        historical_analysis = {}
        if historical_ratios:
            avg_historical_ratio = np.mean(historical_ratios)
            std_historical_ratio = np.std(historical_ratios)
            
            z_score = (long_ratio - avg_historical_ratio) / std_historical_ratio if std_historical_ratio > 0 else 0
            
            historical_analysis = {
                'average_long_ratio': avg_historical_ratio,
                'current_vs_average': long_ratio - avg_historical_ratio,
                'z_score': z_score,
                'percentile': sum(1 for r in historical_ratios if r < long_ratio) / len(historical_ratios) * 100
            }
        
        return {
            'long_ratio': long_ratio,
            'short_ratio': short_ratio,
            'sentiment_score': sentiment_score,
            'sentiment_label': sentiment_label,
            'contrarian_signal': contrarian_signal,
            'historical_analysis': historical_analysis
        }
    
    def calculate_composite_sentiment_score(self, funding_sentiment: Dict,
                                          oi_divergence: Dict,
                                          liquidation_proximity: Dict,
                                          long_short_sentiment: Dict,
                                          rsi_value: float = 50) -> Dict:
        """
        Calculate composite sentiment score from all indicators
        
        Args:
            funding_sentiment: Funding rate sentiment data
            oi_divergence: OI divergence data
            liquidation_proximity: Liquidation proximity data
            long_short_sentiment: Long/short ratio sentiment data
            rsi_value: Current RSI value
            
        Returns:
            Dictionary containing composite sentiment analysis
        """
        scores = []
        weights = []
        
        # Funding rate sentiment (weight: 30%)
        if 'sentiment_score' in funding_sentiment:
            scores.append(funding_sentiment['sentiment_score'])
            weights.append(0.30)
        
        # Long/short ratio sentiment (weight: 25%)
        if 'sentiment_score' in long_short_sentiment:
            scores.append(long_short_sentiment['sentiment_score'])
            weights.append(0.25)
        
        # RSI sentiment (weight: 20%)
        rsi_sentiment = (rsi_value - 50) * 2  # Scale RSI to -100 to +100
        scores.append(rsi_sentiment)
        weights.append(0.20)
        
        # OI divergence sentiment (weight: 15%)
        oi_score = 0
        if 'interpretation' in oi_divergence:
            if oi_divergence['interpretation'] == 'bullish_confirmation':
                oi_score = 50
            elif oi_divergence['interpretation'] == 'bullish_divergence':
                oi_score = 30
            elif oi_divergence['interpretation'] == 'bearish_confirmation':
                oi_score = -50
            elif oi_divergence['interpretation'] == 'bearish_divergence':
                oi_score = -30
            
            scores.append(oi_score)
            weights.append(0.15)
        
        # Liquidation proximity (weight: 10%)
        prox_score = 0
        if 'proximity_score' in liquidation_proximity:
            # Normalize proximity score to -100 to +100 range
            prox_score = min(100, max(-100, liquidation_proximity['proximity_score'] * 10))
            scores.append(prox_score)
            weights.append(0.10)
        
        # Calculate weighted average
        if scores and weights:
            # Normalize weights
            total_weight = sum(weights)
            normalized_weights = [w / total_weight for w in weights]
            
            composite_score = sum(score * weight for score, weight in zip(scores, normalized_weights))
        else:
            composite_score = 0
        
        # Classify composite sentiment
        if composite_score > 60:
            composite_label = 'very_bullish'
        elif composite_score > 30:
            composite_label = 'bullish'
        elif composite_score > 10:
            composite_label = 'slightly_bullish'
        elif composite_score > -10:
            composite_label = 'neutral'
        elif composite_score > -30:
            composite_label = 'slightly_bearish'
        elif composite_score > -60:
            composite_label = 'bearish'
        else:
            composite_label = 'very_bearish'
        
        # Risk assessment
        risk_level = 'medium'
        if abs(composite_score) > 70:
            risk_level = 'high'  # Extreme sentiment = higher risk
        elif abs(composite_score) < 20:
            risk_level = 'low'   # Neutral sentiment = lower risk
        
        return {
            'composite_score': composite_score,
            'composite_label': composite_label,
            'risk_level': risk_level,
            'component_scores': {
                'funding_rate': funding_sentiment.get('sentiment_score', 0),
                'long_short_ratio': long_short_sentiment.get('sentiment_score', 0),
                'rsi': rsi_sentiment,
                'oi_divergence': oi_score,
                'liquidation_proximity': prox_score
            },
            'weights_used': dict(zip(['funding', 'long_short', 'rsi', 'oi_divergence', 'liquidation'], 
                                   normalized_weights if 'normalized_weights' in locals() else [])),
            'analysis_timestamp': datetime.now().isoformat()
        }

def main():
    """Test the sentiment analysis module"""
    print("Testing Sentiment Analysis Module")
    print("=" * 40)
    
    analyzer = SentimentAnalyzer()
    
    # Test funding rate sentiment
    print("\\n1. Testing Funding Rate Sentiment...")
    funding_rates = [0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004]  # Increasing positive funding
    funding_sentiment = analyzer.calculate_funding_rate_sentiment(funding_rates)
    print(f"✓ Funding Sentiment: {funding_sentiment['sentiment_label']}")
    print(f"✓ Sentiment Score: {funding_sentiment['sentiment_score']:.2f}")
    
    # Test OI divergence
    print("\\n2. Testing OI Divergence...")
    prices = [50000, 51000, 52000, 51500, 51000, 50500, 50000]
    open_interest = [100000, 95000, 90000, 92000, 95000, 98000, 101000]  # OI increasing while price declining
    oi_divergence = analyzer.calculate_oi_divergence_score(prices, open_interest)
    if 'interpretation' in oi_divergence:
        print(f"✓ OI Divergence: {oi_divergence['interpretation']}")
        print(f"✓ Divergence Strength: {oi_divergence['divergence_strength']}")
    else:
        print(f"✗ OI Divergence Error: {oi_divergence.get('error', 'Unknown error')}")
    
    # Test liquidation proximity
    print("\\n3. Testing Liquidation Proximity...")
    current_price = 50000
    liquidation_levels = [
        {'price': 48000, 'volume': 1000000},
        {'price': 52000, 'volume': 800000},
        {'price': 47000, 'volume': 1200000},
        {'price': 53000, 'volume': 600000}
    ]
    liq_proximity = analyzer.calculate_liquidation_proximity_score(current_price, liquidation_levels)
    print(f"✓ Proximity Score: {liq_proximity['proximity_score']:.2f}")
    if liq_proximity['nearest_liquidation']:
        print(f"✓ Nearest Liquidation: ${liq_proximity['nearest_liquidation']['price']:.0f}")
    
    # Test long/short ratio sentiment
    print("\\n4. Testing Long/Short Ratio Sentiment...")
    long_short_sentiment = analyzer.calculate_long_short_ratio_sentiment(70, 30)  # 70% long, 30% short
    print(f"✓ Long/Short Sentiment: {long_short_sentiment['sentiment_label']}")
    print(f"✓ Contrarian Signal: {long_short_sentiment['contrarian_signal']}")
    
    # Test composite sentiment
    print("\\n5. Testing Composite Sentiment...")
    composite = analyzer.calculate_composite_sentiment_score(
        funding_sentiment, oi_divergence, liq_proximity, long_short_sentiment, rsi_value=65
    )
    print(f"✓ Composite Sentiment: {composite['composite_label']}")
    print(f"✓ Composite Score: {composite['composite_score']:.2f}")
    print(f"✓ Risk Level: {composite['risk_level']}")
    
    print("\\nSentiment Analysis Module testing completed!")

if __name__ == "__main__":
    main()

