#!/usr/bin/env python3
"""
Trading Strategies Module for Cryptocurrency Analysis
Implementation of proven trading strategies for leveraged perpetual contracts
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
from technical_analysis import TechnicalAnalyzer
from sentiment_analyzer import SentimentAnalyzer

logger = logging.getLogger(__name__)

class TradingStrategies:
    """Implementation of trading strategies for cryptocurrency analysis"""
    
    def __init__(self):
        self.technical_analyzer = TechnicalAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.leverage = 20  # 20x leverage as specified
    
    def detect_liquidity_grab_setup(self, highs: List[float], lows: List[float], 
                                  closes: List[float], volumes: List[float],
                                  liquidation_levels: List[Dict] = None) -> Dict:
        """
        Detect liquidity grab / stop hunt trading setups
        
        Args:
            highs: List of high prices
            lows: List of low prices
            closes: List of closing prices
            volumes: List of volume values
            liquidation_levels: Optional liquidation level data
            
        Returns:
            Dictionary containing liquidity grab analysis
        """
        if len(closes) < 50:
            return {'error': 'Insufficient data for liquidity grab analysis'}
        
        current_price = closes[-1]
        recent_highs = highs[-20:]
        recent_lows = lows[-20:]
        recent_volumes = volumes[-20:]
        
        # Identify significant liquidity pools
        # Recent swing highs and lows where stops might be placed
        swing_highs = []
        swing_lows = []
        
        for i in range(2, len(recent_highs) - 2):
            # Swing high detection
            if (recent_highs[i] > recent_highs[i-1] and recent_highs[i] > recent_highs[i-2] and
                recent_highs[i] > recent_highs[i+1] and recent_highs[i] > recent_highs[i+2]):
                swing_highs.append(recent_highs[i])
            
            # Swing low detection
            if (recent_lows[i] < recent_lows[i-1] and recent_lows[i] < recent_lows[i-2] and
                recent_lows[i] < recent_lows[i+1] and recent_lows[i] < recent_lows[i+2]):
                swing_lows.append(recent_lows[i])
        
        # Find the most significant levels (highest highs, lowest lows)
        if swing_highs:
            key_resistance = max(swing_highs)
            resistance_distance = (key_resistance - current_price) / current_price
        else:
            key_resistance = None
            resistance_distance = float('inf')
        
        if swing_lows:
            key_support = min(swing_lows)
            support_distance = (current_price - key_support) / current_price
        else:
            key_support = None
            support_distance = float('inf')
        
        # Analyze volume at key levels
        volume_analysis = self._analyze_volume_at_levels(closes, volumes, swing_highs + swing_lows)
        
        # Incorporate liquidation data if available
        liquidation_analysis = {}
        if liquidation_levels:
            liquidation_analysis = self.sentiment_analyzer.calculate_liquidation_proximity_score(
                current_price, liquidation_levels
            )
        
        # Determine setup type and direction
        setup_type = 'none'
        setup_direction = 'neutral'
        setup_strength = 0
        target_level = None
        
        # Liquidity grab above (targeting resistance)
        if key_resistance and resistance_distance < 0.05:  # Within 5%
            setup_type = 'liquidity_grab_above'
            setup_direction = 'long'  # Expecting price to grab liquidity then reverse up
            setup_strength = min(100, (0.05 - resistance_distance) * 2000)  # Closer = stronger
            target_level = key_resistance
        
        # Liquidity grab below (targeting support)
        elif key_support and support_distance < 0.05:  # Within 5%
            setup_type = 'liquidity_grab_below'
            setup_direction = 'short'  # Expecting price to grab liquidity then reverse down
            setup_strength = min(100, (0.05 - support_distance) * 2000)  # Closer = stronger
            target_level = key_support
        
        # Calculate entry and exit levels
        entry_zone = None
        take_profit_levels = []
        stop_loss = None
        
        if setup_type != 'none':
            if setup_direction == 'long':
                # Enter after liquidity grab (slight pullback from resistance)
                entry_zone = {
                    'lower': target_level * 0.995,  # 0.5% below resistance
                    'upper': target_level * 1.002   # 0.2% above resistance
                }
                # Take profits at fibonacci levels
                take_profit_levels = [
                    target_level * 1.01,   # TP1: 1% above
                    target_level * 1.025,  # TP2: 2.5% above
                    target_level * 1.05    # TP3: 5% above
                ]
                stop_loss = target_level * 0.985  # 1.5% below resistance
            
            else:  # short
                # Enter after liquidity grab (slight bounce from support)
                entry_zone = {
                    'lower': target_level * 0.998,  # 0.2% below support
                    'upper': target_level * 1.005   # 0.5% above support
                }
                # Take profits at fibonacci levels
                take_profit_levels = [
                    target_level * 0.99,   # TP1: 1% below
                    target_level * 0.975,  # TP2: 2.5% below
                    target_level * 0.95    # TP3: 5% below
                ]
                stop_loss = target_level * 1.015  # 1.5% above support
        
        return {
            'setup_type': setup_type,
            'setup_direction': setup_direction,
            'setup_strength': setup_strength,
            'target_level': target_level,
            'key_resistance': key_resistance,
            'key_support': key_support,
            'resistance_distance_pct': resistance_distance * 100,
            'support_distance_pct': support_distance * 100,
            'entry_zone': entry_zone,
            'take_profit_levels': take_profit_levels,
            'stop_loss': stop_loss,
            'volume_analysis': volume_analysis,
            'liquidation_analysis': liquidation_analysis,
            'current_price': current_price
        }
    
    def detect_mean_reversion_setup(self, closes: List[float], volumes: List[float],
                                  rsi_values: List[float] = None) -> Dict:
        """
        Detect mean reversion trading setups
        
        Args:
            closes: List of closing prices
            volumes: List of volume values
            rsi_values: Optional RSI values
            
        Returns:
            Dictionary containing mean reversion analysis
        """
        if len(closes) < 50:
            return {'error': 'Insufficient data for mean reversion analysis'}
        
        current_price = closes[-1]
        
        # Calculate moving averages
        ma_data = self.technical_analyzer.calculate_moving_averages(closes, [21, 50, 200])
        
        # Calculate RSI if not provided
        if rsi_values is None:
            rsi_values = self.technical_analyzer.calculate_rsi(closes)
        
        current_rsi = rsi_values[-1] if rsi_values[-1] is not None else 50
        
        # Calculate VWAP (simplified)
        recent_closes = closes[-20:]
        recent_volumes = volumes[-20:]
        vwap = sum(p * v for p, v in zip(recent_closes, recent_volumes)) / sum(recent_volumes)
        
        # Analyze deviations from key levels
        deviations = {}
        
        # Deviation from 200 EMA (long-term trend)
        if 'EMA_200' in ma_data and len(ma_data['EMA_200']) > 0:
            ema_200 = ma_data['EMA_200'][-1]
            if ema_200 and not np.isnan(ema_200):
                deviations['ema_200'] = (current_price - ema_200) / ema_200
        
        # Deviation from 50 EMA (medium-term trend)
        if 'EMA_50' in ma_data and len(ma_data['EMA_50']) > 0:
            ema_50 = ma_data['EMA_50'][-1]
            if ema_50 and not np.isnan(ema_50):
                deviations['ema_50'] = (current_price - ema_50) / ema_50
        
        # Deviation from VWAP
        deviations['vwap'] = (current_price - vwap) / vwap
        
        # Check for RSI divergence
        divergence_analysis = self.technical_analyzer.calculate_divergences(closes, rsi_values)
        
        # Determine setup
        setup_type = 'none'
        setup_direction = 'neutral'
        setup_strength = 0
        mean_target = None
        
        # Look for extreme deviations
        extreme_threshold = 0.05  # 5% deviation
        
        for level, deviation in deviations.items():
            if abs(deviation) > extreme_threshold:
                if deviation > 0:  # Price above mean
                    if current_rsi > 70 or divergence_analysis['bearish_divergence']:
                        setup_type = 'mean_reversion_short'
                        setup_direction = 'short'
                        setup_strength = min(100, abs(deviation) * 1000)
                        if level == 'ema_200':
                            mean_target = ma_data['EMA_200'][-1]
                        elif level == 'ema_50':
                            mean_target = ma_data['EMA_50'][-1]
                        else:
                            mean_target = vwap
                        break
                
                else:  # Price below mean
                    if current_rsi < 30 or divergence_analysis['bullish_divergence']:
                        setup_type = 'mean_reversion_long'
                        setup_direction = 'long'
                        setup_strength = min(100, abs(deviation) * 1000)
                        if level == 'ema_200':
                            mean_target = ma_data['EMA_200'][-1]
                        elif level == 'ema_50':
                            mean_target = ma_data['EMA_50'][-1]
                        else:
                            mean_target = vwap
                        break
        
        # Calculate entry and exit levels
        entry_zone = None
        take_profit_levels = []
        stop_loss = None
        
        if setup_type != 'none' and mean_target:
            if setup_direction == 'long':
                # Enter at current level or slight lower
                entry_zone = {
                    'lower': current_price * 0.995,
                    'upper': current_price * 1.005
                }
                # Target the mean and beyond
                take_profit_levels = [
                    current_price + (mean_target - current_price) * 0.5,  # TP1: 50% to mean
                    mean_target,  # TP2: Mean target
                    mean_target + (mean_target - current_price) * 0.3  # TP3: Overshoot
                ]
                stop_loss = current_price * 0.97  # 3% stop loss
            
            else:  # short
                # Enter at current level or slight higher
                entry_zone = {
                    'lower': current_price * 0.995,
                    'upper': current_price * 1.005
                }
                # Target the mean and beyond
                take_profit_levels = [
                    current_price + (mean_target - current_price) * 0.5,  # TP1: 50% to mean
                    mean_target,  # TP2: Mean target
                    mean_target + (mean_target - current_price) * 0.3  # TP3: Overshoot
                ]
                stop_loss = current_price * 1.03  # 3% stop loss
        
        return {
            'setup_type': setup_type,
            'setup_direction': setup_direction,
            'setup_strength': setup_strength,
            'mean_target': mean_target,
            'current_price': current_price,
            'current_rsi': current_rsi,
            'deviations': deviations,
            'divergence_analysis': divergence_analysis,
            'vwap': vwap,
            'entry_zone': entry_zone,
            'take_profit_levels': take_profit_levels,
            'stop_loss': stop_loss
        }
    
    def detect_breakout_setup(self, highs: List[float], lows: List[float], 
                            closes: List[float], volumes: List[float],
                            open_interest_data: List[float] = None) -> Dict:
        """
        Detect breakout from consolidation setups
        
        Args:
            highs: List of high prices
            lows: List of low prices
            closes: List of closing prices
            volumes: List of volume values
            open_interest_data: Optional open interest data
            
        Returns:
            Dictionary containing breakout analysis
        """
        if len(closes) < 50:
            return {'error': 'Insufficient data for breakout analysis'}
        
        current_price = closes[-1]
        
        # Detect chart patterns
        patterns = self.technical_analyzer.detect_chart_patterns(highs, lows, closes)
        
        # Analyze volume trends
        recent_volumes = volumes[-20:]
        avg_volume = np.mean(recent_volumes)
        current_volume = volumes[-1]
        volume_ratio = current_volume / avg_volume
        
        # Volume trend analysis
        volume_trend = np.polyfit(range(len(recent_volumes)), recent_volumes, 1)[0]
        
        # Open Interest analysis
        oi_analysis = {}
        if open_interest_data and len(open_interest_data) >= 10:
            recent_oi = open_interest_data[-10:]
            oi_trend = np.polyfit(range(len(recent_oi)), recent_oi, 1)[0]
            oi_analysis = {
                'trend': 'increasing' if oi_trend > 0 else 'decreasing',
                'trend_strength': abs(oi_trend),
                'current_oi': open_interest_data[-1]
            }
        
        # Determine breakout setup
        setup_type = 'none'
        setup_direction = 'neutral'
        setup_strength = 0
        breakout_level = None
        
        # Check consolidation range breakout
        if patterns['consolidation_range']:
            range_data = patterns['consolidation_range']
            upper_bound = range_data['upper_bound']
            lower_bound = range_data['lower_bound']
            current_position = range_data['current_position']
            
            # Near upper bound with increasing volume
            if current_position > 0.9 and volume_ratio > 1.5:
                setup_type = 'range_breakout_long'
                setup_direction = 'long'
                setup_strength = min(100, volume_ratio * 30 + (current_position - 0.9) * 1000)
                breakout_level = upper_bound
            
            # Near lower bound with increasing volume
            elif current_position < 0.1 and volume_ratio > 1.5:
                setup_type = 'range_breakout_short'
                setup_direction = 'short'
                setup_strength = min(100, volume_ratio * 30 + (0.1 - current_position) * 1000)
                breakout_level = lower_bound
        
        # Check triangle pattern breakout
        elif patterns['triangle_pattern']:
            triangle_data = patterns['triangle_pattern']
            # Triangle breakouts are typically in the direction of the prior trend
            # For simplicity, we'll use volume and OI to determine direction
            
            if volume_ratio > 2.0:  # High volume breakout
                if oi_analysis.get('trend') == 'increasing':
                    setup_type = 'triangle_breakout_long'
                    setup_direction = 'long'
                    setup_strength = min(100, volume_ratio * 25)
                    breakout_level = current_price * 1.02  # 2% above current
                else:
                    setup_type = 'triangle_breakout_short'
                    setup_direction = 'short'
                    setup_strength = min(100, volume_ratio * 25)
                    breakout_level = current_price * 0.98  # 2% below current
        
        # Calculate entry and exit levels
        entry_zone = None
        take_profit_levels = []
        stop_loss = None
        
        if setup_type != 'none' and breakout_level:
            range_size = abs(breakout_level - current_price)
            
            if setup_direction == 'long':
                entry_zone = {
                    'lower': breakout_level,
                    'upper': breakout_level * 1.01  # Enter on breakout confirmation
                }
                # Target based on range size
                take_profit_levels = [
                    breakout_level + range_size * 1.0,  # TP1: 1x range
                    breakout_level + range_size * 1.618,  # TP2: 1.618x range (fibonacci)
                    breakout_level + range_size * 2.0   # TP3: 2x range
                ]
                stop_loss = breakout_level * 0.985  # 1.5% below breakout level
            
            else:  # short
                entry_zone = {
                    'lower': breakout_level * 0.99,  # Enter on breakout confirmation
                    'upper': breakout_level
                }
                # Target based on range size
                take_profit_levels = [
                    breakout_level - range_size * 1.0,  # TP1: 1x range
                    breakout_level - range_size * 1.618,  # TP2: 1.618x range
                    breakout_level - range_size * 2.0   # TP3: 2x range
                ]
                stop_loss = breakout_level * 1.015  # 1.5% above breakout level
        
        return {
            'setup_type': setup_type,
            'setup_direction': setup_direction,
            'setup_strength': setup_strength,
            'breakout_level': breakout_level,
            'current_price': current_price,
            'patterns': patterns,
            'volume_analysis': {
                'current_volume': current_volume,
                'average_volume': avg_volume,
                'volume_ratio': volume_ratio,
                'volume_trend': 'increasing' if volume_trend > 0 else 'decreasing'
            },
            'oi_analysis': oi_analysis,
            'entry_zone': entry_zone,
            'take_profit_levels': take_profit_levels,
            'stop_loss': stop_loss
        }
    
    def detect_hvn_rotation_setup(self, closes: List[float], volumes: List[float]) -> Dict:
        """
        Detect HVN to HVN rotation setups using Volume Profile
        
        Args:
            closes: List of closing prices
            volumes: List of volume values
            
        Returns:
            Dictionary containing HVN rotation analysis
        """
        if len(closes) < 50 or len(volumes) < 50:
            return {'error': 'Insufficient data for HVN rotation analysis'}
        
        current_price = closes[-1]
        
        # Calculate volume profile
        vp_data = self.technical_analyzer.calculate_volume_profile(closes, volumes)
        
        hvn_levels = vp_data['hvn_levels']
        lvn_levels = vp_data['lvn_levels']
        poc_price = vp_data['poc_price']
        
        if len(hvn_levels) < 2:
            return {'error': 'Insufficient HVN levels for rotation analysis'}
        
        # Find current position relative to HVN levels
        hvn_levels_sorted = sorted(hvn_levels)
        current_hvn = None
        target_hvn = None
        
        # Determine which HVN level we're closest to
        distances = [(abs(current_price - hvn), hvn) for hvn in hvn_levels_sorted]
        distances.sort()
        closest_hvn = distances[0][1]
        
        # Find the next HVN level to target
        current_hvn_index = hvn_levels_sorted.index(closest_hvn)
        
        # Check if we're in an LVN zone (low volume area)
        in_lvn_zone = any(abs(current_price - lvn) / current_price < 0.02 for lvn in lvn_levels)
        
        setup_type = 'none'
        setup_direction = 'neutral'
        setup_strength = 0
        
        if in_lvn_zone:
            # We're in a low volume zone, expect move to nearest HVN
            if current_hvn_index < len(hvn_levels_sorted) - 1:
                # Can move up to next HVN
                target_hvn = hvn_levels_sorted[current_hvn_index + 1]
                if target_hvn > current_price:
                    setup_type = 'hvn_rotation_long'
                    setup_direction = 'long'
                    setup_strength = min(100, (target_hvn - current_price) / current_price * 500)
            
            if current_hvn_index > 0:
                # Can move down to previous HVN
                lower_target = hvn_levels_sorted[current_hvn_index - 1]
                if lower_target < current_price:
                    # Choose the closer target or the one with higher strength
                    if target_hvn is None or abs(current_price - lower_target) < abs(current_price - target_hvn):
                        target_hvn = lower_target
                        setup_type = 'hvn_rotation_short'
                        setup_direction = 'short'
                        setup_strength = min(100, (current_price - target_hvn) / current_price * 500)
        
        # Calculate entry and exit levels
        entry_zone = None
        take_profit_levels = []
        stop_loss = None
        
        if setup_type != 'none' and target_hvn:
            distance_to_target = abs(target_hvn - current_price)
            
            if setup_direction == 'long':
                entry_zone = {
                    'lower': current_price * 0.998,
                    'upper': current_price * 1.002
                }
                take_profit_levels = [
                    current_price + distance_to_target * 0.5,  # TP1: 50% to target
                    target_hvn,  # TP2: Target HVN
                    target_hvn + distance_to_target * 0.2  # TP3: Slight overshoot
                ]
                stop_loss = current_price * 0.975  # 2.5% stop loss
            
            else:  # short
                entry_zone = {
                    'lower': current_price * 0.998,
                    'upper': current_price * 1.002
                }
                take_profit_levels = [
                    current_price - distance_to_target * 0.5,  # TP1: 50% to target
                    target_hvn,  # TP2: Target HVN
                    target_hvn - distance_to_target * 0.2  # TP3: Slight overshoot
                ]
                stop_loss = current_price * 1.025  # 2.5% stop loss
        
        return {
            'setup_type': setup_type,
            'setup_direction': setup_direction,
            'setup_strength': setup_strength,
            'current_price': current_price,
            'current_hvn': closest_hvn,
            'target_hvn': target_hvn,
            'poc_price': poc_price,
            'hvn_levels': hvn_levels_sorted,
            'lvn_levels': lvn_levels,
            'in_lvn_zone': in_lvn_zone,
            'entry_zone': entry_zone,
            'take_profit_levels': take_profit_levels,
            'stop_loss': stop_loss,
            'volume_profile_data': vp_data
        }
    
    def _analyze_volume_at_levels(self, prices: List[float], volumes: List[float], 
                                levels: List[float]) -> Dict:
        """
        Analyze volume at specific price levels
        
        Args:
            prices: List of prices
            volumes: List of volumes
            levels: List of price levels to analyze
            
        Returns:
            Dictionary containing volume analysis at levels
        """
        level_volumes = {}
        
        for level in levels:
            # Find prices within 1% of the level
            tolerance = level * 0.01
            volume_at_level = 0
            count = 0
            
            for i, price in enumerate(prices):
                if abs(price - level) <= tolerance:
                    volume_at_level += volumes[i]
                    count += 1
            
            level_volumes[level] = {
                'total_volume': volume_at_level,
                'average_volume': volume_at_level / count if count > 0 else 0,
                'touch_count': count
            }
        
        return level_volumes
    
    def calculate_liquidation_price(self, entry_price: float, direction: str, 
                                  leverage: int = 20, margin_ratio: float = 0.05) -> float:
        """
        Calculate liquidation price for leveraged position
        
        Args:
            entry_price: Entry price of the position
            direction: 'long' or 'short'
            leverage: Leverage multiplier
            margin_ratio: Maintenance margin ratio
            
        Returns:
            Liquidation price
        """
        if direction.lower() == 'long':
            # For long positions: liquidation when price drops
            liquidation_price = entry_price * (1 - (1/leverage) + margin_ratio)
        else:  # short
            # For short positions: liquidation when price rises
            liquidation_price = entry_price * (1 + (1/leverage) - margin_ratio)
        
        return liquidation_price

def main():
    """Test the trading strategies module"""
    print("Testing Trading Strategies Module")
    print("=" * 40)
    
    # Generate sample data for testing
    np.random.seed(42)
    base_price = 50000
    returns = np.random.normal(0, 0.02, 100)
    prices = [base_price]
    
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    # Generate highs, lows, and volumes
    highs = [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices]
    lows = [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices]
    volumes = [np.random.uniform(1000, 10000) for _ in prices]
    
    strategies = TradingStrategies()
    
    # Test liquidity grab detection
    print("\\n1. Testing Liquidity Grab Detection...")
    liq_setup = strategies.detect_liquidity_grab_setup(highs, lows, prices, volumes)
    if 'error' not in liq_setup:
        print(f"✓ Setup Type: {liq_setup['setup_type']}")
        print(f"✓ Direction: {liq_setup['setup_direction']}")
        print(f"✓ Strength: {liq_setup['setup_strength']:.2f}")
    else:
        print(f"✗ Error: {liq_setup['error']}")
    
    # Test mean reversion detection
    print("\\n2. Testing Mean Reversion Detection...")
    mean_setup = strategies.detect_mean_reversion_setup(prices, volumes)
    if 'error' not in mean_setup:
        print(f"✓ Setup Type: {mean_setup['setup_type']}")
        print(f"✓ Direction: {mean_setup['setup_direction']}")
        print(f"✓ Strength: {mean_setup['setup_strength']:.2f}")
    else:
        print(f"✗ Error: {mean_setup['error']}")
    
    # Test breakout detection
    print("\\n3. Testing Breakout Detection...")
    breakout_setup = strategies.detect_breakout_setup(highs, lows, prices, volumes)
    if 'error' not in breakout_setup:
        print(f"✓ Setup Type: {breakout_setup['setup_type']}")
        print(f"✓ Direction: {breakout_setup['setup_direction']}")
        print(f"✓ Strength: {breakout_setup['setup_strength']:.2f}")
    else:
        print(f"✗ Error: {breakout_setup['error']}")
    
    # Test HVN rotation detection
    print("\\n4. Testing HVN Rotation Detection...")
    hvn_setup = strategies.detect_hvn_rotation_setup(prices, volumes)
    if 'error' not in hvn_setup:
        print(f"✓ Setup Type: {hvn_setup['setup_type']}")
        print(f"✓ Direction: {hvn_setup['setup_direction']}")
        print(f"✓ Strength: {hvn_setup['setup_strength']:.2f}")
    else:
        print(f"✗ Error: {hvn_setup['error']}")
    
    # Test liquidation price calculation
    print("\\n5. Testing Liquidation Price Calculation...")
    entry_price = 50000
    long_liq = strategies.calculate_liquidation_price(entry_price, 'long', 20)
    short_liq = strategies.calculate_liquidation_price(entry_price, 'short', 20)
    print(f"✓ Long Liquidation Price: ${long_liq:.2f}")
    print(f"✓ Short Liquidation Price: ${short_liq:.2f}")
    
    print("\\nTrading Strategies Module testing completed!")

if __name__ == "__main__":
    main()

