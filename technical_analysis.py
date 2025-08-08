#!/usr/bin/env python3
"""
Technical Analysis Module for Cryptocurrency Trading
Comprehensive technical analysis calculations for leveraged perpetual contracts
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class TechnicalAnalyzer:
    """Technical analysis calculations for cryptocurrency trading"""
    
    def __init__(self):
        self.indicators = {}
    
    def calculate_moving_averages(self, prices: List[float], periods: List[int] = None) -> Dict:
        """
        Calculate various moving averages
        
        Args:
            prices: List of price values
            periods: List of periods for moving averages
            
        Returns:
            Dictionary containing moving averages
        """
        if periods is None:
            periods = [21, 50, 200]
        
        df = pd.DataFrame({'price': prices})
        ma_data = {}
        
        for period in periods:
            if len(prices) >= period:
                # Simple Moving Average
                ma_data[f'SMA_{period}'] = df['price'].rolling(window=period).mean().tolist()
                
                # Exponential Moving Average
                ma_data[f'EMA_{period}'] = df['price'].ewm(span=period).mean().tolist()
        
        return ma_data
    
    def calculate_rsi(self, prices: List[float], period: int = 14) -> List[float]:
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            prices: List of price values
            period: RSI period (default 14)
            
        Returns:
            List of RSI values
        """
        if len(prices) < period + 1:
            return [None] * len(prices)
        
        df = pd.DataFrame({'price': prices})
        delta = df['price'].diff()
        
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.tolist()
    
    def calculate_bollinger_bands(self, prices: List[float], period: int = 20, std_dev: float = 2) -> Dict:
        """
        Calculate Bollinger Bands
        
        Args:
            prices: List of price values
            period: Moving average period
            std_dev: Standard deviation multiplier
            
        Returns:
            Dictionary with upper, middle, and lower bands
        """
        df = pd.DataFrame({'price': prices})
        
        middle_band = df['price'].rolling(window=period).mean()
        std = df['price'].rolling(window=period).std()
        
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        return {
            'upper_band': upper_band.tolist(),
            'middle_band': middle_band.tolist(),
            'lower_band': lower_band.tolist()
        }
    
    def calculate_volume_profile(self, prices: List[float], volumes: List[float], bins: int = 50) -> Dict:
        """
        Calculate Volume Profile (VPVR)
        
        Args:
            prices: List of price values
            volumes: List of volume values
            bins: Number of price bins
            
        Returns:
            Dictionary containing volume profile data
        """
        if len(prices) != len(volumes):
            raise ValueError("Prices and volumes must have the same length")
        
        df = pd.DataFrame({'price': prices, 'volume': volumes})
        
        # Create price bins
        price_min, price_max = min(prices), max(prices)
        price_bins = np.linspace(price_min, price_max, bins + 1)
        
        # Assign each price to a bin
        df['price_bin'] = pd.cut(df['price'], bins=price_bins, include_lowest=True)
        
        # Calculate volume for each price bin
        volume_profile = df.groupby('price_bin')['volume'].sum().reset_index()
        volume_profile['price_level'] = volume_profile['price_bin'].apply(lambda x: x.mid)
        
        # Find Point of Control (POC) - price level with highest volume
        poc_idx = volume_profile['volume'].idxmax()
        poc_price = volume_profile.loc[poc_idx, 'price_level']
        poc_volume = volume_profile.loc[poc_idx, 'volume']
        
        # Calculate Value Area (70% of total volume)
        total_volume = volume_profile['volume'].sum()
        target_volume = total_volume * 0.7
        
        # Sort by volume to find value area
        sorted_profile = volume_profile.sort_values('volume', ascending=False)
        cumulative_volume = 0
        value_area_prices = []
        
        for _, row in sorted_profile.iterrows():
            cumulative_volume += row['volume']
            value_area_prices.append(row['price_level'])
            if cumulative_volume >= target_volume:
                break
        
        value_area_high = max(value_area_prices)
        value_area_low = min(value_area_prices)
        
        # Identify High Volume Nodes (HVN) and Low Volume Nodes (LVN)
        volume_threshold_high = volume_profile['volume'].quantile(0.8)
        volume_threshold_low = volume_profile['volume'].quantile(0.2)
        
        hvn_levels = volume_profile[volume_profile['volume'] >= volume_threshold_high]['price_level'].tolist()
        lvn_levels = volume_profile[volume_profile['volume'] <= volume_threshold_low]['price_level'].tolist()
        
        return {
            'volume_profile': volume_profile.to_dict('records'),
            'poc_price': poc_price,
            'poc_volume': poc_volume,
            'value_area_high': value_area_high,
            'value_area_low': value_area_low,
            'hvn_levels': hvn_levels,
            'lvn_levels': lvn_levels,
            'total_volume': total_volume
        }
    
    def calculate_support_resistance(self, highs: List[float], lows: List[float], 
                                   lookback: int = 20, min_touches: int = 2) -> Dict:
        """
        Calculate support and resistance levels
        
        Args:
            highs: List of high prices
            lows: List of low prices
            lookback: Lookback period for pivot detection
            min_touches: Minimum touches to confirm level
            
        Returns:
            Dictionary containing support and resistance levels
        """
        def find_pivots(data: List[float], lookback: int, pivot_type: str) -> List[Tuple[int, float]]:
            """Find pivot highs or lows"""
            pivots = []
            
            for i in range(lookback, len(data) - lookback):
                if pivot_type == 'high':
                    if all(data[i] >= data[j] for j in range(i - lookback, i + lookback + 1) if j != i):
                        pivots.append((i, data[i]))
                elif pivot_type == 'low':
                    if all(data[i] <= data[j] for j in range(i - lookback, i + lookback + 1) if j != i):
                        pivots.append((i, data[i]))
            
            return pivots
        
        # Find pivot highs and lows
        pivot_highs = find_pivots(highs, lookback, 'high')
        pivot_lows = find_pivots(lows, lookback, 'low')
        
        # Group similar levels
        def group_levels(pivots: List[Tuple[int, float]], tolerance: float = 0.01) -> List[Dict]:
            """Group similar price levels"""
            if not pivots:
                return []
            
            levels = []
            pivots_sorted = sorted(pivots, key=lambda x: x[1])
            
            current_group = [pivots_sorted[0]]
            
            for i in range(1, len(pivots_sorted)):
                current_price = pivots_sorted[i][1]
                group_avg = sum(p[1] for p in current_group) / len(current_group)
                
                if abs(current_price - group_avg) / group_avg <= tolerance:
                    current_group.append(pivots_sorted[i])
                else:
                    if len(current_group) >= min_touches:
                        avg_price = sum(p[1] for p in current_group) / len(current_group)
                        levels.append({
                            'price': avg_price,
                            'touches': len(current_group),
                            'strength': len(current_group) / len(pivots_sorted)
                        })
                    current_group = [pivots_sorted[i]]
            
            # Don't forget the last group
            if len(current_group) >= min_touches:
                avg_price = sum(p[1] for p in current_group) / len(current_group)
                levels.append({
                    'price': avg_price,
                    'touches': len(current_group),
                    'strength': len(current_group) / len(pivots_sorted)
                })
            
            return levels
        
        resistance_levels = group_levels(pivot_highs)
        support_levels = group_levels(pivot_lows)
        
        return {
            'resistance_levels': resistance_levels,
            'support_levels': support_levels,
            'pivot_highs': pivot_highs,
            'pivot_lows': pivot_lows
        }
    
    def calculate_volatility_metrics(self, prices: List[float], period: int = 20) -> Dict:
        """
        Calculate volatility metrics
        
        Args:
            prices: List of price values
            period: Period for calculations
            
        Returns:
            Dictionary containing volatility metrics
        """
        df = pd.DataFrame({'price': prices})
        
        # Calculate returns
        returns = df['price'].pct_change().dropna()
        
        # Realized volatility (annualized)
        realized_vol = returns.rolling(window=period).std() * np.sqrt(365 * 24)  # For hourly data
        
        # Average True Range (ATR) - simplified version
        high_low = df['price'].rolling(window=2).max() - df['price'].rolling(window=2).min()
        atr = high_low.rolling(window=period).mean()
        
        return {
            'realized_volatility': realized_vol.tolist(),
            'atr': atr.tolist(),
            'current_volatility': realized_vol.iloc[-1] if len(realized_vol) > 0 else None
        }
    
    def detect_chart_patterns(self, highs: List[float], lows: List[float], 
                            closes: List[float], lookback: int = 50) -> Dict:
        """
        Detect basic chart patterns
        
        Args:
            highs: List of high prices
            lows: List of low prices
            closes: List of closing prices
            lookback: Lookback period for pattern detection
            
        Returns:
            Dictionary containing detected patterns
        """
        patterns = {
            'consolidation_range': None,
            'triangle_pattern': None,
            'breakout_potential': None
        }
        
        if len(closes) < lookback:
            return patterns
        
        recent_highs = highs[-lookback:]
        recent_lows = lows[-lookback:]
        recent_closes = closes[-lookback:]
        
        # Detect consolidation range
        high_max = max(recent_highs)
        low_min = min(recent_lows)
        range_size = (high_max - low_min) / low_min
        
        # If range is less than 10%, consider it consolidation
        if range_size < 0.10:
            patterns['consolidation_range'] = {
                'upper_bound': high_max,
                'lower_bound': low_min,
                'range_size_pct': range_size * 100,
                'current_position': (recent_closes[-1] - low_min) / (high_max - low_min)
            }
        
        # Simple triangle pattern detection
        # Check if highs are trending down and lows are trending up
        if len(recent_highs) >= 10:
            high_trend = np.polyfit(range(len(recent_highs)), recent_highs, 1)[0]
            low_trend = np.polyfit(range(len(recent_lows)), recent_lows, 1)[0]
            
            if high_trend < 0 and low_trend > 0:
                patterns['triangle_pattern'] = {
                    'type': 'symmetrical_triangle',
                    'high_trend_slope': high_trend,
                    'low_trend_slope': low_trend,
                    'convergence_point': len(recent_closes) + abs(high_max - low_min) / abs(high_trend - low_trend)
                }
        
        # Breakout potential (price near range boundaries)
        current_price = recent_closes[-1]
        if patterns['consolidation_range']:
            upper_bound = patterns['consolidation_range']['upper_bound']
            lower_bound = patterns['consolidation_range']['lower_bound']
            
            distance_to_upper = (upper_bound - current_price) / current_price
            distance_to_lower = (current_price - lower_bound) / current_price
            
            if distance_to_upper < 0.02:  # Within 2% of upper bound
                patterns['breakout_potential'] = {
                    'direction': 'upward',
                    'distance_to_level': distance_to_upper,
                    'target_level': upper_bound
                }
            elif distance_to_lower < 0.02:  # Within 2% of lower bound
                patterns['breakout_potential'] = {
                    'direction': 'downward',
                    'distance_to_level': distance_to_lower,
                    'target_level': lower_bound
                }
        
        return patterns
    
    def calculate_divergences(self, prices: List[float], rsi_values: List[float]) -> Dict:
        """
        Detect RSI divergences with price
        
        Args:
            prices: List of price values
            rsi_values: List of RSI values
            
        Returns:
            Dictionary containing divergence information
        """
        divergences = {
            'bullish_divergence': False,
            'bearish_divergence': False,
            'divergence_strength': 0
        }
        
        if len(prices) < 20 or len(rsi_values) < 20:
            return divergences
        
        # Look for divergences in the last 20 periods
        recent_prices = prices[-20:]
        recent_rsi = [r for r in rsi_values[-20:] if r is not None]
        
        if len(recent_rsi) < 10:
            return divergences
        
        # Find recent highs and lows
        price_highs = []
        price_lows = []
        rsi_highs = []
        rsi_lows = []
        
        for i in range(2, len(recent_prices) - 2):
            # Price highs
            if (recent_prices[i] > recent_prices[i-1] and recent_prices[i] > recent_prices[i-2] and
                recent_prices[i] > recent_prices[i+1] and recent_prices[i] > recent_prices[i+2]):
                price_highs.append((i, recent_prices[i]))
                if i < len(recent_rsi):
                    rsi_highs.append((i, recent_rsi[i]))
            
            # Price lows
            if (recent_prices[i] < recent_prices[i-1] and recent_prices[i] < recent_prices[i-2] and
                recent_prices[i] < recent_prices[i+1] and recent_prices[i] < recent_prices[i+2]):
                price_lows.append((i, recent_prices[i]))
                if i < len(recent_rsi):
                    rsi_lows.append((i, recent_rsi[i]))
        
        # Check for bullish divergence (price makes lower lows, RSI makes higher lows)
        if len(price_lows) >= 2 and len(rsi_lows) >= 2:
            latest_price_low = price_lows[-1][1]
            previous_price_low = price_lows[-2][1]
            latest_rsi_low = rsi_lows[-1][1]
            previous_rsi_low = rsi_lows[-2][1]
            
            if latest_price_low < previous_price_low and latest_rsi_low > previous_rsi_low:
                divergences['bullish_divergence'] = True
                divergences['divergence_strength'] = abs(latest_rsi_low - previous_rsi_low)
        
        # Check for bearish divergence (price makes higher highs, RSI makes lower highs)
        if len(price_highs) >= 2 and len(rsi_highs) >= 2:
            latest_price_high = price_highs[-1][1]
            previous_price_high = price_highs[-2][1]
            latest_rsi_high = rsi_highs[-1][1]
            previous_rsi_high = rsi_highs[-2][1]
            
            if latest_price_high > previous_price_high and latest_rsi_high < previous_rsi_high:
                divergences['bearish_divergence'] = True
                divergences['divergence_strength'] = abs(latest_rsi_high - previous_rsi_high)
        
        return divergences
    
    def analyze_market_structure(self, highs: List[float], lows: List[float], 
                               closes: List[float]) -> Dict:
        """
        Comprehensive market structure analysis
        
        Args:
            highs: List of high prices
            lows: List of low prices
            closes: List of closing prices
            
        Returns:
            Dictionary containing market structure analysis
        """
        # Calculate all indicators
        ma_data = self.calculate_moving_averages(closes)
        rsi_values = self.calculate_rsi(closes)
        bb_data = self.calculate_bollinger_bands(closes)
        sr_levels = self.calculate_support_resistance(highs, lows)
        volatility = self.calculate_volatility_metrics(closes)
        patterns = self.detect_chart_patterns(highs, lows, closes)
        divergences = self.calculate_divergences(closes, rsi_values)
        
        # Current market state
        current_price = closes[-1]
        current_rsi = rsi_values[-1] if rsi_values[-1] is not None else 50
        
        # Trend analysis
        trend_analysis = {
            'short_term_trend': 'neutral',
            'medium_term_trend': 'neutral',
            'long_term_trend': 'neutral'
        }
        
        if 'EMA_21' in ma_data and len(ma_data['EMA_21']) > 0:
            ema_21 = ma_data['EMA_21'][-1]
            if current_price > ema_21:
                trend_analysis['short_term_trend'] = 'bullish'
            elif current_price < ema_21:
                trend_analysis['short_term_trend'] = 'bearish'
        
        if 'EMA_50' in ma_data and len(ma_data['EMA_50']) > 0:
            ema_50 = ma_data['EMA_50'][-1]
            if current_price > ema_50:
                trend_analysis['medium_term_trend'] = 'bullish'
            elif current_price < ema_50:
                trend_analysis['medium_term_trend'] = 'bearish'
        
        if 'EMA_200' in ma_data and len(ma_data['EMA_200']) > 0:
            ema_200 = ma_data['EMA_200'][-1]
            if current_price > ema_200:
                trend_analysis['long_term_trend'] = 'bullish'
            elif current_price < ema_200:
                trend_analysis['long_term_trend'] = 'bearish'
        
        return {
            'current_price': current_price,
            'current_rsi': current_rsi,
            'moving_averages': ma_data,
            'bollinger_bands': bb_data,
            'support_resistance': sr_levels,
            'volatility_metrics': volatility,
            'chart_patterns': patterns,
            'divergences': divergences,
            'trend_analysis': trend_analysis,
            'analysis_timestamp': datetime.now().isoformat()
        }

def main():
    """Test the technical analysis module"""
    print("Testing Technical Analysis Module")
    print("=" * 40)
    
    # Generate sample data for testing
    np.random.seed(42)
    base_price = 50000
    returns = np.random.normal(0, 0.02, 100)
    prices = [base_price]
    
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    # Generate highs and lows
    highs = [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices]
    lows = [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices]
    volumes = [np.random.uniform(1000, 10000) for _ in prices]
    
    analyzer = TechnicalAnalyzer()
    
    # Test individual functions
    print("\\n1. Testing Moving Averages...")
    ma_data = analyzer.calculate_moving_averages(prices)
    print(f"✓ Calculated MAs for periods: {list(ma_data.keys())}")
    
    print("\\n2. Testing RSI...")
    rsi_values = analyzer.calculate_rsi(prices)
    current_rsi = [r for r in rsi_values if r is not None][-1]
    print(f"✓ Current RSI: {current_rsi:.2f}")
    
    print("\\n3. Testing Volume Profile...")
    vp_data = analyzer.calculate_volume_profile(prices, volumes)
    print(f"✓ POC Price: ${vp_data['poc_price']:.2f}")
    print(f"✓ Value Area: ${vp_data['value_area_low']:.2f} - ${vp_data['value_area_high']:.2f}")
    
    print("\\n4. Testing Support/Resistance...")
    sr_data = analyzer.calculate_support_resistance(highs, lows)
    print(f"✓ Found {len(sr_data['resistance_levels'])} resistance levels")
    print(f"✓ Found {len(sr_data['support_levels'])} support levels")
    
    print("\\n5. Testing Comprehensive Analysis...")
    full_analysis = analyzer.analyze_market_structure(highs, lows, prices)
    print(f"✓ Current Price: ${full_analysis['current_price']:.2f}")
    print(f"✓ Short-term Trend: {full_analysis['trend_analysis']['short_term_trend']}")
    print(f"✓ RSI: {full_analysis['current_rsi']:.2f}")
    
    print("\\nTechnical Analysis Module testing completed!")

if __name__ == "__main__":
    main()

