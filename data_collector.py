#!/usr/bin/env python3
"""
Cryptocurrency Trading Data Collector
Comprehensive data collection system for leveraged perpetual contracts analysis
"""

import sys
import os
import json
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

# Add Manus API Hub to path
sys.path.append('/opt/.manus/.sandbox-runtime')
from data_api import ApiClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CryptoDataCollector:
    """Main data collector for cryptocurrency trading analysis"""
    
    def __init__(self):
        self.api_client = ApiClient()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'CryptoTradingAnalyzer/1.0'
        })
        
    def get_traditional_finance_data(self, symbol: str = 'SPY', period: str = '1mo') -> Dict:
        """
        Get traditional finance data using Yahoo Finance API from Manus Hub
        
        Args:
            symbol: Stock symbol (default: SPY for S&P 500)
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            
        Returns:
            Dictionary containing stock data
        """
        try:
            logger.info(f"Fetching traditional finance data for {symbol}")
            
            response = self.api_client.call_api('YahooFinance/get_stock_chart', query={
                'symbol': symbol,
                'region': 'US',
                'interval': '1d',
                'range': period,
                'includeAdjustedClose': True,
                'events': 'div,split'
            })
            
            if response and 'chart' in response and 'result' in response['chart']:
                result = response['chart']['result'][0]
                return {
                    'meta': result['meta'],
                    'timestamps': result['timestamp'],
                    'quotes': result['indicators']['quote'][0],
                    'symbol': symbol,
                    'success': True
                }
            else:
                logger.error(f"No data found for {symbol}")
                return {'success': False, 'error': 'No data found'}
                
        except Exception as e:
            logger.error(f"Error fetching traditional finance data: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def get_binance_futures_data(self, symbol: str = 'BTCUSDT', interval: str = '1h', limit: int = 500) -> Dict:
        """
        Get Binance futures data using public API
        
        Args:
            symbol: Trading pair symbol (e.g., BTCUSDT)
            interval: Kline interval (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
            limit: Number of data points (max 1500)
            
        Returns:
            Dictionary containing futures data
        """
        try:
            logger.info(f"Fetching Binance futures data for {symbol}")
            
            base_url = "https://fapi.binance.com"
            
            # Get kline data
            kline_url = f"{base_url}/fapi/v1/klines"
            kline_params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            
            kline_response = self.session.get(kline_url, params=kline_params)
            kline_response.raise_for_status()
            kline_data = kline_response.json()
            
            # Get 24hr ticker statistics
            ticker_url = f"{base_url}/fapi/v1/ticker/24hr"
            ticker_params = {'symbol': symbol}
            
            ticker_response = self.session.get(ticker_url, params=ticker_params)
            ticker_response.raise_for_status()
            ticker_data = ticker_response.json()
            
            # Get funding rate
            funding_url = f"{base_url}/fapi/v1/fundingRate"
            funding_params = {'symbol': symbol, 'limit': 10}
            
            funding_response = self.session.get(funding_url, params=funding_params)
            funding_response.raise_for_status()
            funding_data = funding_response.json()
            
            # Get open interest
            oi_url = f"{base_url}/fapi/v1/openInterest"
            oi_params = {'symbol': symbol}
            
            oi_response = self.session.get(oi_url, params=oi_params)
            oi_response.raise_for_status()
            oi_data = oi_response.json()
            
            return {
                'symbol': symbol,
                'kline_data': kline_data,
                'ticker_data': ticker_data,
                'funding_data': funding_data,
                'open_interest': oi_data,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error fetching Binance futures data: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def get_coinmarketcap_dominance(self) -> Dict:
        """
        Get Bitcoin dominance data (using public endpoint)
        Note: For production use, consider using CoinMarketCap API with proper key
        
        Returns:
            Dictionary containing dominance data
        """
        try:
            logger.info("Fetching Bitcoin dominance data")
            
            # Using a public endpoint that provides dominance data
            # In production, use proper CoinMarketCap API
            url = "https://api.coinmarketcap.com/data-api/v3/global-metrics/quotes/latest"
            
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()
            
            if 'data' in data:
                btc_dominance = data['data'].get('btcDominance', 0)
                eth_dominance = data['data'].get('ethDominance', 0)
                
                return {
                    'btc_dominance': btc_dominance,
                    'eth_dominance': eth_dominance,
                    'timestamp': datetime.now().isoformat(),
                    'success': True
                }
            else:
                return {'success': False, 'error': 'No dominance data found'}
                
        except Exception as e:
            logger.error(f"Error fetching dominance data: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def get_coinglass_liquidation_data(self, symbol: str = 'BTC') -> Dict:
        """
        Get liquidation data from CoinGlass (using public endpoints where available)
        Note: For production use, consider CoinGlass API with proper authentication
        
        Args:
            symbol: Cryptocurrency symbol (BTC, ETH, etc.)
            
        Returns:
            Dictionary containing liquidation data
        """
        try:
            logger.info(f"Fetching liquidation data for {symbol}")
            
            # This is a placeholder for CoinGlass API integration
            # In production, implement proper CoinGlass API calls
            
            # For now, return mock structure to demonstrate data format
            return {
                'symbol': symbol,
                'liquidation_data': {
                    'total_liquidations_24h': 0,
                    'long_liquidations': 0,
                    'short_liquidations': 0,
                    'liquidation_levels': []
                },
                'timestamp': datetime.now().isoformat(),
                'success': True,
                'note': 'Mock data - implement CoinGlass API integration'
            }
            
        except Exception as e:
            logger.error(f"Error fetching liquidation data: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def collect_comprehensive_data(self, symbols: List[str] = None) -> Dict:
        """
        Collect comprehensive data for all required analysis
        
        Args:
            symbols: List of cryptocurrency symbols to analyze
            
        Returns:
            Dictionary containing all collected data
        """
        if symbols is None:
            symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
        
        logger.info("Starting comprehensive data collection")
        
        collected_data = {
            'timestamp': datetime.now().isoformat(),
            'traditional_finance': {},
            'crypto_data': {},
            'dominance_data': {},
            'liquidation_data': {}
        }
        
        # Collect traditional finance data
        try:
            spy_data = self.get_traditional_finance_data('SPY', '1mo')
            collected_data['traditional_finance']['SPY'] = spy_data
            
            # Also get VIX for volatility context
            vix_data = self.get_traditional_finance_data('^VIX', '1mo')
            collected_data['traditional_finance']['VIX'] = vix_data
            
        except Exception as e:
            logger.error(f"Error collecting traditional finance data: {str(e)}")
        
        # Collect cryptocurrency data
        for symbol in symbols:
            try:
                crypto_data = self.get_binance_futures_data(symbol, '1h', 168)  # 1 week of hourly data
                collected_data['crypto_data'][symbol] = crypto_data
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error collecting data for {symbol}: {str(e)}")
        
        # Collect dominance data
        try:
            dominance_data = self.get_coinmarketcap_dominance()
            collected_data['dominance_data'] = dominance_data
            
        except Exception as e:
            logger.error(f"Error collecting dominance data: {str(e)}")
        
        # Collect liquidation data
        for symbol in ['BTC', 'ETH', 'SOL']:
            try:
                liq_data = self.get_coinglass_liquidation_data(symbol)
                collected_data['liquidation_data'][symbol] = liq_data
                
            except Exception as e:
                logger.error(f"Error collecting liquidation data for {symbol}: {str(e)}")
        
        logger.info("Comprehensive data collection completed")
        return collected_data
    
    def save_data_to_file(self, data: Dict, filename: str = None) -> str:
        """
        Save collected data to JSON file
        
        Args:
            data: Data dictionary to save
            filename: Optional filename (auto-generated if not provided)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"crypto_data_{timestamp}.json"
        
        filepath = os.path.join('/home/ubuntu', filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Data saved to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            return None

def main():
    """Main function for testing data collection"""
    collector = CryptoDataCollector()
    
    # Test data collection
    print("Testing Cryptocurrency Data Collector")
    print("=" * 50)
    
    # Test traditional finance data
    print("\\n1. Testing Traditional Finance Data (SPY)...")
    spy_data = collector.get_traditional_finance_data('SPY', '5d')
    if spy_data['success']:
        print(f"✓ Successfully fetched SPY data")
        print(f"  Current Price: ${spy_data['meta']['regularMarketPrice']:.2f}")
        print(f"  Data points: {len(spy_data['timestamps'])}")
    else:
        print(f"✗ Failed to fetch SPY data: {spy_data['error']}")
    
    # Test crypto data
    print("\\n2. Testing Cryptocurrency Data (BTCUSDT)...")
    btc_data = collector.get_binance_futures_data('BTCUSDT', '1h', 24)
    if btc_data['success']:
        print(f"✓ Successfully fetched BTCUSDT data")
        print(f"  24h Price Change: {btc_data['ticker_data']['priceChangePercent']}%")
        print(f"  Open Interest: {btc_data['open_interest']['openInterest']}")
        print(f"  Kline data points: {len(btc_data['kline_data'])}")
    else:
        print(f"✗ Failed to fetch BTCUSDT data: {btc_data['error']}")
    
    # Test dominance data
    print("\\n3. Testing Bitcoin Dominance Data...")
    dom_data = collector.get_coinmarketcap_dominance()
    if dom_data['success']:
        print(f"✓ Successfully fetched dominance data")
        print(f"  BTC Dominance: {dom_data['btc_dominance']:.2f}%")
        print(f"  ETH Dominance: {dom_data['eth_dominance']:.2f}%")
    else:
        print(f"✗ Failed to fetch dominance data: {dom_data['error']}")
    
    # Test comprehensive collection
    print("\\n4. Testing Comprehensive Data Collection...")
    comprehensive_data = collector.collect_comprehensive_data(['BTCUSDT', 'ETHUSDT'])
    
    # Save data
    saved_file = collector.save_data_to_file(comprehensive_data)
    if saved_file:
        print(f"✓ Comprehensive data saved to: {saved_file}")
    
    print("\\nData collection testing completed!")

if __name__ == "__main__":
    main()

