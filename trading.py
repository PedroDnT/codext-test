from flask import Blueprint, jsonify, request
from flask_cors import cross_origin
import json
import logging
from datetime import datetime
import threading
import time

# Import our trading analysis modules
from src.market_analyzer import MarketAnalyzer
from src.data_collector import CryptoDataCollector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

trading_bp = Blueprint('trading', __name__)

# Global variables for caching
analysis_cache = {}
cache_timestamp = None
cache_duration = 300  # 5 minutes cache
analysis_lock = threading.Lock()

# Initialize analyzers
market_analyzer = MarketAnalyzer()
data_collector = CryptoDataCollector()

@trading_bp.route('/market-overview', methods=['GET'])
@cross_origin()
def get_market_overview():
    """Get overall market sentiment and overview"""
    try:
        # Get cached data or fetch new
        analysis_data = get_cached_analysis()
        
        if not analysis_data:
            return jsonify({'error': 'No analysis data available'}), 500
        
        market_sentiment = analysis_data.get('market_sentiment', {})
        analysis_summary = analysis_data.get('analysis_summary', {})
        
        overview = {
            'timestamp': analysis_data.get('analysis_timestamp'),
            'overall_sentiment': market_sentiment.get('overall_sentiment', 'neutral'),
            'sentiment_score': market_sentiment.get('sentiment_score', 0),
            'market_bias': market_sentiment.get('market_bias', 'neutral'),
            'assets_analyzed': analysis_summary.get('total_assets_analyzed', 0),
            'valid_analyses': analysis_summary.get('valid_analyses', 0),
            'trade_setups_generated': analysis_summary.get('trade_setups_generated', 0),
            'sentiment_components': market_sentiment.get('components', {})
        }
        
        return jsonify(overview)
        
    except Exception as e:
        logger.error(f"Error in market overview: {str(e)}")
        return jsonify({'error': str(e)}), 500

@trading_bp.route('/trade-opportunities', methods=['GET'])
@cross_origin()
def get_trade_opportunities():
    """Get current trading opportunities"""
    try:
        analysis_data = get_cached_analysis()
        
        if not analysis_data:
            return jsonify({'error': 'No analysis data available'}), 500
        
        trade_setups = analysis_data.get('top_trade_setups', [])
        
        # Format trade setups for frontend
        formatted_setups = []
        for setup in trade_setups:
            formatted_setup = {
                'id': f"{setup['asset']}_{setup['direction']}_{setup['strategy'].replace(' ', '_')}",
                'asset': setup['asset'],
                'direction': setup['direction'],
                'strategy': setup['strategy'],
                'setup_strength': setup['setup_strength'],
                'current_price': setup['current_price'],
                'entry_zone': setup.get('entry_zone', {}),
                'take_profit_levels': setup.get('take_profit_levels', []),
                'stop_loss': setup.get('stop_loss'),
                'leverage': setup.get('leverage', 20),
                'market_context': setup.get('market_context', 'neutral'),
                'strategy_rationale': setup.get('strategy_rationale', ''),
                'risk_metrics': setup.get('risk_metrics', {})
            }
            formatted_setups.append(formatted_setup)
        
        return jsonify({
            'opportunities': formatted_setups,
            'count': len(formatted_setups),
            'timestamp': analysis_data.get('analysis_timestamp')
        })
        
    except Exception as e:
        logger.error(f"Error in trade opportunities: {str(e)}")
        return jsonify({'error': str(e)}), 500

@trading_bp.route('/asset-analysis/<symbol>', methods=['GET'])
@cross_origin()
def get_asset_analysis(symbol):
    """Get detailed analysis for a specific asset"""
    try:
        analysis_data = get_cached_analysis()
        
        if not analysis_data:
            return jsonify({'error': 'No analysis data available'}), 500
        
        asset_analyses = analysis_data.get('asset_analyses', {})
        symbol_key = f"{symbol}USDT" if not symbol.endswith('USDT') else symbol
        
        if symbol_key not in asset_analyses:
            return jsonify({'error': f'No analysis available for {symbol}'}), 404
        
        asset_data = asset_analyses[symbol_key]
        
        if 'error' in asset_data:
            return jsonify({'error': asset_data['error']}), 500
        
        # Format asset analysis for frontend
        formatted_analysis = {
            'symbol': asset_data.get('symbol', symbol_key),
            'current_price': asset_data.get('current_price'),
            'technical_analysis': asset_data.get('technical_analysis', {}),
            'volume_profile': asset_data.get('volume_profile', {}),
            'sentiment_data': asset_data.get('sentiment_data', {}),
            'strategy_analysis': asset_data.get('strategy_analysis', {}),
            'risk_metrics': asset_data.get('risk_metrics', {}),
            'market_data': asset_data.get('market_data', {}),
            'analysis_timestamp': asset_data.get('analysis_timestamp')
        }
        
        return jsonify(formatted_analysis)
        
    except Exception as e:
        logger.error(f"Error in asset analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

@trading_bp.route('/market-data', methods=['GET'])
@cross_origin()
def get_market_data():
    """Get raw market data for charts and analysis"""
    try:
        symbols = request.args.get('symbols', 'BTCUSDT,ETHUSDT,SOLUSDT').split(',')
        
        # Collect fresh market data
        collected_data = data_collector.collect_comprehensive_data(symbols)
        
        # Format for frontend
        formatted_data = {
            'timestamp': collected_data.get('timestamp'),
            'traditional_finance': collected_data.get('traditional_finance', {}),
            'crypto_data': {},
            'dominance_data': collected_data.get('dominance_data', {}),
            'liquidation_data': collected_data.get('liquidation_data', {})
        }
        
        # Format crypto data
        for symbol, data in collected_data.get('crypto_data', {}).items():
            if data.get('success'):
                formatted_data['crypto_data'][symbol] = {
                    'ticker_data': data.get('ticker_data', {}),
                    'funding_data': data.get('funding_data', []),
                    'open_interest': data.get('open_interest', {}),
                    'success': True
                }
            else:
                formatted_data['crypto_data'][symbol] = {
                    'success': False,
                    'error': data.get('error', 'Unknown error')
                }
        
        return jsonify(formatted_data)
        
    except Exception as e:
        logger.error(f"Error in market data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@trading_bp.route('/refresh-analysis', methods=['POST'])
@cross_origin()
def refresh_analysis():
    """Force refresh of analysis data"""
    try:
        symbols = request.json.get('symbols', ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']) if request.json else ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
        
        # Clear cache and run fresh analysis
        global analysis_cache, cache_timestamp
        with analysis_lock:
            analysis_cache = {}
            cache_timestamp = None
        
        # Run analysis in background thread to avoid timeout
        def run_analysis():
            try:
                analysis_result = market_analyzer.run_comprehensive_analysis(symbols)
                with analysis_lock:
                    global analysis_cache, cache_timestamp
                    analysis_cache = analysis_result
                    cache_timestamp = datetime.now()
                logger.info("Analysis refresh completed")
            except Exception as e:
                logger.error(f"Background analysis failed: {str(e)}")
        
        analysis_thread = threading.Thread(target=run_analysis)
        analysis_thread.daemon = True
        analysis_thread.start()
        
        return jsonify({
            'message': 'Analysis refresh initiated',
            'status': 'processing',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in refresh analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

@trading_bp.route('/system-status', methods=['GET'])
@cross_origin()
def get_system_status():
    """Get system status and health check"""
    try:
        status = {
            'status': 'operational',
            'timestamp': datetime.now().isoformat(),
            'cache_status': {
                'has_cache': bool(analysis_cache),
                'cache_age_seconds': (datetime.now() - cache_timestamp).total_seconds() if cache_timestamp else None,
                'cache_valid': is_cache_valid()
            },
            'components': {
                'data_collector': 'operational',
                'market_analyzer': 'operational',
                'technical_analysis': 'operational',
                'sentiment_analysis': 'operational',
                'trading_strategies': 'operational'
            }
        }
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"Error in system status: {str(e)}")
        return jsonify({'error': str(e)}), 500

def get_cached_analysis():
    """Get cached analysis data or fetch new if cache is invalid"""
    global analysis_cache, cache_timestamp
    
    with analysis_lock:
        if is_cache_valid():
            return analysis_cache
        
        # Cache is invalid, fetch new data
        try:
            logger.info("Cache invalid, fetching fresh analysis data")
            symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
            analysis_result = market_analyzer.run_comprehensive_analysis(symbols)
            
            analysis_cache = analysis_result
            cache_timestamp = datetime.now()
            
            return analysis_cache
            
        except Exception as e:
            logger.error(f"Failed to fetch fresh analysis: {str(e)}")
            return analysis_cache if analysis_cache else None

def is_cache_valid():
    """Check if current cache is still valid"""
    if not analysis_cache or not cache_timestamp:
        return False
    
    age_seconds = (datetime.now() - cache_timestamp).total_seconds()
    return age_seconds < cache_duration

# Background task to periodically refresh analysis
def background_refresh():
    """Background task to refresh analysis data periodically"""
    while True:
        try:
            time.sleep(cache_duration)  # Wait for cache duration
            
            if not is_cache_valid():
                logger.info("Background refresh triggered")
                get_cached_analysis()
                
        except Exception as e:
            logger.error(f"Background refresh error: {str(e)}")
            time.sleep(60)  # Wait 1 minute before retry

# Start background refresh thread
refresh_thread = threading.Thread(target=background_refresh)
refresh_thread.daemon = True
refresh_thread.start()

