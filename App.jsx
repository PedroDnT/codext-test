import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { 
  TrendingUp, 
  TrendingDown, 
  RefreshCw, 
  AlertTriangle, 
  Activity,
  DollarSign,
  BarChart3,
  Target,
  Shield,
  Clock
} from 'lucide-react'
import './App.css'

const API_BASE = 'http://localhost:5000/api/trading'

function App() {
  const [marketOverview, setMarketOverview] = useState(null)
  const [tradeOpportunities, setTradeOpportunities] = useState([])
  const [systemStatus, setSystemStatus] = useState(null)
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [lastUpdate, setLastUpdate] = useState(null)

  // Fetch data from API
  const fetchData = async () => {
    try {
      setLoading(true)
      
      // Fetch market overview
      const overviewResponse = await fetch(`${API_BASE}/market-overview`)
      if (overviewResponse.ok) {
        const overviewData = await overviewResponse.json()
        setMarketOverview(overviewData)
      }

      // Fetch trade opportunities
      const opportunitiesResponse = await fetch(`${API_BASE}/trade-opportunities`)
      if (opportunitiesResponse.ok) {
        const opportunitiesData = await opportunitiesResponse.json()
        setTradeOpportunities(opportunitiesData.opportunities || [])
      }

      // Fetch system status
      const statusResponse = await fetch(`${API_BASE}/system-status`)
      if (statusResponse.ok) {
        const statusData = await statusResponse.json()
        setSystemStatus(statusData)
      }

      setLastUpdate(new Date())
    } catch (error) {
      console.error('Error fetching data:', error)
    } finally {
      setLoading(false)
    }
  }

  // Refresh analysis
  const refreshAnalysis = async () => {
    try {
      setRefreshing(true)
      const response = await fetch(`${API_BASE}/refresh-analysis`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          symbols: ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
        })
      })
      
      if (response.ok) {
        // Wait a moment then fetch fresh data
        setTimeout(() => {
          fetchData()
        }, 2000)
      }
    } catch (error) {
      console.error('Error refreshing analysis:', error)
    } finally {
      setRefreshing(false)
    }
  }

  // Auto-refresh every 5 minutes
  useEffect(() => {
    fetchData()
    const interval = setInterval(fetchData, 300000) // 5 minutes
    return () => clearInterval(interval)
  }, [])

  const getSentimentColor = (sentiment) => {
    switch (sentiment?.toLowerCase()) {
      case 'bullish': return 'text-green-600'
      case 'bearish': return 'text-red-600'
      default: return 'text-yellow-600'
    }
  }

  const getSentimentIcon = (sentiment) => {
    switch (sentiment?.toLowerCase()) {
      case 'bullish': return <TrendingUp className="h-4 w-4" />
      case 'bearish': return <TrendingDown className="h-4 w-4" />
      default: return <Activity className="h-4 w-4" />
    }
  }

  const getDirectionColor = (direction) => {
    return direction === 'LONG' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
  }

  const formatPrice = (price) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 6
    }).format(price)
  }

  if (loading && !marketOverview) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <RefreshCw className="h-8 w-8 animate-spin mx-auto mb-4" />
          <p className="text-gray-600">Loading trading dashboard...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">
                Crypto Trading Dashboard
              </h1>
              <p className="text-sm text-gray-600">
                Live opportunities for leveraged perpetual contracts
              </p>
            </div>
            <div className="flex items-center space-x-4">
              {lastUpdate && (
                <div className="text-sm text-gray-500 flex items-center">
                  <Clock className="h-4 w-4 mr-1" />
                  Last update: {lastUpdate.toLocaleTimeString()}
                </div>
              )}
              <Button 
                onClick={refreshAnalysis} 
                disabled={refreshing}
                variant="outline"
                size="sm"
              >
                <RefreshCw className={`h-4 w-4 mr-2 ${refreshing ? 'animate-spin' : ''}`} />
                {refreshing ? 'Refreshing...' : 'Refresh'}
              </Button>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Market Overview Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Market Sentiment</CardTitle>
              {getSentimentIcon(marketOverview?.overall_sentiment)}
            </CardHeader>
            <CardContent>
              <div className={`text-2xl font-bold ${getSentimentColor(marketOverview?.overall_sentiment)}`}>
                {marketOverview?.overall_sentiment || 'Unknown'}
              </div>
              <p className="text-xs text-muted-foreground">
                Score: {marketOverview?.sentiment_score?.toFixed(1) || 'N/A'}/100
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Active Opportunities</CardTitle>
              <Target className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {tradeOpportunities.length}
              </div>
              <p className="text-xs text-muted-foreground">
                High-probability setups
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Assets Analyzed</CardTitle>
              <BarChart3 className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {marketOverview?.assets_analyzed || 0}
              </div>
              <p className="text-xs text-muted-foreground">
                Valid: {marketOverview?.valid_analyses || 0}
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">System Status</CardTitle>
              <Shield className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-green-600">
                {systemStatus?.status === 'operational' ? 'Online' : 'Offline'}
              </div>
              <p className="text-xs text-muted-foreground">
                All systems operational
              </p>
            </CardContent>
          </Card>
        </div>

        {/* Main Content Tabs */}
        <Tabs defaultValue="opportunities" className="space-y-6">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="opportunities">Trading Opportunities</TabsTrigger>
            <TabsTrigger value="market">Market Analysis</TabsTrigger>
            <TabsTrigger value="risk">Risk Assessment</TabsTrigger>
          </TabsList>

          {/* Trading Opportunities Tab */}
          <TabsContent value="opportunities" className="space-y-6">
            {tradeOpportunities.length === 0 ? (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center">
                    <AlertTriangle className="h-5 w-5 mr-2 text-yellow-500" />
                    No High-Conviction Setups Available
                  </CardTitle>
                  <CardDescription>
                    Currently, no trading opportunities meet our stringent criteria for 20x leveraged positions.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <Alert>
                    <AlertTriangle className="h-4 w-4" />
                    <AlertDescription>
                      This conservative approach prioritizes capital preservation over forced trade generation. 
                      Reasons may include volatile market conditions, lack of clear technical setups, 
                      or data limitations affecting analysis quality.
                    </AlertDescription>
                  </Alert>
                  <div className="mt-4">
                    <p className="text-sm text-gray-600 mb-2">Recommendations:</p>
                    <ul className="text-sm text-gray-600 space-y-1">
                      <li>• Wait for clearer market conditions</li>
                      <li>• Monitor for improved technical setups</li>
                      <li>• Consider lower leverage positions</li>
                      <li>• Review risk management parameters</li>
                    </ul>
                  </div>
                </CardContent>
              </Card>
            ) : (
              <div className="grid gap-6">
                {tradeOpportunities.map((opportunity, index) => (
                  <Card key={opportunity.id} className="border-l-4 border-l-blue-500">
                    <CardHeader>
                      <div className="flex justify-between items-start">
                        <div>
                          <CardTitle className="flex items-center space-x-2">
                            <span>{opportunity.asset}/USDT</span>
                            <Badge className={getDirectionColor(opportunity.direction)}>
                              {opportunity.direction}
                            </Badge>
                            <Badge variant="outline">
                              {opportunity.leverage}x
                            </Badge>
                          </CardTitle>
                          <CardDescription>
                            {opportunity.strategy} • Strength: {opportunity.setup_strength.toFixed(1)}/100
                          </CardDescription>
                        </div>
                        <div className="text-right">
                          <div className="text-lg font-semibold">
                            {formatPrice(opportunity.current_price)}
                          </div>
                          <div className="text-sm text-gray-500">Current Price</div>
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        {/* Entry Zone */}
                        <div>
                          <h4 className="font-medium text-sm mb-2">Entry Zone</h4>
                          {opportunity.entry_zone?.lower && opportunity.entry_zone?.upper ? (
                            <div className="text-sm">
                              <div>Lower: {formatPrice(opportunity.entry_zone.lower)}</div>
                              <div>Upper: {formatPrice(opportunity.entry_zone.upper)}</div>
                            </div>
                          ) : (
                            <div className="text-sm text-gray-500">Market entry</div>
                          )}
                        </div>

                        {/* Take Profit */}
                        <div>
                          <h4 className="font-medium text-sm mb-2">Take Profit</h4>
                          <div className="text-sm space-y-1">
                            {opportunity.take_profit_levels?.slice(0, 2).map((tp, i) => (
                              tp && (
                                <div key={i}>
                                  TP{i + 1}: {formatPrice(tp)}
                                </div>
                              )
                            ))}
                            {!opportunity.take_profit_levels?.length && (
                              <div className="text-gray-500">TBD</div>
                            )}
                          </div>
                        </div>

                        {/* Stop Loss */}
                        <div>
                          <h4 className="font-medium text-sm mb-2">Stop Loss</h4>
                          <div className="text-sm">
                            {opportunity.stop_loss ? (
                              formatPrice(opportunity.stop_loss)
                            ) : (
                              <span className="text-gray-500">TBD</span>
                            )}
                          </div>
                        </div>
                      </div>

                      {/* Strategy Rationale */}
                      <div className="mt-4 p-3 bg-gray-50 rounded-lg">
                        <h4 className="font-medium text-sm mb-1">Strategy Rationale</h4>
                        <p className="text-sm text-gray-700">
                          {opportunity.strategy_rationale || 'Technical setup identified based on market structure analysis.'}
                        </p>
                      </div>

                      {/* Risk Metrics */}
                      {opportunity.risk_metrics?.liquidation_prices && (
                        <div className="mt-4 p-3 bg-red-50 rounded-lg border border-red-200">
                          <h4 className="font-medium text-sm mb-1 text-red-800">Risk Warning</h4>
                          <div className="text-sm text-red-700">
                            <div>Liquidation Price: {formatPrice(
                              opportunity.risk_metrics.liquidation_prices[
                                opportunity.direction === 'LONG' ? 'long_position' : 'short_position'
                              ]
                            )}</div>
                            <div className="mt-1 text-xs">
                              ⚠️ 20x leverage involves extreme risk. 5% adverse movement = 100% loss
                            </div>
                          </div>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
          </TabsContent>

          {/* Market Analysis Tab */}
          <TabsContent value="market" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Sentiment Components */}
              <Card>
                <CardHeader>
                  <CardTitle>Sentiment Analysis</CardTitle>
                  <CardDescription>
                    Multi-factor sentiment breakdown
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {marketOverview?.sentiment_components ? (
                    <div className="space-y-4">
                      {Object.entries(marketOverview.sentiment_components).map(([key, component]) => (
                        <div key={key} className="flex justify-between items-center">
                          <span className="text-sm font-medium capitalize">
                            {key.replace('_', ' ')}
                          </span>
                          <div className="text-right">
                            <div className="text-sm font-semibold">
                              {component.score?.toFixed(1) || 'N/A'}
                            </div>
                            <div className="text-xs text-gray-500">
                              {component.interpretation?.replace('_', ' ') || 'Unknown'}
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-sm text-gray-500">No sentiment data available</p>
                  )}
                </CardContent>
              </Card>

              {/* Market Bias */}
              <Card>
                <CardHeader>
                  <CardTitle>Market Bias</CardTitle>
                  <CardDescription>
                    Overall market direction and risk environment
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="text-center">
                    <div className={`text-3xl font-bold mb-2 ${getSentimentColor(marketOverview?.market_bias)}`}>
                      {marketOverview?.market_bias?.replace('_', ' ').toUpperCase() || 'UNKNOWN'}
                    </div>
                    <p className="text-sm text-gray-600">
                      Current market environment classification
                    </p>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Risk Assessment Tab */}
          <TabsContent value="risk" className="space-y-6">
            <Alert className="border-red-200 bg-red-50">
              <AlertTriangle className="h-4 w-4 text-red-600" />
              <AlertDescription className="text-red-800">
                <strong>EXTREME RISK WARNING:</strong> All recommended trades utilize 20x leverage, 
                which significantly amplifies both potential profits and losses. A 5% adverse price 
                movement will result in 100% loss of position margin.
              </AlertDescription>
            </Alert>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Risk Management Guidelines</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3 text-sm">
                    <div>
                      <strong>Position Sizing:</strong>
                      <ul className="mt-1 ml-4 space-y-1 text-gray-600">
                        <li>• Never risk more than 1-2% of total capital per trade</li>
                        <li>• Use proper position sizing calculations</li>
                        <li>• Consider portfolio correlation</li>
                      </ul>
                    </div>
                    <div>
                      <strong>Monitoring:</strong>
                      <ul className="mt-1 ml-4 space-y-1 text-gray-600">
                        <li>• Monitor positions actively during volatile periods</li>
                        <li>• Set up price alerts near liquidation levels</li>
                        <li>• Have contingency plans for adverse scenarios</li>
                      </ul>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>System Limitations</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3 text-sm">
                    <div>
                      <strong>Analysis Limitations:</strong>
                      <ul className="mt-1 ml-4 space-y-1 text-gray-600">
                        <li>• Based on historical patterns and current data</li>
                        <li>• Market conditions can change rapidly</li>
                        <li>• External factors not captured in technical analysis</li>
                      </ul>
                    </div>
                    <div>
                      <strong>Important Notes:</strong>
                      <ul className="mt-1 ml-4 space-y-1 text-gray-600">
                        <li>• This system is for educational purposes only</li>
                        <li>• Always consult qualified financial advisors</li>
                        <li>• Never trade with money you cannot afford to lose</li>
                      </ul>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </main>
    </div>
  )
}

export default App

