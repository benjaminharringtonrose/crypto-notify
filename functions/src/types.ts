export interface Indicators {
  rsi: number;
  prevRsi: number;
  sma7: number;
  sma21: number;
  prevSma7: number;
  prevSma21: number;
  macdLine: number;
  signalLine: number;
  currentPrice: number;
  upperBand: number;
  lowerBand: number;
  obvValues: number[];
  obv: number;
  atr: number;
  atrBaseline: number;
  zScore: number;
  vwap: number;
  stochRsi: number;
  stochRsiSignal: number;
  prevStochRsi: number;
  fib61_8: number;
  volumeOscillator: number;
  prevVolumeOscillator: number;
  isDoubleTop: boolean;
  isHeadAndShoulders: boolean;
  prevMacdLine: number;
  isTripleTop: boolean;
  isTripleBottom: boolean;
  isVolumeSpike: boolean;
  momentum: number;
  priceChangePct: number;
  sma20: number;
  sma50: number;
  sma200: number;
  prices: number[];
  volAdjustedMomentum: number;
  trendRegime: number;
  adxProxy: number;
  // Market Regime Features
  volatilityRegimeScore: number;
  trendRegimeScore: number;
  momentumRegimeScore: number;
  realizedVolatility: number;
  regimeScore: number;
  // EXPERIMENT #61: Advanced Market Microstructure Features
  ichimokuTenkanSen: number;
  ichimokuKijunSen: number;
  ichimokuCloudPosition: number;
  williamsR: number;
  stochasticK: number;
  vpt: number;
  cci: number;
  mfi: number;
  aroonOscillator: number;
}

export interface Probabilities {
  buy: number;
  hold: number;
  sell: number;
}

export interface TradeRecommendation {
  probability: number;
  recommendation: Recommendation;
  timestamp: string;
}

export enum Recommendation {
  Buy = "BUY",
  Hold = "HOLD",
  Sell = "SELL",
}

export enum Granularity {
  OneMinute = "ONE_MINUTE",
  FiveMinute = "FIVE_MINUTE",
  FifteenMinute = "FIFTEEN_MINUTE",
  ThirtyMinute = "THIRTY_MINUTE",
  OneHour = "ONE_HOUR",
  TwoHour = "TWO_HOUR",
  SixHour = "SIX_HOUR",
  OneDay = "ONE_DAY",
}

export enum CryptoIds {
  Bitcoin = "bitcoin",
}

export enum Collections {
  Config = "config",
  TradeRecommendations = "tradeRecommendations",
}

export enum Docs {
  PriceAlert = "priceAlert",
  Bitcoin = "bitcoin",
}

export interface HistoricalData {
  prices: number[];
  volumes: number[];
}
export interface ModelConfig {
  timesteps: number;
  epochs: number;
  batchSize: number;
  initialLearningRate: number;
}

export interface Trade {
  type: Recommendation;
  price: number;
  timestamp: string;
  btcAmount: number;
  usdValue: number;
  buyPrice?: number;
}

export enum StrategyType {
  Momentum = "momentum",
  MeanReversion = "mean_reversion",
  Breakout = "breakout",
  TrendFollowing = "trend_following",
}

export enum CoinbaseProductIds {
  BTC = "BTC-USD",
}

export enum CoinbaseCurrency {
  BTC = "BTC",
  USD = "USD",
}

export interface AdvTradeCandle {
  start: string;
  low: string;
  high: string;
  open: string;
  close: string;
  volume: string;
}

interface Balance {
  value: string;
  currency: string;
}

export interface AdvTradeAccount {
  uuid: string;
  name: string;
  currency: string;
  available_balance: Balance;
  default: boolean;
  active: boolean;
  created_at: string;
  updated_at: string;
  deleted_at: string;
  type: string;
  ready: boolean;
  hold: Balance;
  retail_portfolio_id: string;
  platform: string;
}

export interface AdvTradeAccountsList {
  accounts: AdvTradeAccount[];
  has_next: boolean;
  cursor: string;
  size: number;
}

export interface GetAdvTradeProductCandlesRequest {
  product_id: string;
  start: string;
  end: string;
  granularity:
    | "UNKNOWN_GRANULARITY"
    | "ONE_MINUTE"
    | "FIVE_MINUTE"
    | "FIFTEEN_MINUTE"
    | "THIRTY_MINUTE"
    | "ONE_HOUR"
    | "TWO_HOUR"
    | "SIX_HOUR"
    | "ONE_DAY";
  limit?: number;
}
