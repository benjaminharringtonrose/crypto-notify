export interface CoinGeckoPriceResponse {
  cardano: {
    usd: number;
  };
  bitcoin: {
    usd: number;
  };
}

export interface CoinGeckoMarketChartResponse {
  prices: [number, number][];
  market_caps: [number, number][];
  total_volumes: [number, number][];
}

export interface Indicators {
  rsi?: number;
  prevRsi?: number;
  sma7: number;
  sma21: number;
  prevSma7: number;
  prevSma21: number;
  macdLine: number;
  signalLine: number;
  currentPrice: number;
  upperBand: number;
  obvValues: number[];
  atr: number;
  atrBaseline: number;
  zScore: number;
  vwap: number;
  stochRsi: number;
  prevStochRsi: number;
  fib61_8: number;
  prices: number[];
  volumeOscillator: number;
  prevVolumeOscillator: number;
  isDoubleTop: boolean;
  isHeadAndShoulders: boolean;
  prevMacdLine: number;
  isTripleTop: boolean;
  isVolumeSpike: boolean;
  momentum: number;
  priceChangePct?: number;
  isTripleBottom: boolean;
  volAdjustedMomentum: number;
  sma20: number;
  lowerBand: number;
  obv: number;
  stochRsiSignal: number;
  adxProxy: number;
  btcRsi?: number;
  btcPrevRsi?: number;
  btcSma7?: number;
  btcSma21?: number;
  btcPrevSma7?: number;
  btcPrevSma21?: number;
  btcMacdLine?: number;
  btcSignalLine?: number;
  btcUpperBand?: number;
  btcObvValues?: number[];
  btcAtr?: number;
  btcAtrBaseline?: number;
  btcZScore?: number;
  btcVwap?: number;
  btcStochRsi?: number;
  btcPrevStochRsi?: number;
  btcFib61_8?: number;
  btcPrices?: number[];
  btcVolumeOscillator?: number;
  btcPrevVolumeOscillator?: number;
  btcIsDoubleTop?: boolean;
  btcIsHeadAndShoulders?: boolean;
  btcPrevMacdLine?: number;
  btcIsTripleTop?: boolean;
  btcIsVolumeSpike?: boolean;
  btcMomentum?: number;
  btcPriceChangePct?: number;
  btcVolAdjustedMomentum?: number;
}

export interface StrategyResult {
  buyProb: number;
  sellProb: number;
  holdProb: number;
  recommendation: Recommendation;
}

export interface Probabilities {
  buy: number;
  hold: number;
  sell: number;
}

export interface TradeDecision {
  currentPrice: number;
  rsi?: string;
  sma7: string;
  sma21: string;
  macdLine: string;
  signalLine: string;
  sma20: string;
  upperBand: string;
  lowerBand: string;
  obv: string;
  atr: string;
  zScore: string;
  vwap: string;
  stochRsi: string;
  stochRsiSignal: string;
  fib61_8: string;
  volumeOscillator: string;
  metConditions: string[];
  probabilities: Probabilities;
  recommendation: Recommendation;
  timestamp: any;
  profitPotential: number;
  isDoubleTop: boolean;
  isTripleTop: boolean;
  isHeadAndShoulders: boolean;
  isTripleBottom: boolean;
  momentum: number;
}

export interface RecieveSMSRequest {
  body: {
    text: string;
  };
}

export interface TradeRecommendation {
  probability: number;
  recommendation: Recommendation;
  timestamp: string;
}

export interface MarketData {
  prices: number[];
  volumes: number[];
  dayIndex: number;
  currentPrice: number;
}

export enum Recommendation {
  Buy = "BUY",
  Hold = "HOLD",
  Sell = "SELL",
}

export enum CryptoIds {
  Bitcoin = "bitcoin",
  Cardano = "cardano",
}

export enum Currencies {
  USD = "usd",
}

export enum Collections {
  Config = "config",
  TradeRecommendations = "tradeRecommendations",
}

export enum Docs {
  PriceAlert = "priceAlert",
  Cardano = "cardano",
}

export enum TradingStrategy {
  TrendFollowing = "TrendFollowing",
  MeanReversion = "MeanReversion",
  Breakout = "Breakout",
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
export interface FeatureStats {
  mean: number;
  std: number;
}

export interface Trade {
  type: Recommendation;
  price: number;
  timestamp: string;
  adaAmount: number;
  usdValue: number;
  buyPrice?: number;
}

export interface BacktestResult {
  totalReturn: number;
  totalTrades: number;
  winRate: number;
  sharpeRatio: number;
  maxDrawdown: number;
  portfolioHistory: { timestamp: string; value: number }[];
  trades: Trade[];
}
