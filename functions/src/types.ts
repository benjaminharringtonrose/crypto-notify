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
  rsi: number | undefined;
  prevRsi: number | undefined;
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
  momentum: number | undefined;
  priceChangePct: number | undefined;
  isTripleBottom?: boolean;
  volAdjustedMomentum: number;
  sma20: number;
  lowerBand: number;
  obv: number;
  stochRsiSignal: number;
  adxProxy: number; // New field for trend strength
  btcRsi?: number | undefined;
  btcPrevRsi?: number | undefined;
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
  btcMomentum?: number | undefined;
  btcPriceChangePct?: number | undefined;
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
  HoldBasedOnBuyPrice = "HOLD_BASED_ON_BUY_PRICE",
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
