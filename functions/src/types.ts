import { FieldValue } from "firebase-admin/firestore";

export interface CoinGeckoMarketChartResponse {
  prices: [number, number][];
  market_caps: [number, number][];
  total_volumes: [number, number][];
}

export enum Recommendation {
  Hold = "hold",
  HoldBasedOnBuyPrice = "hold based on buy price",
  Sell = "sell",
}

export enum CryptoIds {
  Cardano = "cardano",
}

export enum Currencies {
  USD = "usd",
}

export enum Collections {
  Config = "config",
  Models = "models",
}

export enum Docs {
  PriceAlert = "priceAlert",
  SellPredictor = "sellPredictor",
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
  priceChangePct: number;
}

export interface PredictSell {
  metConditions: string[];
  probability: number;
  recommendation: Recommendation;
}

export interface SellDecision {
  cryptoSymbol: string;
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
  probability: string;
  recommendation: Recommendation;
  timestamp: FieldValue;
}

export interface Condition {
  name: string;
  met: boolean;
  weight: number;
}

export interface RecieveSMSRequest {
  body: {
    text: string;
  };
}
