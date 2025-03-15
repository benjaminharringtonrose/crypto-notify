import { FieldValue } from "firebase-admin/firestore";

export interface CoinGeckoMarketChartResponse {
  prices: [number, number][];
  market_caps: [number, number][];
  total_volumes: [number, number][];
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
  recommendation: "sell" | "hold";
  timestamp: FieldValue;
}
