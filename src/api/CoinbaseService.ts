import {
  CBAdvancedTradeClient,
  GetAdvTradeProductCandlesRequest,
} from "coinbase-api";
import { AdvTradeCandle, CoinbaseProductIds, Granularity } from "../types";
import { TIME_CONVERSIONS } from "../constants";

interface CoinbaseClientConfig {
  apiKey?: string;
  apiSecret?: string;
}

export class CoinbaseService {
  private client: CBAdvancedTradeClient;

  constructor(config: CoinbaseClientConfig) {
    this.client = new CBAdvancedTradeClient({
      apiKey: config.apiKey,
      apiSecret: config.apiSecret,
    });
  }

  public async getCurrentPrice(product_id: CoinbaseProductIds) {
    try {
      const now = Math.floor(
        Date.now() / TIME_CONVERSIONS.ONE_SECOND_IN_MILLISECONDS
      );
      const oneHourAgo = now - TIME_CONVERSIONS.ONE_HOUR_IN_SECONDS;

      const response: {
        candles: AdvTradeCandle[];
      } = await this.client.getProductCandles({
        product_id,
        granularity: Granularity.OneHour,
        start: oneHourAgo.toString(),
        end: now.toString(),
      });

      const prices = response.candles
        .map((candle) => parseFloat(candle.close))
        .reverse();

      const currentPrice = prices[prices.length - 1];

      return currentPrice;
    } catch (error: any) {
      console.error("Failed to get current price:", error.message || error);
      throw error;
    }
  }

  public async getPricesAndVolumes(
    request: GetAdvTradeProductCandlesRequest
  ): Promise<{ prices: number[]; volumes: number[] }> {
    try {
      const response: {
        candles: AdvTradeCandle[];
      } = await this.client.getProductCandles(request);
      const candles = response.candles;
      const prices = candles
        .map((candle) => parseFloat(candle.close))
        .reverse();
      const volumes = candles
        .map((candle) => parseFloat(candle.volume))
        .reverse();
      return { prices, volumes };
    } catch (error: any) {
      console.error("Failed to fetch market data:", error.message || error);
      throw error;
    }
  }
}
