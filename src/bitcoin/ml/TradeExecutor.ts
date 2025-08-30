import {
  CBAdvancedTradeClient,
  GetAdvTradeProductCandlesRequest,
} from "coinbase-api";
import * as crypto from "crypto";
import {
  AdvTradeAccount,
  AdvTradeAccountsList,
  AdvTradeCandle,
  CoinbaseCurrency,
  CoinbaseProductIds,
  Granularity,
  Recommendation,
  Trade,
} from "../../types";
import { TIME_CONVERSIONS } from "../../constants";

interface CoinbaseClientConfig {
  apiKey?: string;
  apiSecret?: string;
}

export class TradeExecutor {
  private client: CBAdvancedTradeClient;

  constructor(config: CoinbaseClientConfig) {
    this.client = new CBAdvancedTradeClient({
      apiKey: config.apiKey,
      apiSecret: config.apiSecret,
    });
  }

  public async executeTrade(trade: Trade): Promise<any> {
    if (trade.type === Recommendation.Hold) {
      return;
    }
    const orderParams = {
      product_id: CoinbaseProductIds.BTC,
      side: trade.type,
      order_configuration: {
        market_market_ioc: {
          base_size: trade.btcAmount.toString(),
        },
      },
      client_order_id: crypto.randomUUID(),
    };

    try {
      const response = await this.client.submitOrder(orderParams);
      console.log(
        `Trade executed: ${trade.type} ${trade.btcAmount} BTC at $${trade.price}`
      );
      return response;
    } catch (error: any) {
      console.error("Trade execution failed:", error.message || error);
      throw error;
    }
  }

  public async getCurrentPrice(product_id: CoinbaseProductIds) {
    try {
      // Calculate timestamps for a short recent time range (e.g., last hour)
      const now = Math.floor(Date.now() / 1000); // Current time in seconds (March 31, 2025)
      const oneHourAgo = now - TIME_CONVERSIONS.ONE_HOUR_IN_SECONDS;

      // Fetch the latest candlestick data
      const response = await this.getMarketData({
        product_id,
        granularity: Granularity.OneHour,
        start: oneHourAgo.toString(),
        end: now.toString(),
      });

      // Get the most recent closing price
      const currentPrice = response.prices[response.prices.length - 1];

      console.log(
        `Current BTC-USD Price (as of ${new Date().toISOString()}): $${currentPrice.toFixed(
          2
        )}`
      );
      return currentPrice;
    } catch (error) {
      console.error("Error fetching current price:", error);
      throw error;
    }
  }

  public async getMarketData(
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

  public async getAccountBalances(): Promise<{
    usd?: AdvTradeAccount;
    btc?: AdvTradeAccount;
  }> {
    try {
      const response: AdvTradeAccountsList = await this.client.getAccounts();
      return {
        usd: response.accounts.find((i) => i.currency === CoinbaseCurrency.USD),
        btc: response.accounts.find((i) => i.currency === CoinbaseCurrency.BTC),
      };
    } catch (error: any) {
      console.error("Failed to fetch balance:", error.message || error);
      throw error;
    }
  }
}
