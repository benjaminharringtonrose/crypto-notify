import axios, { AxiosInstance } from "axios";
import * as crypto from "crypto";
import { Trade } from "../types";

interface CoinbaseConfig {
  apiKey: string;
  apiSecret: string;
  baseUrl?: string;
}

export class CoinbaseTradeExecutor {
  private apiKey: string;
  private apiSecret: string;
  private client: AxiosInstance;
  private baseUrl: string;

  constructor(config: CoinbaseConfig) {
    this.apiKey = config.apiKey;
    this.apiSecret = config.apiSecret;
    this.baseUrl =
      config.baseUrl || "https://api.coinbase.com/api/v3/brokerage";
    this.client = axios.create({
      baseURL: this.baseUrl,
      headers: { "Content-Type": "application/json" },
    });
  }

  private generateAuthHeaders(
    method: string,
    endpoint: string,
    body: string = ""
  ): Record<string, string> {
    const timestamp = Math.floor(Date.now() / 1000).toString();
    const message = `${timestamp}${method}${endpoint}${body}`;
    const signature = crypto
      .createHmac("sha256", this.apiSecret)
      .update(message)
      .digest("hex");
    return {
      "CB-ACCESS-KEY": this.apiKey,
      "CB-ACCESS-SIGN": signature,
      "CB-ACCESS-TIMESTAMP": timestamp,
    };
  }

  public async executeTrade(trade: Trade): Promise<any> {
    const endpoint = "/orders";
    const orderPayload = {
      client_order_id: crypto.randomUUID(),
      product_id: "ADA-USD", // Hardcoded for now; adapt as needed
      side: trade.type.toUpperCase(),
      order_configuration: {
        market_market_ioc: {
          base_size: trade.adaAmount.toString(),
        },
      },
    };

    const method = "POST";
    const body = JSON.stringify(orderPayload);
    const headers = this.generateAuthHeaders(method, endpoint, body);

    try {
      const response = await this.client.post(endpoint, orderPayload, {
        headers,
      });
      console.log(
        `Trade executed: ${trade.type} ${trade.adaAmount} ADA at $${trade.price}`
      );
      return response.data;
    } catch (error: any) {
      console.error(
        "Trade execution failed:",
        error.response?.data || error.message
      );
      throw error;
    }
  }

  public async getMarketData(
    productId: string = "ADA-USD"
  ): Promise<{ prices: number[]; volumes: number[] }> {
    const endpoint = `/products/${productId}/candles`;
    const method = "GET";
    const params = {
      granularity: 86400, // Daily candles (86400 seconds)
      limit: 30, // Match TIMESTEPS from constants.ts
    };
    const headers = this.generateAuthHeaders(method, endpoint);

    try {
      const response = await this.client.get(endpoint, { headers, params });
      const candles = response.data.candles;
      const prices = candles
        .map((candle: any) => parseFloat(candle.close))
        .reverse();
      const volumes = candles
        .map((candle: any) => parseFloat(candle.volume))
        .reverse();
      return { prices, volumes };
    } catch (error: any) {
      console.error(
        "Failed to fetch market data:",
        error.response?.data || error.message
      );
      throw error;
    }
  }

  public async getAccountBalance(): Promise<any> {
    const endpoint = "/accounts";
    const method = "GET";
    const headers = this.generateAuthHeaders(method, endpoint);

    try {
      const response = await this.client.get(endpoint, { headers });
      return response.data;
    } catch (error: any) {
      console.error(
        "Failed to fetch balance:",
        error.response?.data || error.message
      );
      throw error;
    }
  }
}
