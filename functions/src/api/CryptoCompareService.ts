import axios from "axios";
import { TIME_CONVERSIONS } from "../constants";

export interface BitcoinDataPoint {
  Date: Date;
  Open: number;
  High: number;
  Low: number;
  Close: number;
  Volume: number;
  Dividends?: number;
  Stock_Splits?: number;
}

export class CryptoCompareService {
  private baseUrl: string = "https://min-api.cryptocompare.com/data/v2";

  public constructor() {}

  public async getHistoricalData(
    cryptoSymbol: string,
    totalDays: number,
    chunkDays: number = 90
  ): Promise<{ prices: number[]; volumes: number[] }> {
    const prices: number[] = [];
    const volumes: number[] = [];

    const endDate = new Date();
    const totalMilliseconds =
      totalDays * TIME_CONVERSIONS.ONE_DAY_IN_MILLISECONDS;
    const chunkMilliseconds =
      chunkDays * TIME_CONVERSIONS.ONE_DAY_IN_MILLISECONDS;

    const numChunks = Math.ceil(totalDays / chunkDays);

    // Throttle function to delay requests
    const delay = (ms: number) =>
      new Promise((resolve) => setTimeout(resolve, ms));

    try {
      for (let i = 0; i < numChunks; i++) {
        const chunkEnd = new Date(endDate.getTime() - i * chunkMilliseconds);
        const chunkStart = new Date(chunkEnd.getTime() - chunkMilliseconds);
        const actualChunkStart =
          chunkStart < new Date(endDate.getTime() - totalMilliseconds)
            ? new Date(endDate.getTime() - totalMilliseconds)
            : chunkStart;

        const toTimestamp = Math.floor(
          chunkEnd.getTime() / TIME_CONVERSIONS.ONE_SECOND_IN_MILLISECONDS
        );
        const limit = Math.min(
          chunkDays,
          Math.ceil(
            (toTimestamp -
              Math.floor(
                actualChunkStart.getTime() /
                  TIME_CONVERSIONS.ONE_SECOND_IN_MILLISECONDS
              )) /
              TIME_CONVERSIONS.ONE_DAY_IN_SECONDS
          )
        );

        // Add delay before each request (50ms ensures < 20 calls/second)
        if (i > 0) await delay(50);

        const response = await axios.get(`${this.baseUrl}/histoday`, {
          params: {
            fsym: cryptoSymbol.toUpperCase(),
            tsym: "USD",
            limit: limit,
            toTs: toTimestamp,
            api_key: process.env.CRYPTOCOMPARE_API_KEY,
          },
        });

        if (
          !response.data ||
          !response.data.Data ||
          !Array.isArray(response.data.Data.Data)
        ) {
          throw new Error(
            `Invalid API response structure: ${JSON.stringify(response.data)}`
          );
        }

        const data = response.data.Data.Data;

        if (data.length === 0) {
          console.warn(
            `Empty data array received for ${cryptoSymbol} chunk ${i + 1}`
          );
          continue;
        }

        const chunkPrices = data.map((entry: any) => {
          if (entry.close === undefined) {
            throw new Error(
              `Missing 'close' price in entry: ${JSON.stringify(entry)}`
            );
          }
          return entry.close;
        });
        const chunkVolumes = data.map((entry: any) => {
          if (entry.volumeto === undefined) {
            throw new Error(
              `Missing 'volumeto' in entry: ${JSON.stringify(entry)}`
            );
          }
          return entry.volumeto;
        });

        prices.unshift(...chunkPrices);
        volumes.unshift(...chunkVolumes);
      }

      if (prices.length === 0 || volumes.length === 0) {
        throw new Error(`No valid data retrieved for ${cryptoSymbol}`);
      }

      return { prices, volumes };
    } catch (error) {
      console.error(
        `Error fetching historical data for ${cryptoSymbol}:`,
        error
      );
      throw error;
    }
  }

  /**
   * Fetch full OHLCV historical data for Bitcoin price prediction
   * Returns data compatible with the ensemble system's BitcoinDataPoint interface
   */
  public async getFullHistoricalData(
    cryptoSymbol: string = "BTC",
    totalDays: number = 365,
    chunkDays: number = 90
  ): Promise<BitcoinDataPoint[]> {
    console.log(`Fetching ${totalDays} days of full OHLCV data for ${cryptoSymbol}...`);

    const allData: BitcoinDataPoint[] = [];

    const endDate = new Date();
    const totalMilliseconds =
      totalDays * TIME_CONVERSIONS.ONE_DAY_IN_MILLISECONDS;
    const chunkMilliseconds =
      chunkDays * TIME_CONVERSIONS.ONE_DAY_IN_MILLISECONDS;

    const numChunks = Math.ceil(totalDays / chunkDays);

    // Throttle function to delay requests
    const delay = (ms: number) =>
      new Promise((resolve) => setTimeout(resolve, ms));

    try {
      for (let i = 0; i < numChunks; i++) {
        const chunkEnd = new Date(endDate.getTime() - i * chunkMilliseconds);
        const chunkStart = new Date(chunkEnd.getTime() - chunkMilliseconds);
        const actualChunkStart =
          chunkStart < new Date(endDate.getTime() - totalMilliseconds)
            ? new Date(endDate.getTime() - totalMilliseconds)
            : chunkStart;

        const toTimestamp = Math.floor(
          chunkEnd.getTime() / TIME_CONVERSIONS.ONE_SECOND_IN_MILLISECONDS
        );
        const limit = Math.min(
          chunkDays,
          Math.ceil(
            (toTimestamp -
              Math.floor(
                actualChunkStart.getTime() /
                  TIME_CONVERSIONS.ONE_SECOND_IN_MILLISECONDS
              )) /
              TIME_CONVERSIONS.ONE_DAY_IN_SECONDS
          )
        );

        // Add delay before each request (50ms ensures < 20 calls/second)
        if (i > 0) await delay(50);

        const response = await axios.get(`${this.baseUrl}/histoday`, {
          params: {
            fsym: cryptoSymbol.toUpperCase(),
            tsym: "USD",
            limit: limit,
            toTs: toTimestamp,
            api_key: process.env.CRYPTOCOMPARE_API_KEY,
          },
        });

        if (
          !response.data ||
          !response.data.Data ||
          !Array.isArray(response.data.Data.Data)
        ) {
          throw new Error(
            `Invalid API response structure: ${JSON.stringify(response.data)}`
          );
        }

        const data = response.data.Data.Data;

        if (data.length === 0) {
          console.warn(
            `Empty data array received for ${cryptoSymbol} chunk ${i + 1}`
          );
          continue;
        }

        // Convert CryptoCompare data format to BitcoinDataPoint format
        const chunkData: BitcoinDataPoint[] = data.map((entry: any) => {
          // Validate required fields
          if (entry.time === undefined || entry.open === undefined || 
              entry.high === undefined || entry.low === undefined || 
              entry.close === undefined || entry.volumeto === undefined) {
            throw new Error(
              `Missing required fields in entry: ${JSON.stringify(entry)}`
            );
          }

          return {
            Date: new Date(entry.time * 1000), // Convert Unix timestamp to Date
            Open: entry.open,
            High: entry.high,
            Low: entry.low,
            Close: entry.close,
            Volume: entry.volumeto,
            Dividends: 0, // Bitcoin doesn't have dividends
            Stock_Splits: 0, // Bitcoin doesn't have stock splits
          };
        });

        // Add to beginning of array to maintain chronological order
        allData.unshift(...chunkData);
      }

      if (allData.length === 0) {
        throw new Error(`No valid data retrieved for ${cryptoSymbol}`);
      }

      // Sort by date to ensure chronological order
      allData.sort((a, b) => a.Date.getTime() - b.Date.getTime());

      console.log(`Successfully fetched ${allData.length} days of ${cryptoSymbol} data`);
      console.log(`Date range: ${allData[0].Date.toISOString()} to ${allData[allData.length - 1].Date.toISOString()}`);
      console.log(`Price range: $${Math.min(...allData.map(d => d.Close)).toLocaleString()} - $${Math.max(...allData.map(d => d.Close)).toLocaleString()}`);

      return allData;
    } catch (error) {
      console.error(
        `Error fetching full historical data for ${cryptoSymbol}:`,
        error
      );
      throw error;
    }
  }
}
