import axios from "axios";
import { CRYPTOCOMPARE_API_URL } from "../constants";

export const getHistoricalData = async (
  cryptoSymbol: string,
  totalDays: number,
  chunkDays: number = 90
): Promise<{ prices: number[]; volumes: number[] }> => {
  const prices: number[] = [];
  const volumes: number[] = [];
  const millisecondsPerDay = 24 * 60 * 60 * 1000;
  const endDate = new Date();
  const totalMilliseconds = totalDays * millisecondsPerDay;
  const chunkMilliseconds = chunkDays * millisecondsPerDay;

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

      const toTimestamp = Math.floor(chunkEnd.getTime() / 1000);
      const limit = Math.min(
        chunkDays,
        Math.ceil(
          (toTimestamp - Math.floor(actualChunkStart.getTime() / 1000)) /
            (24 * 60 * 60)
        )
      );

      // Add delay before each request (50ms ensures < 20 calls/second)
      if (i > 0) await delay(50);

      const response = await axios.get(`${CRYPTOCOMPARE_API_URL}/histoday`, {
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
    console.error(`Error fetching historical data for ${cryptoSymbol}:`, error);
    throw error;
  }
};
