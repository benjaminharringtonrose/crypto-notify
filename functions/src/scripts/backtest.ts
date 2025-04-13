import { TradeModelBacktester } from "../cardano/TradeModelBacktester";
import { CoinbaseProductIds, Granularity } from "../types";
import { CoinbaseService } from "../api/CoinbaseService";
import { MODEL_CONFIG } from "../constants";

async function fetchDataInChunks(
  coinbase: CoinbaseService,
  productId: CoinbaseProductIds,
  start: number,
  end: number,
  granularity: Granularity,
  chunkDays: number = 300
): Promise<{ prices: number[]; volumes: number[] }> {
  const chunkSeconds = chunkDays * 86400;
  const chunks: { prices: number[]; volumes: number[] }[] = [];
  let currentStart = start;

  while (currentStart < end) {
    const chunkEnd = Math.min(currentStart + chunkSeconds, end);
    try {
      const chunkData = await coinbase.getPricesAndVolumes({
        product_id: productId,
        granularity,
        start: currentStart.toString(),
        end: chunkEnd.toString(),
      });
      console.log(
        `Fetched chunk for ${productId} (${currentStart} to ${chunkEnd}): ${chunkData.prices.length} candles`
      );
      chunks.push(chunkData);
    } catch (error) {
      console.error(
        `Error fetching chunk for ${productId} (${currentStart} to ${chunkEnd}):`,
        error
      );
      throw error;
    }
    currentStart = chunkEnd + 86400; // Move to next day
  }

  // Combine chunks, ensuring chronological order
  const combined = chunks.reduce(
    (acc, chunk) => ({
      prices: [...acc.prices, ...chunk.prices],
      volumes: [...acc.volumes, ...chunk.volumes],
    }),
    { prices: [], volumes: [] }
  );

  // Validate lengths
  if (combined.prices.length !== combined.volumes.length) {
    throw new Error(
      `Mismatched lengths for ${productId}: prices (${combined.prices.length}) vs volumes (${combined.volumes.length})`
    );
  }

  return combined;
}

async function runMultiPeriodBacktest() {
  const backtester = new TradeModelBacktester();
  const timesteps = MODEL_CONFIG.TIMESTEPS;

  const coinbase = new CoinbaseService({
    apiKey: process.env.COINBASE_API_KEY,
    apiSecret: process.env.COINBASE_API_SECRET,
  });

  // Current Unix timestamp (seconds) for April 13, 2025
  const currentTimestamp = Math.floor(Date.now() / 1000);

  // Define periods with Unix timestamps (seconds)
  const periods = [
    {
      name: "Recent 500 Days",
      start: currentTimestamp - 500 * 86400, // ~April 29, 2024
      end: currentTimestamp, // April 13, 2025
    },
    {
      name: "Middle 500 Days",
      start: currentTimestamp - 1000 * 86400, // ~November 30, 2022
      end: currentTimestamp - 500 * 86400, // ~April 29, 2024
    },
    {
      name: "Older 300 Days", // Shortened to avoid data issues
      start: currentTimestamp - 1300 * 86400, // ~September 25, 2021
      end: currentTimestamp - 1000 * 86400, // ~November 30, 2022
    },
    {
      name: "Full Period",
      start: currentTimestamp - 1000 * 86400, // ~July 16, 2022
      end: currentTimestamp, // April 13, 2025
    },
  ];

  for (const period of periods) {
    console.log(
      `Fetching data for ${period.name} (Timestamp ${period.start} to ${period.end})...`
    );

    try {
      const adaData = await fetchDataInChunks(
        coinbase,
        CoinbaseProductIds.ADA,
        period.start,
        period.end,
        Granularity.OneDay
      );

      const btcData = await fetchDataInChunks(
        coinbase,
        CoinbaseProductIds.BTC,
        period.start,
        period.end,
        Granularity.OneDay
      );

      // Align data lengths
      const minLength = Math.min(adaData.prices.length, btcData.prices.length);
      if (
        adaData.prices.length !== btcData.prices.length ||
        adaData.volumes.length !== btcData.volumes.length ||
        adaData.prices.length !== adaData.volumes.length
      ) {
        console.warn(
          `Data length mismatch for ${period.name}: ADA prices=${adaData.prices.length}, volumes=${adaData.volumes.length}, BTC prices=${btcData.prices.length}, volumes=${btcData.volumes.length}. Truncating to ${minLength}.`
        );
        adaData.prices = adaData.prices.slice(0, minLength);
        adaData.volumes = adaData.volumes.slice(0, minLength);
        btcData.prices = btcData.prices.slice(0, minLength);
        btcData.volumes = btcData.volumes.slice(0, minLength);
      }

      const totalDays = adaData.prices.length;

      // Use array indices
      const startIndex = timesteps;
      const endIndex = totalDays - 1;

      if (endIndex - startIndex < timesteps) {
        console.warn(
          `Skipping ${period.name}: insufficient data (length: ${
            endIndex - startIndex
          })`
        );
        continue;
      }

      console.log(
        `Backtesting ${period.name} (Days ${totalDays - endIndex} to ${
          totalDays - startIndex
        })...`
      );
      const result = await backtester.backtest(
        adaData,
        btcData,
        startIndex,
        endIndex
      );
      await backtester.evaluateBacktest(result);
    } catch (error) {
      console.error(`Error in ${period.name}:`, error);
    }
  }
}

async function main() {
  await runMultiPeriodBacktest();
}

main().catch(console.error);
