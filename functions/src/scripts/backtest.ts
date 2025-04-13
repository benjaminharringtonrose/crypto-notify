import { TradeModelBacktester } from "../cardano/TradeModelBacktester";
import { CoinbaseService } from "../api/CoinbaseService";
import { HistoricalData } from "../types";
import { TIME_CONVERSIONS } from "../constants";

async function fetchHistoricalData(
  coinbaseService: CoinbaseService,
  productId: string,
  start: number,
  end: number
): Promise<HistoricalData> {
  const granularity = "ONE_DAY";
  const startUnix = Math.floor(start / 1000);
  const endUnix = Math.floor(end / 1000);
  let maxCandles = 300; // Start with 300, reduce if needed
  const oneDaySec = TIME_CONVERSIONS.ONE_DAY_IN_SECONDS;
  const chunks: { prices: number[]; volumes: number[] }[] = [];

  let currentStart = startUnix;
  while (currentStart < endUnix) {
    const currentEnd = Math.min(currentStart + maxCandles * oneDaySec, endUnix);
    try {
      console.log(
        `Fetching ${productId} data (Timestamp ${currentStart} to ${currentEnd})...`
      );
      const { prices, volumes } = await coinbaseService.getPricesAndVolumes({
        product_id: productId,
        granularity,
        start: currentStart.toString(),
        end: currentEnd.toString(),
      });
      console.log(`Fetched chunk for ${productId}: ${prices.length} candles`);
      chunks.push({ prices, volumes });
      currentStart = currentEnd + oneDaySec; // Avoid overlap
    } catch (error: any) {
      console.warn(`Fetch failed for ${productId}:`, error.message || error);
      if (maxCandles > 100) {
        maxCandles = Math.max(100, Math.floor(maxCandles / 2)); // Reduce chunk size
        console.log(`Retrying with smaller chunk size: ${maxCandles} candles`);
        continue;
      }
      throw error;
    }
  }

  // Merge chunks, reversing to maintain chronological order
  const mergedPrices: number[] = [];
  const mergedVolumes: number[] = [];
  for (const chunk of chunks.reverse()) {
    mergedPrices.push(...chunk.prices);
    mergedVolumes.push(...chunk.volumes);
  }

  return { prices: mergedPrices, volumes: mergedVolumes };
}

async function runBacktest() {
  const backtester = new TradeModelBacktester(10000);
  const coinbaseService = new CoinbaseService({
    apiKey: process.env.COINBASE_API_KEY,
    apiSecret: process.env.COINBASE_API_SECRET,
  });

  const recentDays = 500;
  const middleDays = 500;
  const olderDays = 300;
  const now = Date.now();
  const oneDayMs = 24 * 60 * 60 * 1000;

  // Recent: April 2024 - April 2025
  const recentEnd = now;
  const recentStart = recentEnd - recentDays * oneDayMs;
  const recentAdaData = await fetchHistoricalData(
    coinbaseService,
    "ADA-USD",
    recentStart,
    recentEnd
  );
  const recentBtcData = await fetchHistoricalData(
    coinbaseService,
    "BTC-USD",
    recentStart,
    recentEnd
  );
  // Trim to shortest length
  const recentMinLength = Math.min(
    recentAdaData.prices.length,
    recentAdaData.volumes.length,
    recentBtcData.prices.length,
    recentBtcData.volumes.length
  );
  recentAdaData.prices = recentAdaData.prices.slice(0, recentMinLength);
  recentAdaData.volumes = recentAdaData.volumes.slice(0, recentMinLength);
  recentBtcData.prices = recentBtcData.prices.slice(0, recentMinLength);
  recentBtcData.volumes = recentBtcData.volumes.slice(0, recentMinLength);
  console.log(`Recent Data Trimmed to ${recentMinLength} candles`);
  console.log(
    `Backtesting Recent ${recentDays} Days (Days 1 to ${
      recentAdaData.prices.length - 30
    })...`
  );
  const recentResult = await backtester.backtest(
    recentAdaData,
    recentBtcData,
    30,
    recentAdaData.prices.length - 1
  );
  await backtester.evaluateBacktest(recentResult);

  // Middle: November 2022 - April 2024
  const middleEnd = recentStart - oneDayMs;
  const middleStart = middleEnd - middleDays * oneDayMs;
  const middleAdaData = await fetchHistoricalData(
    coinbaseService,
    "ADA-USD",
    middleStart,
    middleEnd
  );
  const middleBtcData = await fetchHistoricalData(
    coinbaseService,
    "BTC-USD",
    middleStart,
    middleEnd
  );
  // Trim to shortest length
  const middleMinLength = Math.min(
    middleAdaData.prices.length,
    middleAdaData.volumes.length,
    middleBtcData.prices.length,
    middleBtcData.volumes.length
  );
  middleAdaData.prices = middleAdaData.prices.slice(0, middleMinLength);
  middleAdaData.volumes = middleAdaData.volumes.slice(0, middleMinLength);
  middleBtcData.prices = middleBtcData.prices.slice(0, middleMinLength);
  middleBtcData.volumes = middleBtcData.volumes.slice(0, middleMinLength);
  console.log(`Middle Data Trimmed to ${middleMinLength} candles`);
  console.log(
    `Backtesting Middle ${middleDays} Days (Days 1 to ${
      middleAdaData.prices.length - 30
    })...`
  );
  const middleResult = await backtester.backtest(
    middleAdaData,
    middleBtcData,
    30,
    middleAdaData.prices.length - 1
  );
  await backtester.evaluateBacktest(middleResult);

  // Older: January 1, 2021 - October 28, 2021
  const olderEnd = new Date("2021-10-28").getTime();
  const olderStart = new Date("2021-01-01").getTime();
  const olderAdaData = await fetchHistoricalData(
    coinbaseService,
    "ADA-USD",
    olderStart,
    olderEnd
  );
  const olderBtcData = await fetchHistoricalData(
    coinbaseService,
    "BTC-USD",
    olderStart,
    olderEnd
  );
  // Trim to shortest length
  const olderMinLength = Math.min(
    olderAdaData.prices.length,
    olderAdaData.volumes.length,
    olderBtcData.prices.length,
    olderBtcData.volumes.length
  );
  olderAdaData.prices = olderAdaData.prices.slice(0, olderMinLength);
  olderAdaData.volumes = olderAdaData.volumes.slice(0, olderMinLength);
  olderBtcData.prices = olderBtcData.prices.slice(0, olderMinLength);
  olderBtcData.volumes = olderBtcData.volumes.slice(0, olderMinLength);
  console.log(`Older Data Trimmed to ${olderMinLength} candles`);
  console.log(
    `Backtesting Older ${olderDays} Days (Days 1 to ${
      olderAdaData.prices.length - 30
    })...`
  );
  const olderResult = await backtester.backtest(
    olderAdaData,
    olderBtcData,
    30,
    olderAdaData.prices.length - 1
  );
  await backtester.evaluateBacktest(olderResult);

  // Full: January 2021 - April 2025
  const fullStart = olderStart;
  const fullAdaData = await fetchHistoricalData(
    coinbaseService,
    "ADA-USD",
    fullStart,
    recentEnd
  );
  const fullBtcData = await fetchHistoricalData(
    coinbaseService,
    "BTC-USD",
    fullStart,
    recentEnd
  );
  // Trim to shortest length
  const fullMinLength = Math.min(
    fullAdaData.prices.length,
    fullAdaData.volumes.length,
    fullBtcData.prices.length,
    fullBtcData.volumes.length
  );
  fullAdaData.prices = fullAdaData.prices.slice(0, fullMinLength);
  fullAdaData.volumes = fullAdaData.volumes.slice(0, fullMinLength);
  fullBtcData.prices = fullBtcData.prices.slice(0, fullMinLength);
  fullBtcData.volumes = fullBtcData.volumes.slice(0, fullMinLength);
  console.log(`Full Data Trimmed to ${fullMinLength} candles`);
  console.log(
    `Backtesting Full Period (Days 1 to ${fullAdaData.prices.length - 30})...`
  );
  const fullResult = await backtester.backtest(
    fullAdaData,
    fullBtcData,
    30,
    fullAdaData.prices.length - 1
  );
  await backtester.evaluateBacktest(fullResult);
}

runBacktest().catch((error) => {
  console.error("Backtest failed:", error);
  process.exit(1);
});
