import { TradeModelBacktester } from "../bitcoin/TradeModelBacktester";
import { CoinbaseService } from "../api/CoinbaseService";
import { HistoricalData } from "../types";
import { TIME_CONVERSIONS } from "../constants";

// Interface for backtest results
interface BacktestPeriodResult {
  period: string;
  totalReturn: number;
  annualizedReturn: number;
  winRate: number;
  maxDrawdown: number;
  sharpeRatio: number;
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  averageHoldingDays: number;
  strategyDistribution: { [key: string]: number };
  confidenceDistribution: { [key: string]: number };
}

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

// Function to print comprehensive final summary
function printFinalSummary(results: BacktestPeriodResult[]) {
  console.log("\n" + "=".repeat(100));
  console.log("🎯 COMPREHENSIVE BACKTEST SUMMARY");
  console.log("=".repeat(100));

  // Print summary table
  console.log("\n📊 PERFORMANCE SUMMARY BY PERIOD:");
  console.log("─".repeat(100));
  console.log(
    "Period".padEnd(15) +
      "Return".padEnd(12) +
      "Ann. Return".padEnd(12) +
      "Win Rate".padEnd(10) +
      "Max DD".padEnd(8) +
      "Sharpe".padEnd(8) +
      "Trades".padEnd(8) +
      "Avg Hold".padEnd(10)
  );
  console.log("─".repeat(100));

  let totalCombinedReturn = 0;
  let totalTrades = 0;
  let totalWinningTrades = 0;
  let totalLosingTrades = 0;
  let totalDays = 0;

  results.forEach((result) => {
    console.log(
      result.period.padEnd(15) +
        `${(result.totalReturn * 100).toFixed(2)}%`.padEnd(12) +
        `${result.annualizedReturn.toFixed(2)}%`.padEnd(12) +
        `${result.winRate.toFixed(1)}%`.padEnd(10) +
        `${(result.maxDrawdown * 100).toFixed(2)}%`.padEnd(8) +
        `${result.sharpeRatio.toFixed(2)}`.padEnd(8) +
        `${result.totalTrades}`.padEnd(8) +
        `${result.averageHoldingDays.toFixed(1)}d`.padEnd(10)
    );

    totalCombinedReturn += result.totalReturn;
    totalTrades += result.totalTrades;
    totalWinningTrades += result.winningTrades;
    totalLosingTrades += result.losingTrades;
    totalDays += result.averageHoldingDays * result.totalTrades;
  });

  console.log("─".repeat(100));

  // Calculate aggregated statistics
  const overallWinRate =
    totalTrades > 0 ? (totalWinningTrades / totalTrades) * 100 : 0;
  const avgHoldingDays = totalTrades > 0 ? totalDays / totalTrades : 0;
  const avgReturn =
    results.length > 0 ? totalCombinedReturn / results.length : 0;

  console.log("\n📈 AGGREGATED STATISTICS:");
  console.log("─".repeat(50));
  console.log(
    `   Total Combined Return: ${(totalCombinedReturn * 100).toFixed(2)}%`
  );
  console.log(`   Average Return per Period: ${(avgReturn * 100).toFixed(2)}%`);
  console.log(`   Overall Win Rate: ${overallWinRate.toFixed(1)}%`);
  console.log(`   Total Trades: ${totalTrades}`);
  console.log(`   Winning Trades: ${totalWinningTrades}`);
  console.log(`   Losing Trades: ${totalLosingTrades}`);
  console.log(`   Average Holding Days: ${avgHoldingDays.toFixed(1)}`);

  // Strategy distribution analysis
  console.log("\n🎯 STRATEGY DISTRIBUTION ANALYSIS:");
  console.log("─".repeat(50));
  const strategyTotals: { [key: string]: number } = {};
  results.forEach((result) => {
    Object.entries(result.strategyDistribution).forEach(([strategy, count]) => {
      strategyTotals[strategy] = (strategyTotals[strategy] || 0) + count;
    });
  });

  Object.entries(strategyTotals).forEach(([strategy, count]) => {
    const percentage = totalTrades > 0 ? (count / totalTrades) * 100 : 0;
    console.log(
      `   ${strategy.padEnd(15)}: ${count} trades (${percentage.toFixed(1)}%)`
    );
  });

  // Confidence distribution analysis
  console.log("\n🎯 CONFIDENCE DISTRIBUTION ANALYSIS:");
  console.log("─".repeat(50));
  const confidenceTotals: { [key: string]: number } = {};
  results.forEach((result) => {
    Object.entries(result.confidenceDistribution).forEach(([range, count]) => {
      confidenceTotals[range] = (confidenceTotals[range] || 0) + count;
    });
  });

  Object.entries(confidenceTotals).forEach(([range, count]) => {
    const percentage = totalTrades > 0 ? (count / totalTrades) * 100 : 0;
    console.log(
      `   ${range.padEnd(10)}: ${count} trades (${percentage.toFixed(1)}%)`
    );
  });

  // Performance insights
  console.log("\n💡 PERFORMANCE INSIGHTS:");
  console.log("─".repeat(50));

  const bestPeriod = results.reduce((best, current) =>
    current.totalReturn > best.totalReturn ? current : best
  );
  const worstPeriod = results.reduce((worst, current) =>
    current.totalReturn < worst.totalReturn ? current : worst
  );

  console.log(
    `   Best Performing Period: ${bestPeriod.period} (${(
      bestPeriod.totalReturn * 100
    ).toFixed(2)}%)`
  );
  console.log(
    `   Worst Performing Period: ${worstPeriod.period} (${(
      worstPeriod.totalReturn * 100
    ).toFixed(2)}%)`
  );

  const profitablePeriods = results.filter((r) => r.totalReturn > 0).length;
  const totalPeriods = results.length;
  console.log(
    `   Profitable Periods: ${profitablePeriods}/${totalPeriods} (${(
      (profitablePeriods / totalPeriods) *
      100
    ).toFixed(1)}%)`
  );

  console.log("=".repeat(100));
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

  const results: BacktestPeriodResult[] = [];

  // Recent: April 2024 - April 2025
  const recentEnd = now;
  const recentStart = recentEnd - recentDays * oneDayMs;
  const recentBtcData = await fetchHistoricalData(
    coinbaseService,
    "BTC-USD",
    recentStart,
    recentEnd
  );
  console.log(`Recent Data: ${recentBtcData.prices.length} candles`);
  console.log(
    `Backtesting Recent ${recentDays} Days (Days 1 to ${
      recentBtcData.prices.length - 30
    })...`
  );
  const recentResult = await backtester.backtest(
    recentBtcData,
    30,
    recentBtcData.prices.length - 1
  );
  const recentEvaluation = await backtester.evaluateBacktest(recentResult);
  results.push({
    period: "Recent (500d)",
    ...recentEvaluation,
  });

  // Middle: November 2022 - April 2024
  const middleEnd = recentStart - oneDayMs;
  const middleStart = middleEnd - middleDays * oneDayMs;
  const middleBtcData = await fetchHistoricalData(
    coinbaseService,
    "BTC-USD",
    middleStart,
    middleEnd
  );
  console.log(`Middle Data: ${middleBtcData.prices.length} candles`);
  console.log(
    `Backtesting Middle ${middleDays} Days (Days 1 to ${
      middleBtcData.prices.length - 30
    })...`
  );
  const middleResult = await backtester.backtest(
    middleBtcData,
    30,
    middleBtcData.prices.length - 1
  );
  const middleEvaluation = await backtester.evaluateBacktest(middleResult);
  results.push({
    period: "Middle (500d)",
    ...middleEvaluation,
  });

  // Older: January 1, 2021 - October 28, 2021
  const olderEnd = new Date("2021-10-28").getTime();
  const olderStart = new Date("2021-01-01").getTime();
  const olderBtcData = await fetchHistoricalData(
    coinbaseService,
    "BTC-USD",
    olderStart,
    olderEnd
  );
  console.log(`Older Data: ${olderBtcData.prices.length} candles`);
  console.log(
    `Backtesting Older ${olderDays} Days (Days 1 to ${
      olderBtcData.prices.length - 30
    })...`
  );
  const olderResult = await backtester.backtest(
    olderBtcData,
    30,
    olderBtcData.prices.length - 1
  );
  const olderEvaluation = await backtester.evaluateBacktest(olderResult);
  results.push({
    period: "Older (300d)",
    ...olderEvaluation,
  });

  // Full: January 2021 - April 2025
  const fullStart = olderStart;
  const fullBtcData = await fetchHistoricalData(
    coinbaseService,
    "BTC-USD",
    fullStart,
    recentEnd
  );
  console.log(`Full Data: ${fullBtcData.prices.length} candles`);
  console.log(
    `Backtesting Full Period (Days 1 to ${fullBtcData.prices.length - 30})...`
  );
  const fullResult = await backtester.backtest(
    fullBtcData,
    30,
    fullBtcData.prices.length - 1
  );
  const fullEvaluation = await backtester.evaluateBacktest(fullResult);
  results.push({
    period: "Full Period",
    ...fullEvaluation,
  });

  // Print comprehensive final summary
  printFinalSummary(results);
}

runBacktest().catch((error) => {
  console.error("Backtest failed:", error);
  process.exit(1);
});
