import { TradeModelBacktester } from "../bitcoin/ml/TradeModelBacktester";
import { CoinbaseService } from "../api/CoinbaseService";
import { HistoricalData } from "../types";
import { TIME_CONVERSIONS, STRATEGY_CONFIG } from "../constants";

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

// Enhanced function to print comprehensive final summary
function printFinalSummary(results: BacktestPeriodResult[]) {
  console.log("\n" + "=".repeat(120));
  console.log("🎯 ENHANCED BACKTEST SUMMARY - MODEL IMPROVEMENTS EVALUATION");
  console.log("=".repeat(120));

  // Print enhanced summary table
  console.log("\n📊 PERFORMANCE SUMMARY BY PERIOD:");
  console.log("─".repeat(120));
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
  console.log("─".repeat(120));

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

  console.log("─".repeat(120));

  // Calculate aggregated statistics
  const overallWinRate =
    totalTrades > 0 ? (totalWinningTrades / totalTrades) * 100 : 0;
  const avgHoldingDays = totalTrades > 0 ? totalDays / totalTrades : 0;
  const avgReturn =
    results.length > 0 ? totalCombinedReturn / results.length : 0;

  console.log("\n📈 AGGREGATED STATISTICS:");
  console.log("─".repeat(60));
  console.log(
    `   Total Combined Return: ${(totalCombinedReturn * 100).toFixed(2)}%`
  );
  console.log(`   Average Return per Period: ${(avgReturn * 100).toFixed(2)}%`);
  console.log(`   Overall Win Rate: ${overallWinRate.toFixed(1)}%`);
  console.log(`   Total Trades: ${totalTrades}`);
  console.log(`   Winning Trades: ${totalWinningTrades}`);
  console.log(`   Losing Trades: ${totalLosingTrades}`);
  console.log(`   Average Holding Days: ${avgHoldingDays.toFixed(1)}`);

  // Enhanced strategy distribution analysis
  console.log("\n🎯 STRATEGY DISTRIBUTION ANALYSIS:");
  console.log("─".repeat(60));
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

  // Enhanced confidence distribution analysis
  console.log("\n🎯 CONFIDENCE DISTRIBUTION ANALYSIS:");
  console.log("─".repeat(60));
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

  // Performance insights and recommendations
  console.log("\n💡 PERFORMANCE INSIGHTS & RECOMMENDATIONS:");
  console.log("─".repeat(60));

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

  // Model improvement recommendations
  console.log("\n🔧 MODEL IMPROVEMENT RECOMMENDATIONS:");
  console.log("─".repeat(60));

  if (overallWinRate < 65) {
    console.log("   ⚠️  Win rate below target (65%+). Consider:");
    console.log("      - Increasing confidence thresholds");
    console.log("      - Improving feature engineering");
    console.log("      - Adjusting strategy parameters");
  } else {
    console.log("   ✅ Win rate meets target requirements");
  }

  if (avgReturn < 0.15) {
    console.log("   ⚠️  Average return below target (15%+). Consider:");
    console.log("      - More aggressive position sizing");
    console.log("      - Better trend following strategies");
    console.log("      - Enhanced market regime detection");
  } else {
    console.log("   ✅ Average return meets target requirements");
  }

  // Configuration summary
  console.log("\n⚙️  CURRENT CONFIGURATION SUMMARY:");
  console.log("─".repeat(60));
  console.log(`   Min Confidence: ${STRATEGY_CONFIG.MIN_CONFIDENCE_DEFAULT}`);
  console.log(
    `   Stop Loss Multiplier: ${STRATEGY_CONFIG.STOP_LOSS_MULTIPLIER_DEFAULT}`
  );
  console.log(
    `   Profit Take Multiplier: ${STRATEGY_CONFIG.PROFIT_TAKE_MULTIPLIER_DEFAULT}`
  );
  console.log(
    `   Base Position Size: ${STRATEGY_CONFIG.BASE_POSITION_SIZE_DEFAULT}`
  );
  console.log(`   Min Hold Days: ${STRATEGY_CONFIG.MIN_HOLD_DAYS_DEFAULT}`);

  console.log("=".repeat(120));
}

async function runBacktest() {
  console.log("🚀 Starting Enhanced BTC Model Backtest");
  console.log("=".repeat(60));

  const coinbaseService = new CoinbaseService({
    apiKey: process.env.COINBASE_API_KEY,
    apiSecret: process.env.COINBASE_API_SECRET,
  });
  const backtester = new TradeModelBacktester(10000);
  const results: BacktestPeriodResult[] = [];

  const recentDays = 500;
  const middleDays = 500;
  const now = Date.now();
  const oneDayMs = 24 * 60 * 60 * 1000;

  try {
    // Recent: April 2024 - April 2025
    console.log("\n📊 Testing Recent Period (500 days)...");
    const recentEnd = now;
    const recentStart = recentEnd - recentDays * oneDayMs;

    const recentBtcData = await fetchHistoricalData(
      coinbaseService,
      "BTC-USD",
      recentStart,
      recentEnd
    );
    console.log(`   Data: ${recentBtcData.prices.length} candles`);

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
    console.log("\n📊 Testing Middle Period (500 days)...");
    const middleEnd = recentStart - oneDayMs;
    const middleStart = middleEnd - middleDays * oneDayMs;
    const middleBtcData = await fetchHistoricalData(
      coinbaseService,
      "BTC-USD",
      middleStart,
      middleEnd
    );
    console.log(`   Data: ${middleBtcData.prices.length} candles`);

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
    console.log("\n📊 Testing Older Period (300 days)...");
    const olderEnd = new Date("2021-10-28").getTime();
    const olderStart = new Date("2021-01-01").getTime();
    const olderBtcData = await fetchHistoricalData(
      coinbaseService,
      "BTC-USD",
      olderStart,
      olderEnd
    );
    console.log(`   Data: ${olderBtcData.prices.length} candles`);

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
    console.log("\n📊 Testing Full Period...");
    const fullStart = olderStart;
    const fullBtcData = await fetchHistoricalData(
      coinbaseService,
      "BTC-USD",
      fullStart,
      recentEnd
    );
    console.log(`   Data: ${fullBtcData.prices.length} candles`);

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
  } catch (error) {
    console.error("\n❌ Backtest failed:", error);
    console.error(
      "Stack trace:",
      error instanceof Error ? error.stack : "No stack trace"
    );
    process.exit(1);
  }
}

runBacktest().catch((error) => {
  console.error("Backtest failed:", error);
  process.exit(1);
});
