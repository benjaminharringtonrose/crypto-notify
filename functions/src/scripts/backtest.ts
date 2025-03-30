import { CryptoCompareService } from "../api/CryptoCompareService";
import { TradeModelBacktester } from "../cardano/TradeModelBacktester";

async function runMultiPeriodBacktest() {
  const backtester = new TradeModelBacktester();
  const cryptoCompare = new CryptoCompareService();
  const results: { period: string; result: any }[] = [];

  const startYear = 2020;
  const endYear = 2023;
  const periodLengthMonths = 6;
  const stepMonths = 3;

  for (
    let year = startYear;
    year <= endYear - periodLengthMonths / 12;
    year++
  ) {
    for (let month = 0; month <= 12 - periodLengthMonths; month += stepMonths) {
      const startDate = new Date(year, month, 1);
      const endDate = new Date(year, month + periodLengthMonths, 0);
      if (endDate > new Date()) continue;

      console.log(
        `Running backtest for ${startDate.toISOString()} to ${endDate.toISOString()}`
      );
      const result = await backtester.backtest(startDate, endDate);

      const adaPrices = (
        await cryptoCompare.getHistoricalData(
          "ADA",
          Math.ceil(
            (endDate.getTime() - startDate.getTime()) / (1000 * 60 * 60 * 24)
          )
        )
      ).prices;
      const atr = calculateATR(adaPrices);
      const trend =
        (adaPrices[adaPrices.length - 1] - adaPrices[0]) / adaPrices[0];

      results.push({
        period: `${startDate.toISOString().slice(0, 10)} - ${endDate
          .toISOString()
          .slice(0, 10)}`,
        result: { ...result, atr, trend },
      });
    }
  }

  const summary = aggregateResults(results);
  console.log("Multi-Period Backtest Summary:", summary);
}

function calculateATR(prices: number[], period: number = 14): number {
  if (prices.length < period + 1) return 0.01;
  const trueRanges = [];
  for (let i = 1; i < prices.length; i++) {
    trueRanges.push(Math.abs(prices[i] - prices[i - 1]));
  }
  const recentRanges = trueRanges.slice(-period);
  return recentRanges.reduce((sum, range) => sum + range, 0) / period;
}

interface Metric {
  values: number[];
  avg: number;
  std: number;
}

function aggregateResults(results: { period: string; result: any }[]) {
  const metrics: { [key: string]: Metric } = {
    totalReturn: { values: [], avg: 0, std: 0 },
    winRate: { values: [], avg: 0, std: 0 },
    sharpeRatio: { values: [], avg: 0, std: 0 },
    maxDrawdown: { values: [], avg: 0, std: 0 },
  };

  results.forEach(({ result }) => {
    metrics.totalReturn.values.push(result.totalReturn);
    metrics.winRate.values.push(result.winRate);
    metrics.sharpeRatio.values.push(result.sharpeRatio);
    metrics.maxDrawdown.values.push(result.maxDrawdown);
  });

  Object.keys(metrics).forEach((key) => {
    const values = metrics[key].values;
    metrics[key].avg = values.reduce((a, b) => a + b, 0) / values.length;
    metrics[key].std = Math.sqrt(
      values.reduce((a, b) => a + Math.pow(b - metrics[key].avg, 2), 0) /
        values.length
    );
  });

  const correlations = {
    atrVsReturn: calculateCorrelation(
      results.map((r) => r.result.atr),
      results.map((r) => r.result.totalReturn)
    ),
    trendVsReturn: calculateCorrelation(
      results.map((r) => r.result.trend),
      results.map((r) => r.result.totalReturn)
    ),
  };

  return { metrics, correlations, periods: results };
}

function calculateCorrelation(x: number[], y: number[]): number {
  const n = x.length;
  const meanX = x.reduce((a, b) => a + b, 0) / n;
  const meanY = y.reduce((a, b) => a + b, 0) / n;
  const cov =
    x.reduce((sum, xi, i) => sum + (xi - meanX) * (y[i] - meanY), 0) / n;
  const stdX = Math.sqrt(
    x.reduce((sum, xi) => sum + Math.pow(xi - meanX, 2), 0) / n
  );
  const stdY = Math.sqrt(
    y.reduce((sum, yi) => sum + Math.pow(yi - meanY, 2), 0) / n
  );
  return cov / (stdX * stdY) || 0;
}

runMultiPeriodBacktest().catch((error) =>
  console.error("Backtest failed:", error)
);
