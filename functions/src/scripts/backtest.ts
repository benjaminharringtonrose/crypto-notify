import { TradeModelBacktester } from "../cardano/TradeModelBacktester";

async function runMultiPeriodBacktest() {
  const backtester = new TradeModelBacktester();
  const results: { period: string; result: any }[] = [];

  const startYear = 2018;
  const endYear = 2024;
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
      results.push({
        period: `${startDate.toISOString().slice(0, 10)} - ${endDate
          .toISOString()
          .slice(0, 10)}`,
        result,
      });
    }
  }

  const summary = aggregateResults(results);
  console.log("Multi-Period Backtest Summary:", summary);
}

function aggregateResults(results: { period: string; result: any }[]) {
  const metrics: {
    [key: string]: { values: number[]; avg: number; std: number };
  } = {
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

  const atrValues = results.map((r) =>
    r.result.trades.length > 0
      ? r.result.trades[r.result.trades.length - 1].price * 0.01
      : 0
  ); // Proxy ATR
  const correlations = {
    atrVsReturn: calculateCorrelation(
      atrValues,
      results.map((r) => r.result.totalReturn)
    ),
    trendVsReturn: calculateCorrelation(
      results.map((r) =>
        r.result.trades.length > 0
          ? (r.result.trades[r.result.trades.length - 1].price -
              r.result.trades[0].price) /
            r.result.trades[0].price
          : 0
      ),
      results.map((r) => r.result.totalReturn)
    ),
  };

  return { metrics, correlations, periods: results };
}

function calculateCorrelation(x: number[], y: number[]): number {
  const n = x.length;
  if (n !== y.length || n < 2) return 0;
  const meanX = x.reduce((a, b) => a + b, 0) / n;
  const meanY = y.reduce((a, b) => a + b, 0) / n;
  const cov =
    x.reduce((sum, xi, i) => sum + (xi - meanX) * (y[i] - meanY), 0) / (n - 1);
  const stdX = Math.sqrt(
    x.reduce((sum, xi) => sum + Math.pow(xi - meanX, 2), 0) / (n - 1)
  );
  const stdY = Math.sqrt(
    y.reduce((sum, yi) => sum + Math.pow(yi - meanY, 2), 0) / (n - 1)
  );
  return stdX === 0 || stdY === 0 ? 0 : cov / (stdX * stdY);
}

runMultiPeriodBacktest().catch((error) =>
  console.error("Backtest failed:", error)
);
