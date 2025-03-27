import { TradeModelBacktester } from "../cardano/TradeModelBacktester";

async function runBacktest() {
  const backtester = new TradeModelBacktester(
    10000, // initialCapital
    0.1, // positionSize
    0.001, // slippage
    0.1, // commission
    0.7, // buyThreshold
    0.6 // sellThreshold
  );

  const startDate = new Date("2023-01-01");
  const endDate = new Date("2023-12-31");

  try {
    const result = await backtester.backtest(startDate, endDate);
    console.log("Backtest Results:");
    console.log(`Total Return: ${(result.totalReturn * 100).toFixed(2)}%`);
    console.log(`Total Trades: ${result.totalTrades}`);
    console.log(`Win Rate: ${(result.winRate * 100).toFixed(2)}%`);
    console.log(`Sharpe Ratio: ${result.sharpeRatio.toFixed(2)}`);
    console.log(`Max Drawdown: ${(result.maxDrawdown * 100).toFixed(2)}%`);
    console.log("Trades:", result.trades);
  } catch (error) {
    console.error("Backtest failed:", error);
  }
}

setTimeout(() => {
  runBacktest();
}, 5000);
