import { TradeModelBacktester } from "../cardano/TradeModelBacktester";
import { CryptoCompareService } from "../api/CryptoCompareService";
import { MODEL_CONFIG } from "../constants";

async function runBacktest() {
  const cryptoCompare = new CryptoCompareService();
  const startDaysAgo = 1000;
  const [adaData, btcData] = await Promise.all([
    cryptoCompare.getHistoricalData("ADA", startDaysAgo),
    cryptoCompare.getHistoricalData("BTC", startDaysAgo),
  ]);

  const backtester = new TradeModelBacktester(10000);
  const startIndex = MODEL_CONFIG.TIMESTEPS; // Start after enough data for sequences
  const endIndex = adaData.prices.length - 1; // End at latest data

  const result = await backtester.backtest(
    adaData,
    btcData,
    startIndex,
    endIndex
  );
  await backtester.evaluateBacktest(result);

  // Optionally, save or visualize equityCurve
  console.log("Equity Curve:", result.equityCurve);
}

runBacktest().catch(console.error);
