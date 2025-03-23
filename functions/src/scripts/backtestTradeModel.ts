import { TradeModelBacktester } from "../cardano/TradeModelBacktester";

TradeModelBacktester.runWithTimeout()
  .then((result) => console.log("Backtest Result:", result))
  .catch((error) => console.error("Backtest Error:", error));
