import { https, logger } from "firebase-functions";
import { HttpsOptions } from "firebase-functions/https";
import { CoinbaseProductIds, Granularity } from "../types";
import { sendSMS } from "../utils";
import { TradingStrategy } from "../bitcoin/TradingStrategy";
import { TradeExecutor } from "../bitcoin/TradeExecutor";
import { TIME_CONVERSIONS } from "../constants";

const NOW_CONFIG: HttpsOptions = {
  memory: "512MiB",
};

export const runTradeNow = https.onRequest(NOW_CONFIG, async (_, res) => {
  try {
    const strategy = new TradingStrategy();

    const trader = new TradeExecutor({
      apiKey: process.env.COINBASE_API_KEY,
      apiSecret: process.env.COINBASE_API_SECRET,
    });

    const now = Math.floor(Date.now() / 1000);
    const start = now - TIME_CONVERSIONS.TIMESTEP_IN_SECONDS;

    const btcData = await trader.getMarketData({
      product_id: CoinbaseProductIds.BTC,
      granularity: Granularity.OneDay,
      start: start.toString(),
      end: now.toString(),
    });

    const balances = await trader.getAccountBalances();
    const capital = parseFloat(balances.usd?.available_balance.value || "0");
    const holdings = parseFloat(balances.btc?.available_balance.value || "0");

    const { trade, confidence } = await strategy.decideTrade({
      btcPrices: btcData.prices,
      btcVolumes: btcData.volumes,
      capital,
      holdings,
      lastBuyPrice: undefined,
      peakPrice: undefined,
      buyTimestamp: undefined,
      currentTimestamp: new Date().toISOString(),
    });

    if (trade) {
      sendSMS(JSON.stringify(trade));
      // const tradeResult = await trader.executeTrade(trade);
      // res.status(200).json({ trade, confidence, tradeResult });
    } else {
      res.status(200).json({ message: "No trade recommended", confidence });
    }
  } catch (error: any) {
    logger.error("Trade execution failed:", error);
    res.status(500).json({ error: error.message });
  }
});
