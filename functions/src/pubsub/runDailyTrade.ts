import { logger } from "firebase-functions";
import { onSchedule, ScheduleOptions } from "firebase-functions/v2/scheduler";
import { TradeExecutor } from "../cardano/TradeExecutor";
import { TradingStrategy } from "../cardano/TradingStrategy";
import { sendSMS } from "../utils";
import { CoinbaseProductIds, Granularity } from "../types";
import { TIME_CONVERSIONS } from "../constants";

const strategy = new TradingStrategy();

const trader = new TradeExecutor({
  apiKey: process.env.COINBASE_API_KEY,
  apiSecret: process.env.COINBASE_API_SECRET,
});

const CONFIG: ScheduleOptions = {
  schedule: "every 24 hours",
  memory: "512MiB",
};

// Scheduled function to run every 24 hours
export const runDailyTrade = onSchedule(CONFIG, async () => {
  try {
    const now = Math.floor(Date.now() / 1000);
    const start = now - TIME_CONVERSIONS.TIMESTEP_IN_SECONDS;

    const adaData = await trader.getMarketData({
      product_id: CoinbaseProductIds.ADA,
      granularity: Granularity.OneDay,
      start: start.toString(),
      end: now.toString(),
    });

    const btcData = await trader.getMarketData({
      product_id: CoinbaseProductIds.BTC,
      granularity: Granularity.OneDay,
      start: start.toString(),
      end: now.toString(),
    });

    // Fetch current portfolio state (simplified; adjust based on your needs)
    const balances = await trader.getAccountBalances();
    const capital = parseFloat(balances.usd?.available_balance.value || "0");
    const holdings = parseFloat(balances.ada?.available_balance.value || "0");

    // Decide trade using existing strategy
    const { trade, confidence } = await strategy.decideTrade({
      adaPrices: adaData.prices,
      adaVolumes: adaData.volumes,
      btcPrices: btcData.prices,
      btcVolumes: btcData.volumes,
      capital,
      holdings,
      lastBuyPrice: undefined, // Persist this in Firestore if needed
      peakPrice: undefined,
      buyTimestamp: undefined,
      currentTimestamp: new Date().toISOString(),
    });

    if (trade) {
      sendSMS(JSON.stringify(trade));
      // // Execute the trade on Coinbase
      // const tradeResult = await trader.executeTrade(trade);
      // functions.logger.info("Trade executed:", {
      //   trade,
      //   confidence,
      //   tradeResult,
      // });
      // // Optionally store trade in Firestore
      // await admin
      //   .firestore()
      //   .collection("trades")
      //   .add({
      //     ...trade,
      //     confidence,
      //     executedAt: admin.firestore.FieldValue.serverTimestamp(),
      //     coinbaseResult: tradeResult,
      //   });
    } else {
      logger.info("No trade recommended", { confidence });
    }
  } catch (error) {
    logger.error("Trading error:", error);
    throw error;
  }
});
