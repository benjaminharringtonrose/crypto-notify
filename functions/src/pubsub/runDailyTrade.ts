import * as functions from "firebase-functions";
import { onSchedule } from "firebase-functions/v2/scheduler";
import { CoinbaseTradeExecutor } from "../cardano/CoinbaseTradeExecutor";
import { TradingStrategy } from "../cardano/TradingStrategy";
import { sendSMS } from "../utils";

const strategy = new TradingStrategy();

// Scheduled function to run every 24 hours
export const executeDailyTrade = onSchedule("every 24 hours", async () => {
  const config = {
    apiKey: process.env.COINBASE_API_KEY ?? "",
    apiSecret: process.env.COINBASE_API_SECRET ?? "",
  };

  const trader = new CoinbaseTradeExecutor(config);

  try {
    // Fetch live market data
    const adaData = await trader.getMarketData("ADA-USD");
    const btcData = await trader.getMarketData("BTC-USD");

    // Fetch current portfolio state (simplified; adjust based on your needs)
    const balances = await trader.getAccountBalance();
    const usdAccount = balances.accounts.find(
      (acc: any) => acc.currency === "USD"
    );
    const adaAccount = balances.accounts.find(
      (acc: any) => acc.currency === "ADA"
    );
    const capital = parseFloat(usdAccount?.available || "0");
    const holdings = parseFloat(adaAccount?.available || "0");

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
      functions.logger.info("No trade recommended", { confidence });
    }
  } catch (error) {
    functions.logger.error("Trading error:", error);
    throw error;
  }
});

// HTTP-triggered function for manual execution (optional)
export const executeTradeNow = functions.https.onRequest(async (req, res) => {
  const config = {
    apiKey: process.env.COINBASE_API_KEY ?? "",
    apiSecret: process.env.COINBASE_API_SECRET ?? "",
  };

  const trader = new CoinbaseTradeExecutor(config);

  try {
    const adaData = await trader.getMarketData("ADA-USD");
    const btcData = await trader.getMarketData("BTC-USD");
    const balances = await trader.getAccountBalance();
    const usdAccount = balances.accounts.find(
      (acc: any) => acc.currency === "USD"
    );
    const adaAccount = balances.accounts.find(
      (acc: any) => acc.currency === "ADA"
    );
    const capital = parseFloat(usdAccount?.available || "0");
    const holdings = parseFloat(adaAccount?.available || "0");

    const { trade, confidence } = await strategy.decideTrade({
      adaPrices: adaData.prices,
      adaVolumes: adaData.volumes,
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
    functions.logger.error("Trade execution failed:", error);
    res.status(500).json({ error: error.message });
  }
});
