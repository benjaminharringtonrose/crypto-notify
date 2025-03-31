import { onSchedule, ScheduleOptions } from "firebase-functions/v2/scheduler";
import {
  CoinbaseProductIds,
  Collections,
  CryptoIds,
  Docs,
  Granularity,
  Recommendation,
  TradeRecommendation,
} from "../types";
import { sendSMS, formatAnalysisResults } from "../utils";
import { getFirestore } from "firebase-admin/firestore";
import { TradingStrategy } from "../cardano/TradingStrategy";
import { TradeExecutor } from "../cardano/TradeExecutor";
import { TIME_CONVERSIONS } from "../constants";

const strategy = new TradingStrategy();

const trader = new TradeExecutor({
  apiKey: process.env.COINBASE_API_KEY,
  apiSecret: process.env.COINBASE_API_SECRET,
});

const CONFIG: ScheduleOptions = {
  schedule: "*/10 * * * *",
  memory: "512MiB",
};

export const runTradeModelADA = onSchedule(CONFIG, async () => {
  try {
    const now = Math.floor(
      Date.now() / TIME_CONVERSIONS.ONE_SECOND_IN_MILLISECONDS
    );
    const start = now - TIME_CONVERSIONS.TIMESTEP_IN_SECONDS;

    const currentPrice = await trader.getCurrentPrice(CoinbaseProductIds.ADA);

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

    if (
      adaData.prices.length < TIME_CONVERSIONS.ONE_MONTH_IN_DAYS ||
      btcData.prices.length < TIME_CONVERSIONS.ONE_MONTH_IN_DAYS
    ) {
      throw new Error("Insufficient historical data for prediction");
    }

    const { trade, buyProb, sellProb } = await strategy.decideTrade({
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

    const db = getFirestore();
    const recommendationRef = db
      .collection(Collections.TradeRecommendations)
      .doc(Docs.Cardano);

    const previousDoc = await recommendationRef.get();
    const previous = (previousDoc.exists ? previousDoc.data() : undefined) as
      | TradeRecommendation
      | undefined;

    const analysisResults = formatAnalysisResults({
      cryptoSymbol: CryptoIds.Cardano,
      currentPrice,
      probabilities: {
        buy: buyProb,
        sell: sellProb,
        hold: 1 - Math.max(buyProb, sellProb),
      },
      recommendation: trade?.type ?? Recommendation.Hold,
    });

    console.log(analysisResults);

    const sameRecommendation = trade?.type === previous?.recommendation;

    if (!previous) {
      switch (trade?.type) {
        case Recommendation.Buy:
          await sendSMS(analysisResults);
          await recommendationRef.set({
            recommendation: trade.type,
            probability: buyProb,
            timestamp: new Date().toISOString(),
          });
          return;
        case Recommendation.Sell:
          await sendSMS(analysisResults);
          await recommendationRef.set({
            recommendation: trade.type,
            probability: sellProb,
            timestamp: new Date().toISOString(),
          });
          return;
        case Recommendation.Hold:
          await recommendationRef.set({
            recommendation: trade.type,
            probability: 1 - Math.max(buyProb, sellProb),
            timestamp: new Date().toISOString(),
          });
          return;
        default:
          console.log("Recommendation unexpected");
      }
    }

    if (!sameRecommendation) {
      switch (trade?.type) {
        case Recommendation.Buy:
          await sendSMS(analysisResults);
          await recommendationRef.set({
            recommendation: trade.type,
            probability: buyProb,
            timestamp: new Date().toISOString(),
          });
          return;
        case Recommendation.Sell:
          await sendSMS(analysisResults);
          await recommendationRef.set({
            recommendation: trade.type,
            probability: sellProb,
            timestamp: new Date().toISOString(),
          });
          return;
        case Recommendation.Hold:
          return;
        default:
          console.log("Recommendation unexpected");
      }
    }

    if (sameRecommendation) {
      switch (trade?.type) {
        case Recommendation.Buy:
          if (previous?.probability && buyProb > previous.probability) {
            await sendSMS(`Buy probability increased: ${buyProb}`);
            await recommendationRef.set({
              recommendation: trade.type,
              probability: buyProb,
              timestamp: new Date().toISOString(),
            });
          }
          return;
        case Recommendation.Sell:
          if (previous?.probability && sellProb > previous.probability) {
            await sendSMS(`Sell probability increased: ${sellProb}`);
            await recommendationRef.set({
              recommendation: trade.type,
              probability: sellProb,
              timestamp: new Date().toISOString(),
            });
          }
          return;
        case Recommendation.Hold:
          return;
        default:
          console.log("Recommendation unexpected");
      }
    }
  } catch (error) {
    console.log("runTradeModel Error:", JSON.stringify(error));
  }
});
