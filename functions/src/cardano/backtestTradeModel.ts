import * as admin from "firebase-admin";
import { determineTrade } from "./determineTrade";
import { getHistoricalData } from "../api/getHistoricalData";
import { FIVE_YEARS_IN_DAYS } from "../constants";
import { Recommendation, TradeDecision } from "../types";
import serviceAccount from "../../../serviceAccount.json";

admin.initializeApp({
  credential: admin.credential.cert(serviceAccount as any),
  storageBucket: process.env.STORAGE_BUCKET || "your-bucket-name",
});

interface Trade {
  type: "buy" | "sell";
  price: number;
  timestamp: string;
  adaAmount: number;
  usdValue: number;
  buyPrice?: number;
}

interface BacktestResult {
  totalReturn: number;
  totalTrades: number;
  winRate: number;
  portfolioHistory: { timestamp: string; value: number }[];
  trades: Trade[];
}

const TRANSACTION_FEE = 0.002;
const INITIAL_USD = 1000;
const BUY_POSITION_SIZE = 0.5;
const SELL_POSITION_SIZE = 0.5;

export const backtestTradeModel = async (
  startDaysAgo: number = 91,
  endDaysAgo: number = 1,
  stepDays: number = 3
): Promise<BacktestResult> => {
  console.log("Starting backtest...");

  const { prices: adaPrices, volumes: adaVolumes } = await getHistoricalData(
    "ADA",
    Math.min(startDaysAgo, FIVE_YEARS_IN_DAYS)
  );
  const { prices: btcPrices, volumes: btcVolumes } = await getHistoricalData(
    "BTC",
    Math.min(startDaysAgo, FIVE_YEARS_IN_DAYS)
  );

  const startIndex = Math.max(0, adaPrices.length - startDaysAgo);
  const endIndex = Math.max(0, adaPrices.length - endDaysAgo);
  const slicedAdaPrices = adaPrices.slice(startIndex, endIndex);
  const slicedAdaVolumes = adaVolumes.slice(startIndex, endIndex);
  const slicedBtcPrices = btcPrices.slice(startIndex, endIndex);
  const slicedBtcVolumes = btcVolumes.slice(startIndex, endIndex);

  console.log(
    `ADA data points: ${slicedAdaPrices.length}, BTC data points: ${slicedBtcPrices.length}`
  );

  let usdBalance = INITIAL_USD;
  let adaBalance = 0;
  let avgBuyPrice = 0;
  let totalAdaBought = 0;
  const trades: Trade[] = [];
  const portfolioHistory: { timestamp: string; value: number }[] = [];
  let wins = 0;
  let completedCycles = 0;

  const initialPrice = slicedAdaPrices[0];
  const initialTimestamp = new Date(
    Date.now() - (adaPrices.length - startIndex) * 24 * 60 * 60 * 1000
  ).toISOString();
  const initialUsdToSpend =
    usdBalance * BUY_POSITION_SIZE * (1 - TRANSACTION_FEE);
  adaBalance = initialUsdToSpend / initialPrice;
  totalAdaBought += adaBalance;
  avgBuyPrice = initialPrice;
  usdBalance -= initialUsdToSpend / (1 - TRANSACTION_FEE);
  trades.push({
    type: "buy",
    price: initialPrice,
    timestamp: initialTimestamp,
    adaAmount: adaBalance,
    usdValue: initialUsdToSpend,
  });
  console.log(
    `Initial Buy at $${initialPrice.toFixed(4)}, ADA: ${adaBalance.toFixed(2)}`
  );
  portfolioHistory.push({
    timestamp: initialTimestamp,
    value: usdBalance + adaBalance * initialPrice,
  });

  for (let i = 3; i < slicedAdaPrices.length - 5; i += stepDays) {
    const currentAdaPrice = slicedAdaPrices[i];
    const timestamp = new Date(
      Date.now() - (adaPrices.length - startIndex - i) * 24 * 60 * 60 * 1000
    ).toISOString();

    const historicalAdaPrices = slicedAdaPrices.slice(0, i + 1);
    const historicalAdaVolumes = slicedAdaVolumes.slice(0, i + 1);
    const historicalBtcPrices = slicedBtcPrices.slice(0, i + 1);
    const historicalBtcVolumes = slicedBtcVolumes.slice(0, i + 1);

    const originalGetHistorical =
      require("../api/getHistoricalPricesAndVolumes").getHistoricalPricesAndVolumes;
    const originalGetCurrent =
      require("../api/getCurrentPrices").getCurrentPrices;

    require("../api/getHistoricalPricesAndVolumes").getHistoricalPricesAndVolumes =
      async () => ({
        adaPrices: historicalAdaPrices,
        adaVolumes: historicalAdaVolumes,
        btcPrices: historicalBtcPrices,
        btcVolumes: historicalBtcVolumes,
      });
    require("../api/getCurrentPrices").getCurrentPrices = async () => ({
      currentAdaPrice,
      currentBtcPrice: btcPrices[i],
    });

    const decision: TradeDecision = await determineTrade(); // No avgBuyPrice passed
    console.log(
      `Day ${i - 34}: Price $${currentAdaPrice.toFixed(4)}, ` +
        `Probabilities - Buy: ${decision.probabilities.buy.toFixed(3)}, ` +
        `Sell: ${decision.probabilities.sell.toFixed(3)}, ` +
        `Hold: ${decision.probabilities.hold.toFixed(3)}, ` +
        `Recommendation: ${decision.recommendation}`
    );

    if (
      trades.length > 0 &&
      (new Date(timestamp).getTime() -
        new Date(trades[trades.length - 1].timestamp).getTime()) /
        (1000 * 60 * 60 * 24) <
        3
    ) {
      console.log("Holding - less than 3 days since last trade");
      portfolioHistory.push({
        timestamp,
        value: usdBalance + adaBalance * currentAdaPrice,
      });
      continue;
    }

    if (decision.recommendation === Recommendation.Buy && usdBalance > 0) {
      const usdToSpend = usdBalance * BUY_POSITION_SIZE * (1 - TRANSACTION_FEE);
      const adaBought = usdToSpend / currentAdaPrice;
      adaBalance += adaBought;
      totalAdaBought += adaBought;
      avgBuyPrice =
        (avgBuyPrice * (totalAdaBought - adaBought) +
          currentAdaPrice * adaBought) /
        totalAdaBought;
      usdBalance -= usdToSpend / (1 - TRANSACTION_FEE);
      trades.push({
        type: "buy",
        price: currentAdaPrice,
        timestamp,
        adaAmount: adaBought,
        usdValue: usdToSpend,
      });
      console.log(
        `Buy at $${currentAdaPrice.toFixed(4)}, ADA: ${adaBought.toFixed(2)}`
      );
    } else if (
      decision.recommendation === Recommendation.Sell &&
      adaBalance > 0
    ) {
      const adaToSell = adaBalance * SELL_POSITION_SIZE;
      const usdReceived = adaToSell * currentAdaPrice * (1 - TRANSACTION_FEE);
      adaBalance -= adaToSell;
      usdBalance += usdReceived;
      trades.push({
        type: "sell",
        price: currentAdaPrice,
        timestamp,
        adaAmount: adaToSell,
        usdValue: usdReceived,
        buyPrice: avgBuyPrice,
      });
      console.log(
        `Sell at $${currentAdaPrice.toFixed(4)}, USD: ${usdReceived.toFixed(2)}`
      );
      if (usdReceived > adaToSell * avgBuyPrice) wins++;
      completedCycles++;
    }

    const portfolioValue = usdBalance + adaBalance * currentAdaPrice;
    portfolioHistory.push({ timestamp, value: portfolioValue });

    require("../api/getHistoricalPricesAndVolumes").getHistoricalPricesAndVolumes =
      originalGetHistorical;
    require("../api/getCurrentPrices").getCurrentPrices = originalGetCurrent;
  }

  const finalValue =
    usdBalance + adaBalance * slicedAdaPrices[slicedAdaPrices.length - 6];
  const totalReturn = ((finalValue - INITIAL_USD) / INITIAL_USD) * 100;
  const totalTrades = trades.length;
  const winRate = completedCycles > 0 ? (wins / completedCycles) * 100 : 0;

  console.log(`Backtest completed:
    Initial USD: $${INITIAL_USD}
    Final Value: $${finalValue.toFixed(2)}
    Total Return: ${totalReturn.toFixed(2)}%
    Total Trades: ${totalTrades}
    Win Rate: ${winRate.toFixed(2)}%`);

  return { totalReturn, totalTrades, winRate, portfolioHistory, trades };
};

backtestTradeModel()
  .then((result) => console.log("Backtest Result:", result))
  .catch((error) => console.error("Backtest Error:", error));
