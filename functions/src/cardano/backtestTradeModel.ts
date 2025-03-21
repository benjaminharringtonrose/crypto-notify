import * as admin from "firebase-admin";
import { determineTrade } from "./determineTrade";
import { getHistoricalData } from "../api/getHistoricalData";
import { FIVE_YEARS_IN_DAYS } from "../constants";
import { TradeDecision, Recommendation } from "../types";
import serviceAccount from "../../../serviceAccount.json";

admin.initializeApp({
  credential: admin.credential.cert(serviceAccount as any),
  storageBucket:
    process.env.STORAGE_BUCKET || "crypto-notify-ee5bc.firebasestorage.app",
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
const MIN_TRADE_USD = 100;
const MIN_HOLD_DAYS = 1;
const CASH_RESERVE = 100;
const STOP_LOSS_THRESHOLD = -0.1; // -10%

export const backtestTradeModel = async (
  startDaysAgo: number = 91,
  endDaysAgo: number = 1,
  stepDays: number = 1
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

  console.log(
    `ADA data points: ${adaPrices.length}, BTC data points: ${btcPrices.length}`
  );

  const startIndex = Math.max(0, adaPrices.length - startDaysAgo);
  const endIndex = Math.max(0, adaPrices.length - endDaysAgo);
  const slicedAdaPrices = adaPrices.slice(startIndex, endIndex);
  const slicedAdaVolumes = adaVolumes.slice(startIndex, endIndex);
  const slicedBtcPrices = btcPrices.slice(startIndex, endIndex);
  const slicedBtcVolumes = btcVolumes.slice(startIndex, endIndex);

  let usdBalance = INITIAL_USD;
  let adaBalance = 0;
  let avgBuyPrice = 0;
  const trades: Trade[] = [];
  const portfolioHistory: { timestamp: string; value: number }[] = [];
  let wins = 0;
  let completedCycles = 0;
  let daysSinceLastTrade = 0;

  for (let i = 14; i < slicedAdaPrices.length - 5; i += stepDays) {
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

    const decision: TradeDecision = await determineTrade();

    console.log(
      `Day ${i - 13}: Price ${currentAdaPrice.toFixed(
        4
      )}, Probabilities - Buy: ${decision.probabilities.buy.toFixed(
        3
      )}, Sell: ${decision.probabilities.sell.toFixed(
        3
      )}, Hold: ${decision.probabilities.hold.toFixed(3)}, Recommendation: ${
        decision.recommendation
      }`
    );

    const confidence = Math.max(
      decision.probabilities.buy,
      decision.probabilities.sell
    );
    const positionSizeFactor = Math.min(0.75, confidence);
    const portfolioValue = usdBalance + adaBalance * currentAdaPrice;
    const unrealizedProfit =
      adaBalance > 0 ? (currentAdaPrice - avgBuyPrice) / avgBuyPrice : 0;

    const buyCondition =
      decision.recommendation === Recommendation.Buy &&
      usdBalance > CASH_RESERVE + MIN_TRADE_USD &&
      daysSinceLastTrade >= MIN_HOLD_DAYS;
    const sellCondition =
      (decision.recommendation === Recommendation.Sell &&
        adaBalance > 0 &&
        daysSinceLastTrade >= MIN_HOLD_DAYS) ||
      (adaBalance > 0 && unrealizedProfit <= STOP_LOSS_THRESHOLD);

    if (buyCondition) {
      const usdToSpend = Math.max(
        MIN_TRADE_USD,
        usdBalance * positionSizeFactor * (1 - TRANSACTION_FEE)
      );
      if (usdBalance - usdToSpend < CASH_RESERVE) continue;
      const adaBought = usdToSpend / currentAdaPrice;
      adaBalance += adaBought;
      avgBuyPrice =
        adaBalance > 0
          ? (avgBuyPrice * (adaBalance - adaBought) +
              currentAdaPrice * adaBought) /
            adaBalance
          : currentAdaPrice;
      usdBalance -= usdToSpend / (1 - TRANSACTION_FEE);
      trades.push({
        type: "buy",
        price: currentAdaPrice,
        timestamp,
        adaAmount: adaBought,
        usdValue: usdToSpend,
      });
      console.log(
        `Buy at $${currentAdaPrice.toFixed(4)}, ADA: ${adaBought.toFixed(
          2
        )}, USD: ${usdToSpend.toFixed(2)}`
      );
      daysSinceLastTrade = 0;
    } else if (sellCondition) {
      const adaToSell = adaBalance;
      const usdReceived = adaToSell * currentAdaPrice * (1 - TRANSACTION_FEE);
      usdBalance += usdReceived;
      adaBalance -= adaToSell;
      trades.push({
        type: "sell",
        price: currentAdaPrice,
        timestamp,
        adaAmount: adaToSell,
        usdValue: usdReceived,
        buyPrice: avgBuyPrice,
      });
      console.log(
        `Sell at $${currentAdaPrice.toFixed(4)}, ADA: ${adaToSell.toFixed(
          2
        )}, USD: ${usdReceived.toFixed(2)}`
      );
      if (usdReceived > adaToSell * avgBuyPrice) wins++;
      if (adaBalance === 0) {
        completedCycles++;
        avgBuyPrice = 0;
      }
      daysSinceLastTrade = 0;
    } else {
      daysSinceLastTrade++;
    }

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

setTimeout(() => {
  backtestTradeModel()
    .then((result) => console.log("Backtest Result:", result))
    .catch((error) => console.error("Backtest Error:", error));
}, 2000);
