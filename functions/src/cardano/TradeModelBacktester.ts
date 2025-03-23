import * as admin from "firebase-admin";
import { getHistoricalData } from "../api/getHistoricalData";
import { FIVE_YEARS_IN_DAYS } from "../constants";
import { Recommendation } from "../types";
import serviceAccount from "../../../serviceAccount.json";
import TradePredictor from "./TradeModelPredictor";

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
  sharpeRatio: number;
  maxDrawdown: number;
  portfolioHistory: { timestamp: string; value: number }[];
  trades: Trade[];
}

export class TradeModelBacktester {
  private TRANSACTION_FEE = 0.002;
  private INITIAL_USD = 1000;
  private MIN_TRADE_USD = 100;
  private MIN_HOLD_DAYS = 1;
  private CASH_RESERVE = 100;
  private STOP_LOSS_THRESHOLD = -0.05;
  private TAKE_PROFIT_THRESHOLD = 0.1;

  private startDaysAgo: number;
  private endDaysAgo: number;
  private stepDays: number;

  constructor(
    startDaysAgo: number = 91,
    endDaysAgo: number = 1,
    stepDays: number = 1
  ) {
    this.startDaysAgo = startDaysAgo;
    this.endDaysAgo = endDaysAgo;
    this.stepDays = stepDays;

    if (!admin.apps.length) {
      admin.initializeApp({
        credential: admin.credential.cert(serviceAccount as any),
        storageBucket:
          process.env.STORAGE_BUCKET ||
          "crypto-notify-ee5bc.firebasestorage.app",
      });
    }
  }

  private async fetchHistoricalData(): Promise<{
    adaPrices: number[];
    adaVolumes: number[];
    btcPrices: number[];
    btcVolumes: number[];
  }> {
    const { prices: adaPrices, volumes: adaVolumes } = await getHistoricalData(
      "ADA",
      Math.min(this.startDaysAgo, FIVE_YEARS_IN_DAYS)
    );
    const { prices: btcPrices, volumes: btcVolumes } = await getHistoricalData(
      "BTC",
      Math.min(this.startDaysAgo, FIVE_YEARS_IN_DAYS)
    );
    return { adaPrices, adaVolumes, btcPrices, btcVolumes };
  }

  private sliceData(
    adaPrices: number[],
    adaVolumes: number[],
    btcPrices: number[],
    btcVolumes: number[]
  ): {
    slicedAdaPrices: number[];
    slicedAdaVolumes: number[];
    slicedBtcPrices: number[];
    slicedBtcVolumes: number[];
    startIndex: number;
  } {
    const startIndex = Math.max(0, adaPrices.length - this.startDaysAgo);
    const endIndex = Math.max(0, adaPrices.length - this.endDaysAgo);
    return {
      slicedAdaPrices: adaPrices.slice(startIndex, endIndex),
      slicedAdaVolumes: adaVolumes.slice(startIndex, endIndex),
      slicedBtcPrices: btcPrices.slice(startIndex, endIndex),
      slicedBtcVolumes: btcVolumes.slice(startIndex, endIndex),
      startIndex,
    };
  }

  private setupMocks(
    historicalAdaPrices: number[],
    historicalAdaVolumes: number[],
    historicalBtcPrices: number[],
    historicalBtcVolumes: number[],
    currentAdaPrice: number,
    currentBtcPrice: number
  ): { originalGetHistorical: any; originalGetCurrent: any } {
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
      currentBtcPrice,
    });

    return { originalGetHistorical, originalGetCurrent };
  }

  private restoreMocks(
    originalGetHistorical: any,
    originalGetCurrent: any
  ): void {
    require("../api/getHistoricalPricesAndVolumes").getHistoricalPricesAndVolumes =
      originalGetHistorical;
    require("../api/getCurrentPrices").getCurrentPrices = originalGetCurrent;
  }

  private calculateSharpeRatio(returns: number[]): number {
    const meanReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
    const stdDev =
      Math.sqrt(
        returns.reduce((a, b) => a + Math.pow(b - meanReturn, 2), 0) /
          (returns.length - 1)
      ) || 1;
    return (meanReturn / stdDev) * Math.sqrt(365);
  }

  private calculateMaxDrawdown(portfolioValues: number[]): number {
    let maxDD = 0;
    let peak = portfolioValues[0];
    for (const value of portfolioValues) {
      if (value > peak) peak = value;
      const dd = (peak - value) / peak;
      if (dd > maxDD) maxDD = dd;
    }
    return maxDD * 100;
  }

  public async run(): Promise<BacktestResult> {
    console.log("Starting backtest...");

    const { adaPrices, adaVolumes, btcPrices, btcVolumes } =
      await this.fetchHistoricalData();
    console.log(
      `ADA data points: ${adaPrices.length}, BTC data points: ${btcPrices.length}`
    );

    const {
      slicedAdaPrices,
      slicedAdaVolumes,
      slicedBtcPrices,
      slicedBtcVolumes,
      startIndex,
    } = this.sliceData(adaPrices, adaVolumes, btcPrices, btcVolumes);

    let usdBalance = this.INITIAL_USD;
    let adaBalance = 0;
    let avgBuyPrice = 0;
    const trades: Trade[] = [];
    const portfolioHistory: { timestamp: string; value: number }[] = [];
    const dailyReturns: number[] = [];
    let wins = 0;
    let completedCycles = 0;
    let daysSinceLastTrade = 0;

    for (let i = 14; i < slicedAdaPrices.length - 5; i += this.stepDays) {
      const currentAdaPrice = slicedAdaPrices[i];
      const timestamp = new Date(
        Date.now() - (adaPrices.length - startIndex - i) * 24 * 60 * 60 * 1000
      ).toISOString();

      const historicalAdaPrices = slicedAdaPrices.slice(0, i + 1);
      const historicalAdaVolumes = slicedAdaVolumes.slice(0, i + 1);
      const historicalBtcPrices = slicedBtcPrices.slice(0, i + 1);
      const historicalBtcVolumes = slicedBtcVolumes.slice(0, i + 1);

      const { originalGetHistorical, originalGetCurrent } = this.setupMocks(
        historicalAdaPrices,
        historicalAdaVolumes,
        historicalBtcPrices,
        historicalBtcVolumes,
        currentAdaPrice,
        btcPrices[i]
      );

      const predictor = new TradePredictor();
      const decision = await predictor.predict();

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
      const positionSizeFactor = Math.min(0.9, confidence);
      const portfolioValue = usdBalance + adaBalance * currentAdaPrice;
      const unrealizedProfit =
        adaBalance > 0 ? (currentAdaPrice - avgBuyPrice) / avgBuyPrice : 0;

      const buyCondition =
        decision.recommendation === Recommendation.Buy &&
        usdBalance > this.CASH_RESERVE + this.MIN_TRADE_USD &&
        daysSinceLastTrade >= this.MIN_HOLD_DAYS;
      const sellCondition =
        (decision.recommendation === Recommendation.Sell &&
          adaBalance > 0 &&
          daysSinceLastTrade >= this.MIN_HOLD_DAYS) ||
        (adaBalance > 0 &&
          (unrealizedProfit <= this.STOP_LOSS_THRESHOLD ||
            unrealizedProfit >= this.TAKE_PROFIT_THRESHOLD));

      if (buyCondition) {
        const usdToSpend = Math.max(
          this.MIN_TRADE_USD,
          usdBalance * positionSizeFactor * (1 - this.TRANSACTION_FEE)
        );
        if (usdBalance - usdToSpend < this.CASH_RESERVE) continue;
        const adaBought = usdToSpend / currentAdaPrice;
        adaBalance += adaBought;
        avgBuyPrice =
          adaBalance > 0
            ? (avgBuyPrice * (adaBalance - adaBought) +
                currentAdaPrice * adaBought) /
              adaBalance
            : currentAdaPrice;
        usdBalance -= usdToSpend / (1 - this.TRANSACTION_FEE);
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
          )}, USD: ${usdToSpend.toFixed(2)}, RSI: ${
            decision.rsi
          }, Profit Potential: ${(decision.profitPotential * 100).toFixed(2)}%`
        );
        daysSinceLastTrade = 0;
      } else if (sellCondition) {
        const adaToSell = adaBalance;
        const usdReceived =
          adaToSell * currentAdaPrice * (1 - this.TRANSACTION_FEE);
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
          )}, USD: ${usdReceived.toFixed(2)}, RSI: ${
            decision.rsi
          }, Profit Potential: ${(decision.profitPotential * 100).toFixed(2)}%`
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
      dailyReturns.push(
        portfolioHistory.length > 1
          ? (portfolioValue -
              portfolioHistory[portfolioHistory.length - 2].value) /
              portfolioHistory[portfolioHistory.length - 2].value
          : 0
      );
      this.restoreMocks(originalGetHistorical, originalGetCurrent);
    }

    const finalValue =
      usdBalance + adaBalance * slicedAdaPrices[slicedAdaPrices.length - 6];
    const totalReturn =
      ((finalValue - this.INITIAL_USD) / this.INITIAL_USD) * 100;
    const totalTrades = trades.length;
    const winRate = completedCycles > 0 ? (wins / completedCycles) * 100 : 0;
    const sharpeRatio = this.calculateSharpeRatio(dailyReturns);
    const maxDrawdown = this.calculateMaxDrawdown(
      portfolioHistory.map((h) => h.value)
    );

    console.log(`Backtest completed:
      Initial USD: $${this.INITIAL_USD}
      Final Value: $${finalValue.toFixed(2)}
      Total Return: ${totalReturn.toFixed(2)}%
      Total Trades: ${totalTrades}
      Win Rate: ${winRate.toFixed(2)}%
      Sharpe Ratio: ${sharpeRatio.toFixed(2)}
      Max Drawdown: ${maxDrawdown.toFixed(2)}%`);

    return {
      totalReturn,
      totalTrades,
      winRate,
      sharpeRatio,
      maxDrawdown,
      portfolioHistory,
      trades,
    };
  }

  public static async runWithTimeout(
    startDaysAgo?: number,
    endDaysAgo?: number,
    stepDays?: number
  ): Promise<BacktestResult> {
    const backtester = new TradeModelBacktester(
      startDaysAgo,
      endDaysAgo,
      stepDays
    );
    return new Promise((resolve, reject) => {
      setTimeout(() => {
        backtester.run().then(resolve).catch(reject);
      }, 2000);
    });
  }
}
