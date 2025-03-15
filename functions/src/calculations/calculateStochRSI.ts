import { calculateRSI } from "./calculateRSI";

/**
 * Calculates the Stochastic RSI (StochRSI) over a period with a smoothing signal line.
 * @param {number[]} prices - Array of daily closing prices.
 * @param {number} rsiPeriod - Period for RSI calculation (default: 14).
 * @param {number} stochPeriod - Period for StochRSI lookback (default: 14).
 * @param {number} smoothPeriod - Period for smoothing the signal line (default: 3).
 * @returns {{ stochRsi: number, signal: number }} Current StochRSI and its signal line.
 */
export function calculateStochRSI(
  prices: number[],
  rsiPeriod: number = 14,
  stochPeriod: number = 14,
  smoothPeriod: number = 3
): { stochRsi: number; signal: number } {
  if (prices.length < rsiPeriod + stochPeriod)
    return { stochRsi: 50, signal: 50 }; // Fallback

  // Calculate RSI for each day in the lookback period
  const rsiValues: number[] = [];
  for (let i = 0; i <= prices.length - rsiPeriod; i++) {
    const slice = prices.slice(i, i + rsiPeriod + 1); // +1 for initial change
    const rsi = calculateRSI(slice);
    if (rsi) {
      rsiValues.push(rsi);
    }
  }

  // Get the last 'stochPeriod' RSI values
  const recentRsis = rsiValues.slice(-stochPeriod);
  const currentRsi = recentRsis[recentRsis.length - 1];
  const lowestRsi = Math.min(...recentRsis);
  const highestRsi = Math.max(...recentRsis);

  // Calculate StochRSI
  const stochRsi =
    highestRsi === lowestRsi
      ? 50
      : ((currentRsi - lowestRsi) / (highestRsi - lowestRsi)) * 100;

  // Calculate signal line (3-day SMA of StochRSI)
  const recentStochRsis = rsiValues
    .slice(-stochPeriod - smoothPeriod + 1)
    .map((_, i) => {
      const subRsis = rsiValues.slice(i, i + stochPeriod);
      const subLow = Math.min(...subRsis);
      const subHigh = Math.max(...subRsis);
      return subHigh === subLow
        ? 50
        : ((subRsis[subRsis.length - 1] - subLow) / (subHigh - subLow)) * 100;
    });
  const signal =
    recentStochRsis.slice(-smoothPeriod).reduce((sum, val) => sum + val, 0) /
    smoothPeriod;

  return { stochRsi, signal };
}
