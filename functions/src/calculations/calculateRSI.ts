/**
 * The Relative Strength Index, developed by J. Welles Wilder Jr.,
 * is calculated using the average gains and losses of an asset’s price
 * over a specific period (typically 14 days). It ranges from 0 to 100,
 * with higher values indicating stronger upward momentum and lower values
 * suggesting downward pressure.
 * @param prices
 * @param period
 * @returns
 */

export const calculateRSI = (prices: number[], period = 14) => {
  if (prices.length < period + 1) return;
  let gains = 0,
    losses = 0;

  // Calculate initial gains/losses
  for (let i = 1; i <= period; i++) {
    const change = prices[i] - prices[i - 1];
    if (change > 0) gains += change;
    else losses += Math.abs(change);
  }

  let avgGain = gains / period;
  let avgLoss = losses / period;

  // Smooth RSI for remaining periods
  for (let i = period + 1; i < prices.length; i++) {
    const change = prices[i] - prices[i - 1];
    const gain = change > 0 ? change : 0;
    const loss = change < 0 ? Math.abs(change) : 0;
    avgGain = (avgGain * (period - 1) + gain) / period;
    avgLoss = (avgLoss * (period - 1) + loss) / period;
  }

  const rs = avgGain / avgLoss;
  return 100 - 100 / (1 + rs);
};
