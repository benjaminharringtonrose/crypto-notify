/**
 * Calculates the Average True Range (ATR) over a period using daily price changes.
 * @param {number[]} prices - Array of daily closing prices.
 * @param {number} period - Number of days for averaging (default: 14).
 * @returns {number} The ATR value.
 */
export const calculateATR = (prices: number[], period: number = 14): number => {
  if (prices.length < period + 1) return 0; // Need enough data
  const trueRanges = [];
  for (let i = 1; i < prices.length; i++) {
    trueRanges.push(Math.abs(prices[i] - prices[i - 1]));
  }
  const recentRanges = trueRanges.slice(-period);
  return recentRanges.reduce((sum, range) => sum + range, 0) / period;
};
