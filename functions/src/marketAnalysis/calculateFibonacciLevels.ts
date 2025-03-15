/**
 * Calculates Fibonacci retracement levels based on a price range.
 * @param {number[]} prices - Array of daily closing prices.
 * @param {number} period - Period to find high/low (default: 30 days).
 * @returns {{ levels: number[], high: number, low: number }} Fibonacci levels, high, and low.
 */
export const calculateFibonacciLevels = (
  prices: number[],
  period: number = 30
): { levels: number[]; high: number; low: number } => {
  if (prices.length < period)
    return {
      levels: [0, 0, 0, 0, 0],
      high: prices[prices.length - 1],
      low: prices[0],
    };

  const recentPrices = prices.slice(-period);
  const high = Math.max(...recentPrices);
  const low = Math.min(...recentPrices);
  const range = high - low;

  const levels = [
    low + range * 0.236, // 23.6%
    low + range * 0.382, // 38.2%
    low + range * 0.5, // 50%
    low + range * 0.618, // 61.8%
    high, // 100%
  ];

  return { levels, high, low };
};
