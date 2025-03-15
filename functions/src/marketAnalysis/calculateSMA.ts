/**
 * The Simple Moving Average is calculated by taking the average (arithmetic mean)
 * of a set of prices over a defined number of time periods.
 * For example, a 7-day SMA averages the closing prices of the last 7 days.
 */

export const calculateSMA = (prices: number[]) => {
  return prices.reduce((sum, price) => sum + price, 0) / prices.length;
};
