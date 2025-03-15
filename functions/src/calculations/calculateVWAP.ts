/**
 * Calculates the Volume Weighted Average Price (VWAP) over a period.
 * @param {number[]} prices - Array of daily closing prices.
 * @param {number[]} volumes - Array of daily trading volumes.
 * @param {number} period - Number of days for calculation (default: 7).
 * @returns {number} The VWAP value.
 */
export const calculateVWAP = (
  prices: number[],
  volumes: number[],
  period: number = 7
): number => {
  if (prices.length < period || volumes.length < period)
    return prices[prices.length - 1]; // Fallback to latest price
  const recentPrices = prices.slice(-period);
  const recentVolumes = volumes.slice(-period);
  const priceVolumeSum = recentPrices.reduce(
    (sum, price, i) => sum + price * recentVolumes[i],
    0
  );
  const volumeSum = recentVolumes.reduce((sum, vol) => sum + vol, 0);
  return priceVolumeSum / volumeSum;
};
