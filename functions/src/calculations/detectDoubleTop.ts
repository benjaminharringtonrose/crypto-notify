/**
 * Detects a Double Top pattern with volume confirmation in a price and volume series.
 * @param {number[]} prices - Array of daily closing prices.
 * @param {number[]} volumes - Array of daily trading volumes.
 * @param {number} currentPrice - The current price to check against the pattern.
 * @returns {boolean} True if a volume-confirmed Double Top pattern is detected.
 */
export const detectDoubleTop = (
  prices: number[],
  volumes: number[],
  currentPrice: number
): boolean => {
  let firstTop = 0;
  let secondTop = 0;
  let trough = Infinity;
  let firstTopIndex = -1;
  let secondTopIndex = -1;

  for (let i = 1; i < prices.length - 1; i++) {
    if (prices[i] > prices[i - 1] && prices[i] > prices[i + 1]) {
      // Peak
      if (firstTop === 0) {
        firstTop = prices[i];
        firstTopIndex = i;
      } else if (
        Math.abs(prices[i] - firstTop) / firstTop < 0.05 &&
        i > firstTopIndex + 2
      ) {
        // Within 5%, not adjacent
        secondTop = prices[i];
        secondTopIndex = i;
        break;
      }
    }
  }

  if (secondTop !== 0) {
    for (let i = firstTopIndex + 1; i < secondTopIndex; i++) {
      if (prices[i] < trough) {
        trough = prices[i];
      }
    }
  }

  const avgVolume = volumes.reduce((sum, vol) => sum + vol, 0) / volumes.length;
  const secondTopVolume = secondTopIndex >= 0 ? volumes[secondTopIndex] : 0;
  const postTopVolume =
    secondTopIndex + 1 < volumes.length
      ? volumes[secondTopIndex + 1]
      : volumes[volumes.length - 1];

  return (
    secondTop !== 0 &&
    trough !== Infinity &&
    currentPrice < trough &&
    (secondTopVolume > avgVolume * 2 || postTopVolume < secondTopVolume * 0.8)
  );
};
