/**
 * Detects a Triple Top pattern with volume confirmation in a price and volume series.
 * @param {number[]} prices - Array of daily closing prices.
 * @param {number[]} volumes - Array of daily trading volumes.
 * @param {number} currentPrice - The current price to check against the pattern.
 * @returns {boolean} True if a volume-confirmed Triple Top pattern is detected.
 */
export const detectTripleTop = (
  prices: number[],
  volumes: number[],
  currentPrice: number
): boolean => {
  let firstTop = 0;
  let secondTop = 0;
  let thirdTop = 0;
  let firstTrough = Infinity;
  let secondTrough = Infinity;
  let firstTopIndex = -1;
  let secondTopIndex = -1;
  let thirdTopIndex = -1;

  for (let i = 1; i < prices.length - 1; i++) {
    if (prices[i] > prices[i - 1] && prices[i] > prices[i + 1]) {
      // Peak
      if (firstTop === 0) {
        firstTop = prices[i];
        firstTopIndex = i;
      } else if (
        Math.abs(prices[i] - firstTop) / firstTop < 0.05 &&
        i > firstTopIndex + 2 &&
        secondTop === 0
      ) {
        secondTop = prices[i];
        secondTopIndex = i;
      } else if (
        secondTop !== 0 &&
        Math.abs(prices[i] - secondTop) / secondTop < 0.05 &&
        i > secondTopIndex + 2
      ) {
        thirdTop = prices[i];
        thirdTopIndex = i;
        break;
      }
    }
  }

  if (thirdTop !== 0) {
    for (let i = firstTopIndex + 1; i < secondTopIndex; i++) {
      if (prices[i] < firstTrough) firstTrough = prices[i];
    }
    for (let i = secondTopIndex + 1; i < thirdTopIndex; i++) {
      if (prices[i] < secondTrough) secondTrough = prices[i];
    }
  }

  const supportLevel = Math.min(firstTrough, secondTrough);
  const avgVolume = volumes.reduce((sum, vol) => sum + vol, 0) / volumes.length;
  const thirdTopVolume = thirdTopIndex >= 0 ? volumes[thirdTopIndex] : 0;
  const postTopVolume =
    thirdTopIndex + 1 < volumes.length
      ? volumes[thirdTopIndex + 1]
      : volumes[volumes.length - 1];

  return (
    thirdTop !== 0 &&
    firstTrough !== Infinity &&
    secondTrough !== Infinity &&
    currentPrice < supportLevel &&
    (thirdTopVolume > avgVolume * 2 || postTopVolume < thirdTopVolume * 0.8)
  );
};
