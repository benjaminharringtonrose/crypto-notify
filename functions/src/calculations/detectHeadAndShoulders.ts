/**
 * Detects a Head and Shoulders pattern with volume confirmation in a price and volume series.
 * @param {number[]} prices - Array of daily closing prices.
 * @param {number[]} volumes - Array of daily trading volumes.
 * @param {number} currentPrice - The current price to check against the pattern.
 * @returns {boolean} True if a volume-confirmed Head and Shoulders pattern is detected.
 */
export const detectHeadAndShoulders = (
  prices: number[],
  volumes: number[],
  currentPrice: number
): boolean => {
  let leftShoulder = 0,
    head = 0,
    rightShoulder = 0;
  let leftTrough = Infinity,
    rightTrough = Infinity;
  let leftShoulderIndex = -1,
    headIndex = -1,
    rightShoulderIndex = -1;

  for (let i = 1; i < prices.length - 1; i++) {
    if (prices[i] > prices[i - 1] && prices[i] > prices[i + 1]) {
      // Peak
      if (leftShoulder === 0) {
        leftShoulder = prices[i];
        leftShoulderIndex = i;
      } else if (
        prices[i] > leftShoulder &&
        head === 0 &&
        i > leftShoulderIndex + 2
      ) {
        head = prices[i];
        headIndex = i;
      } else if (
        head !== 0 &&
        prices[i] < head &&
        Math.abs(prices[i] - leftShoulder) / leftShoulder < 0.1 &&
        i > headIndex + 2
      ) {
        rightShoulder = prices[i];
        rightShoulderIndex = i;
        break;
      }
    }
  }

  if (rightShoulder !== 0) {
    for (let i = leftShoulderIndex + 1; i < headIndex; i++) {
      if (prices[i] < leftTrough) leftTrough = prices[i];
    }
    for (let i = headIndex + 1; i < rightShoulderIndex; i++) {
      if (prices[i] < rightTrough) rightTrough = prices[i];
    }
  }

  const neckline = Math.min(leftTrough, rightTrough);
  const avgVolume = volumes.reduce((sum, vol) => sum + vol, 0) / volumes.length;
  const headVolume = headIndex >= 0 ? volumes[headIndex] : 0;
  const rightShoulderVolume =
    rightShoulderIndex >= 0 ? volumes[rightShoulderIndex] : 0;

  return (
    rightShoulder !== 0 &&
    leftTrough !== Infinity &&
    rightTrough !== Infinity &&
    currentPrice < neckline &&
    (headVolume > avgVolume * 1.5 || rightShoulderVolume < headVolume * 0.8)
  );
};
