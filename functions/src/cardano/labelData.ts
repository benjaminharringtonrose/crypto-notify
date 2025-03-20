export const labelData = ({
  prices,
  dayIndex,
  threshold = 0.015, // Increased from 0.01
  horizon = 3,
}: {
  prices: number[];
  dayIndex: number;
  threshold: number;
  horizon: number;
}): number => {
  if (dayIndex + horizon >= prices.length) return 1; // Default to hold if insufficient data
  const futureAvg =
    prices
      .slice(dayIndex + 1, dayIndex + horizon + 1)
      .reduce((a, b) => a + b, 0) / horizon;
  const priceChangePercent = (futureAvg - prices[dayIndex]) / prices[dayIndex];

  if (priceChangePercent > threshold) return 2; // Buy
  if (priceChangePercent < -threshold) return 0; // Sell
  return 1; // Hold
};
