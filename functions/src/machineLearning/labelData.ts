export const labelData = ({
  prices,
  dayIndex,
  threshold = 0.01,
  horizon = 3,
}: {
  prices: number[];
  dayIndex: number;
  threshold: number;
  horizon: number;
}): number => {
  if (dayIndex + horizon >= prices.length) return 0;
  const futureAvg =
    prices
      .slice(dayIndex + 1, dayIndex + horizon + 1)
      .reduce((a, b) => a + b, 0) / horizon;
  const dropPercent = (prices[dayIndex] - futureAvg) / prices[dayIndex];
  return dropPercent > threshold ? 1 : 0;
};
