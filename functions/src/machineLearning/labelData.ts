// Label data: 1 if price drops >5% in 3 days
export const labelData = (prices: number[], dayIndex: number): number => {
  if (dayIndex + 3 >= prices.length) return 0;
  const futureAvg =
    (prices[dayIndex + 1] + prices[dayIndex + 2] + prices[dayIndex + 3]) / 3;
  const dropPercent = (prices[dayIndex] - futureAvg) / prices[dayIndex];
  return dropPercent > 0.015 ? 1 : 0;
};
