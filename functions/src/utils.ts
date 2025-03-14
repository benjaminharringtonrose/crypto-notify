export const isAboveThreshold = ({
  prices,
  currentPrice,
}: {
  prices: number[];
  currentPrice: number;
}) => {
  return prices.find((threshold) => currentPrice >= threshold);
};
