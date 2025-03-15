export function calculateEMA(prices: number[], period: number): number {
  const k = 2 / (period + 1); // Smoothing factor
  let ema = prices[0]; // Start with first price
  for (let i = 1; i < prices.length; i++) {
    ema = prices[i] * k + ema * (1 - k);
  }
  return ema;
}
