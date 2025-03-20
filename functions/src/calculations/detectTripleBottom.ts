export const detectTripleBottom = (
  prices: number[],
  volumes: number[],
  currentPrice: number
): boolean => {
  if (prices.length < 20 || volumes.length < 20) return false;
  const recentPrices = prices.slice(-20);
  const recentVolumes = volumes.slice(-20);

  // Identify valleys (local lows)
  const valleys = [];
  for (let i = 1; i < recentPrices.length - 1; i++) {
    if (
      recentPrices[i] < recentPrices[i - 1] &&
      recentPrices[i] < recentPrices[i + 1]
    ) {
      valleys.push({
        price: recentPrices[i],
        volume: recentVolumes[i],
        index: prices.length - 20 + i,
      });
    }
  }
  if (valleys.length < 3) return false;

  // Take the last three valleys
  const lastThreeValleys = valleys.slice(-3);
  const priceRange = Math.max(...recentPrices) - Math.min(...recentPrices);

  // Check if the valleys are at similar price levels (within 5% of range)
  const areSimilar = lastThreeValleys.every(
    (v) => Math.abs(v.price - lastThreeValleys[0].price) < priceRange * 0.05
  );

  // Check volume trend: decreasing across valleys, then spiking on breakout
  const volumeDecreasing =
    lastThreeValleys[1].volume < lastThreeValleys[0].volume &&
    lastThreeValleys[2].volume < lastThreeValleys[1].volume;
  const avgValleyVolume =
    lastThreeValleys.reduce((sum, v) => sum + v.volume, 0) / 3;
  const breakoutVolume = recentVolumes[recentVolumes.length - 1]; // Current volume
  const isVolumeSpike = breakoutVolume > avgValleyVolume * 1.5; // 50% higher than average valley volume

  // Check if price is breaking above recent resistance
  const recentHigh = Math.max(...recentPrices.slice(-5));
  const isBreakingUp = currentPrice > recentHigh;

  return areSimilar && volumeDecreasing && isVolumeSpike && isBreakingUp;
};
