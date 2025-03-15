/**
 * Calculates the standard deviation of an array of numbers.
 * @param {number[]} values - Array of numbers (e.g., prices).
 * @param {number} mean - Pre-calculated mean of the values.
 * @returns {number} The standard deviation.
 */
export function calculateStdDev(values: number[], mean: number): number {
  const variance =
    values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) /
    values.length;
  return Math.sqrt(variance);
}
