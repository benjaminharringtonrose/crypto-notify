export const calculateIfEnoughData = <T>(
  minDays: number,
  dayIndex: number,
  calculateFn: () => T
): T | number => {
  return dayIndex >= minDays - 1 ? calculateFn() || 0 : 0;
};
