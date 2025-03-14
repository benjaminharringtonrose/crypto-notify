export const isAboveThreshold = ({
  prices,
  currentPrice,
}: {
  prices: number[];
  currentPrice: number;
}) => {
  return prices.find((threshold) => currentPrice >= threshold);
};

export const notificationSentMessage = (
  currentPrice: number,
  exceededThreshold: number
) => {
  return `Notification sent for price $${currentPrice} exceeding threshold $${exceededThreshold}`;
};

export const cooldownMessage = ({
  exceededThreshold,
  notificationCooldown,
  timeSinceLastNotification,
}: {
  exceededThreshold: number;
  notificationCooldown: number;
  timeSinceLastNotification: number;
}) => {
  return `Price threshold $${exceededThreshold} exceeded but notification cooldown active. ${Math.floor(
    (notificationCooldown - timeSinceLastNotification) / 60000
  )}`;
};

export const notExceededMessage = (currentPrice: number) => {
  return `No price thresholds exceeded. Current: $${currentPrice}`;
};

export const checkCardanoPriceErrorMessage = (error: any) => {
  return `Error: ${JSON.stringify(error)}`;
};

export const textMessage = (threshold: number, currentPrice: number) => {
  return `CARDANO ALERT: ADA has risen above $${threshold} and is now at $${currentPrice}`;
};

export const currentCardanoPriceMessage = (currentPrice: number) => {
  return `Current Cardano price: $${currentPrice}`;
};

export const fetchCardanoPriceErrorMessage = (error: any) => {
  return `Error fetching Cardano price: ${JSON.stringify(error)}`;
};

export const sendSmsErrorMessage = (error: any) => {
  return `Error sending SMS notification: ${JSON.stringify(error)}`;
};
