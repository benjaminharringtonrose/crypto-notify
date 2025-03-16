import axios from "axios";

export const getHistoricalData = async (
  cryptoSymbol: string,
  totalDays: number,
  chunkDays: number = 90
): Promise<{ prices: number[]; volumes: number[] }> => {
  const prices: number[] = [];
  const volumes: number[] = [];
  const millisecondsPerDay = 24 * 60 * 60 * 1000;
  const endDate = new Date();
  const totalMilliseconds = totalDays * millisecondsPerDay;
  const chunkMilliseconds = chunkDays * millisecondsPerDay;

  const numChunks = Math.ceil(totalDays / chunkDays);

  for (let i = 0; i < numChunks; i++) {
    const chunkEnd = new Date(endDate.getTime() - i * chunkMilliseconds);
    const chunkStart = new Date(chunkEnd.getTime() - chunkMilliseconds);
    const actualChunkStart =
      chunkStart < new Date(endDate.getTime() - totalMilliseconds)
        ? new Date(endDate.getTime() - totalMilliseconds)
        : chunkStart;

    const toTimestamp = Math.floor(chunkEnd.getTime() / 1000);
    const limit = Math.min(
      chunkDays,
      Math.ceil(
        (toTimestamp - Math.floor(actualChunkStart.getTime() / 1000)) /
          (24 * 60 * 60)
      )
    );

    const response = await axios.get(
      `https://min-api.cryptocompare.com/data/v2/histoday`,
      {
        params: {
          fsym: cryptoSymbol.toUpperCase(),
          tsym: "USD",
          limit: limit,
          toTs: toTimestamp,
        },
      }
    );

    const data = response.data.Data.Data;
    const chunkPrices = data.map((entry: any) => entry.close);
    const chunkVolumes = data.map((entry: any) => entry.volumeto);

    prices.unshift(...chunkPrices);
    volumes.unshift(...chunkVolumes);

    // Delay to respect rate limits (adjust as needed)
    if (i < numChunks - 1) {
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }
  }

  return { prices, volumes };
};
