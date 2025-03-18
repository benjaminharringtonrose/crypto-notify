import axios from "axios";
import { CoinGeckoPriceResponse } from "../types";
import { CARDANO_BITCOIN_PRICE_URL } from "../constants";

export const getCurrentPrices = async () => {
  const priceResponse = await axios.get<CoinGeckoPriceResponse>(
    CARDANO_BITCOIN_PRICE_URL
  );

  const currentAdaPrice = priceResponse.data.cardano.usd;
  const currentBtcPrice = priceResponse.data.bitcoin.usd;

  return {
    currentAdaPrice,
    currentBtcPrice,
  };
};
