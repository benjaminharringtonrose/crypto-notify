import axios from "axios";
import { fetchCardanoPriceErrorMessage } from "../utils";
import { CryptoIds, Currencies } from "../types";

export class CoinGeckoService {
  private baseUrl: string = "https://api.coingecko.com/api/v3";
  private id: CryptoIds;
  private currency: Currencies;

  public constructor({
    id,
    currency,
  }: {
    id: CryptoIds.Cardano;
    currency: Currencies.USD;
  }) {
    this.id = id;
    this.currency = currency;
  }

  public async getCurrentPrice(): Promise<number> {
    try {
      const response = await axios.get(`${this.baseUrl}/simple/price`, {
        params: {
          ids: this.id,
          vs_currencies: this.currency,
        },
      });
      return response.data.cardano.usd;
    } catch (error) {
      console.log(fetchCardanoPriceErrorMessage(error));
      throw error;
    }
  }
}
