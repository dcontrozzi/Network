import pandas as pd

import Bond.Sectors as sectors_enums
import Bond.Rating as ratings_enums
import Bond.BondIndic as bond_indic

class BondsRefDataMap:

    def __init__(self):
        self.isin_to_indic_df = pd.DataFrame()
        self.isin_to_indic_map = {}
        self.isin_to_tenor = {}
        self.isin_to_sector = {}
        self.isin_to_rating = {}
        self.isin_to_aggregated_rating = {}
        self.isin_to_rating_category = {}
        self.isin_to_payment_rank = {}
        self.isin_to_ticker = {}
        self.ticker_to_isin = {}

    def load_bond_indic(self, path):
        self.isin_to_indic_df = pd.read_csv(path)
        index_names = self.isin_to_indic_df[self.isin_to_indic_df['id_cusip'] == '#N/A Invalid Security'].index
        self.isin_to_indic_df.drop(index_names, inplace=True)
        df_dict = self.isin_to_indic_df.to_dict(orient="index")
        self.isin_to_indic_map = {v['isin']: bond_indic.BondIndic(**v) for k,v in df_dict.items()}
        self.isin_to_tenor = {val['isin']: val['blp_spread_benchmark_name'] for i, val in df_dict.items() if 'blp_spread_benchmark_name' in val}
        self.isin_to_sector = {val['isin']: sectors_enums.Sectors.from_string(val['industry_sector']) for i, val in df_dict.items() if 'industry_sector' in val}
        self.isin_to_rating = {val['isin']: ratings_enums.Rating.from_str(val['bb_composite']) for i, val in df_dict.items() if 'bb_composite' in val}
        self.isin_to_aggregated_rating = {val['isin']: ratings_enums.AggregatedRating.from_str(val['bb_composite']) for i, val in df_dict.items() if 'bb_composite' in val}
        self.isin_to_rating_category = {i: ratings_enums.RatingCategory.get_rating_category(val) for i, val in self.isin_to_rating.items()}
        self.isin_to_payment_rank = {val['isin']: val['payment_rank'] for i, val in df_dict.items() if 'payment_rank' in val}
        self.isin_to_ticker = {val['isin']: val['ticker'] for i, val in df_dict.items() if 'ticker' in val}
        tickers = set(self.isin_to_ticker.values())
        for t in tickers:
            self.ticker_to_isin[t] = [isin for isin, ticker in self.isin_to_ticker.items() if ticker == t]

    def select(self, key, value):
        return [isin for isin, val in self.isin_to_indic_map.items() if getattr(val, key) == value]

    def standard_factor(self, isin):
        tenor = str(self.isin_to_tenor[isin]) if isin in self.isin_to_tenor else 'tenor'
        sector = sectors_enums.Sectors.to_string(self.isin_to_sector[isin]) if isin in self.isin_to_sector else 'sector'
        rating = ratings_enums.Rating.to_str(self.isin_to_rating[isin]) if isin in self.isin_to_rating else 'rating'
        return tenor + '_' + sector + '_' + rating
