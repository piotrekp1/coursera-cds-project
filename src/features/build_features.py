from src.features.hard_aggregates import get_aggregates
import pandas as pd

df_total = pd.read_hdf('data/processed/train/total.hdf')
df_test = pd.read_hdf('data/processed/test/test.hdf')
df_valid = pd.read_hdf('data/processed/validation/valid.hdf')

keys = ['shop_id', 'item_id', 'year', 'month']


def build_features(df_indexes):
    df_aggr = get_aggregates(df_indexes)
    return df_aggr


#df_aggr_test = build_features(df_test)
#df_aggr_test.to_hdf('data/processed/test/aggr_test.hdf', 'aggr', mode='w')

#df_aggr_total = build_features(
#    df_total[df_total['year'] > 2013][['shop_id', 'item_id', 'year', 'month']].drop_duplicates())
#df_total_y = df_total.groupby(keys, as_index=False)['item_cnt_day'].sum()
#df_aggr_total = pd.merge(df_aggr_total, df_total_y, on=keys)
#df_aggr_total.to_hdf('data/processed/train/aggr_total.hdf', 'aggr', mode='w')


df_aggr_valid = build_features(df_valid)
df_aggr_valid.to_hdf('data/processed/validation/aggr_valid.hdf', 'aggr_valid', mode='w')