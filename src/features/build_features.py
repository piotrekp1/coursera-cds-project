from src.features.hard_aggregates import get_aggregates, weekly_aggregates
from calendar import weekday, monthrange, FRIDAY
import pandas as pd

df_total = pd.read_hdf('data/processed/train/total.hdf')
df_train = pd.read_hdf('data/processed/train/with_zeros.hdf')
df_test = pd.read_hdf('data/processed/test/test.hdf')
df_valid = pd.read_hdf('data/processed/validation/valid3.hdf')

keys = ['shop_id', 'item_id', 'year', 'month']


def friday_num(y, m):
    return sum(1 for d in range(*monthrange(y, m)) if weekday(y, m, d + 1) == FRIDAY)


def join_fridays(df):
    fridays_df_arr = [[row['year'], row['month'], friday_num(row['year'], row['month'])] for _, row in
                      df[['year', 'month']].drop_duplicates().iterrows()]
    fridays_df = pd.DataFrame(fridays_df_arr, columns=['year', 'month', 'fridays'])
    return pd.merge(df, fridays_df, on=['year', 'month'], copy=False)


def build_features(df_indexes):
    df_aggr = get_aggregates(df_indexes)

    df_aggr = weekly_aggregates(df_aggr)

    # date features:
    df_aggr = join_fridays(df_aggr)
    df_aggr['days_in_a_month'] = df_aggr['date'].dt.daysinmonth

    df_aggr['expected_sales'] = df_aggr['count_last_week_shop_id_item_id'] / 7 * df_aggr['days_in_a_month']

    return df_aggr


df_aggr_test = build_features(df_test)
df_aggr_test.to_hdf('data/processed/test/aggr_test.hdf', 'aggr', mode='w')

df_aggr_total = build_features(
    df_total[df_total['year'] > 2013][['shop_id', 'item_id', 'year', 'month']].drop_duplicates())
df_total_y = df_total.groupby(keys, as_index=False)['item_cnt_day'].sum()
df_aggr_total = pd.merge(df_aggr_total, df_total_y, on=keys)
df_aggr_total.to_hdf('data/processed/train/aggr_total.hdf', 'aggr', mode='w')

df_aggr_train = build_features(df_train)
df_aggr_train = pd.merge(
    df_aggr_train, df_total_y[keys + ['item_cnt_day']],
    on=keys,
    how='left'
).fillna(0)
df_aggr_train.to_hdf('data/processed/train/aggr_train_zeros.hdf', 'aggr_train_zeros', mode='w')

df_aggr_valid = build_features(df_valid)
df_aggr_valid = pd.merge(
    df_aggr_valid, df_total_y[keys + ['item_cnt_day']],
    on=keys,
    how='left'
).fillna(0)

df_aggr_valid.to_hdf('data/processed/validation/aggr_valid.hdf', 'aggr_valid', mode='w')
