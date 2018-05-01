import pandas as pd
import numpy as np

df_train = pd.read_hdf('data/processed/train/total.hdf')
df_test = pd.read_hdf('data/processed/test/test.hdf')
df_valid = pd.read_hdf('data/processed/validation/valid_case1.hdf')

df_items = pd.read_csv('data/raw/items.csv')
df_categories = pd.read_csv('data/raw/item_categories.csv')
df_categories['big_category'] = df_categories['item_category_name'].str.split().apply(lambda x: x[0])
df_train = pd.merge(df_train, df_items, on='item_id')
df_train = pd.merge(df_train, df_categories, on='item_category_id')

def aggregates_dates():
    now = pd.DateOffset(days=0)
    dates = {
        '1_month': (pd.DateOffset(months=1), now),
        '3_months': (pd.DateOffset(months=3), now),
        # '6_months': (pd.DateOffset(months=6), now),
        #    'year': (relativedelta(months=12), now),
        # '1_month_ago': (pd.DateOffset(months=2), pd.DateOffset(months=1))
        #    '2_quartals_ago': (relativedelta(months=9), relativedelta(months=6)),
    }
    return dates


def count_diff(df_aggr, key, dates):
    df_aggr = df_aggr.copy()
    df_aggr['aggr_from_date'] = df_aggr['date'] - dates[0]
    df_aggr['aggr_to_date'] = df_aggr['date'] - dates[1]

    df_aggr = pd.merge(df_aggr, df_aggr[key + ['date', 'cumcount_item']],
                       left_on=key + ['aggr_from_date'], right_on=key + ['date'],
                       suffixes=('', '_from')
                       )

    df_aggr = pd.merge(df_aggr, df_aggr[key + ['date', 'cumcount_item']],
                       left_on=key + ['aggr_to_date'], right_on=key + ['date'],
                       suffixes=('', '_to')
                       )
    df_aggr['count_aggr'] = df_aggr['cumcount_item_to'] - df_aggr['cumcount_item_from']
    return df_aggr


def get_aggregates(df, df_train):
    """


    :param: df - dataframe with historical data
    :param: df_train - dataframe with columns ['shop_id', 'item_id', 'year', 'month'],
            aggregates will be calculated for these rows

    """
    keys = [
        ['shop_id'],
        ['item_id'],
        ['shop_id', 'item_id']
    ]
    dates = aggregates_dates()
    for key in keys:
        df_key = df.groupby(key + ['year', 'month'], as_index=False)[['item_cnt_day']].sum()
        df_key['date'] = pd.to_datetime(df_key['year'].astype(str) + '-' + df_key['month'].astype(str) + '-1')

        # calc aggregates
        df_key['cumcount_item'] = df_key.groupby(key)['item_cnt_day'].cumsum()
        df_key['date'] += pd.DateOffset(months=1)
        for date_name, date in dates.items():
            df_aggr = count_diff(df_key, key, date)
            df_aggr.rename(columns={'count_aggr': 'count_aggr_' + date_name}, inplace=True)
            df_key = pd.merge(
                df_key, df_aggr[key + ['date', 'count_aggr_' + date_name]],
                on=key + ['date']
            )

        # necessary because one month was added
        df_key['year'], df_key['month'] = df_key['date'].dt.year, df_key['date'].dt.month

        df_key.drop(['date', 'item_cnt_day', 'cumcount_item'], axis=1, inplace=True)
        df_key.columns = [col + '_' + '_'.join(key) if col not in key + ['year', 'month'] else col for col in
                          df_key.columns]

        df_train = pd.merge(df_train, df_key, on=key + ['year', 'month'], how='left')
    return df_train

print(df_test.columns)
df_aggr = get_aggregates(df_train,
                         df_test)
print(df_aggr.columns)
#train_key = ['shop_id', 'item_id', 'year', 'month']
#df_aggr = pd.merge(
#    df_aggr, df_train.groupby(train_key, as_index=False)[['item_cnt_day']].sum(),
#    on=train_key
#    )
#df_aggr.rename(columns={'item_cnt_day': 'sold_items'}, inplace=True)


df_aggr.to_hdf('data/processed/test/aggr_test.hdf', 'aggr', mode='w')
