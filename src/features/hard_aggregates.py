import pandas as pd
from datetime import datetime



df_items = pd.read_hdf('data/processed/dimensions/items.hdf')
df_categories = pd.read_hdf('data/processed/dimensions/item_categories.hdf')
df_shops = pd.read_hdf('data/processed/dimensions/shops.hdf')

from itertools import product

raw_keys = [
    ['shop_id', 'mall', 'city'],
    ['item_id', 'item_category_id', 'first_big_category', 'last_big_category']
]

keys = [
           [key] for key_sets in raw_keys for key in key_sets
       ] + list(map(list, product(*raw_keys)))


def aggregates_dates():
    now = pd.DateOffset(days=0)
    dates = {
        '1_month': (pd.DateOffset(months=1), now),
        '3_months': (pd.DateOffset(months=3), now),
        '6_months': (pd.DateOffset(months=6), now),
        'year': (pd.DateOffset(months=12), now),
        '1_month_ago': (pd.DateOffset(months=2), pd.DateOffset(months=1)),
        '2_quartals_ago': (pd.DateOffset(months=9), pd.DateOffset(months=6)),
    }
    return dates


def count_diff(df_cum, df_records, key, dates, aggr_name):
    df_records = df_records.copy()
    df_records['date'] = pd.to_datetime(df_records['year'].astype(str) + '-' + df_records['month'].astype(str) + '-1')
    df_records['aggr_from_date'] = df_records['date'] - dates[0]
    df_records['aggr_to_date'] = df_records['date'] - dates[1]

    df_records = pd.merge(df_records,
                          df_cum[key + ['max_date', 'min_date', 'cumcount_max']].drop_duplicates(),
                          on=key,
                          how='left',
                          copy=False
                          )
    df_records['cumcount_max'].fillna(0, inplace=True)
    df_records['max_date'].fillna(datetime(2000, 1, 1), inplace=True)
    df_records['min_date'].fillna(datetime(2000, 1, 1), inplace=True)

    df_records = pd.merge(df_records, df_cum[key + ['date', 'cumcount']],
                          left_on=key + ['aggr_from_date'], right_on=key + ['date'],
                          how='left',
                          copy=False,
                          suffixes=('','_right')
                          )
    df_records.drop('date_right', axis=1,inplace=True)
    df_records.rename(columns={'cumcount': 'cumcount_from'}, inplace=True)
    # make sure that records before last record have 0 cum, and after the last have total cum
    df_records['cumcount_from'].fillna(
        (df_records['aggr_from_date'] > df_records['max_date']) * df_records['cumcount_max'],
        inplace=True
    )

    df_records = pd.merge(df_records, df_cum[key + ['date', 'cumcount']],
                          left_on=key + ['aggr_to_date'], right_on=key + ['date'],
                          how='left',
                          copy=False,
                          suffixes=('', '_right')
                          )
    df_records.drop('date_right', axis=1,inplace=True)
    df_records.rename(columns={'cumcount': 'cumcount_to'}, inplace=True)
    # make sure that records before last record have 0 cum, and after the last have total cum
    df_records['cumcount_to'].fillna(
        (df_records['aggr_to_date'] > df_records['max_date']) * df_records['cumcount_max'],
        inplace=True
    )
    df_records['count_aggr'] = df_records['cumcount_to'] - df_records['cumcount_from']

    df_records.drop(['cumcount_from', 'cumcount_to', 'cumcount_max', 'min_date', 'max_date'], axis=1, inplace=True)
    df_records.rename(columns={'count_aggr': aggr_name}, inplace=True)
    return df_records


def join_max_cumcount(df, key):
    """
    Assumes that there is already max_date calculated for every key
    :param df: data grouped by year,month and the key
    :return:
    """
    df_cumcount_max = df[df['date'] == df['max_date']][
        key + ['cumcount']
    ]
    df = pd.merge(df, df_cumcount_max, on=key, suffixes=('', '_max'))
    return df


def join_max_min_date(df, key):
    """
    :param df: data grouped by year,month and the key
    :return:
    """
    df_max_min_date = df.groupby(key)['date'].agg(['min', 'max']).reset_index()
    df_max_min_date['max_date'] = df_max_min_date['max']
    df_max_min_date['min_date'] = df_max_min_date['min']

    df = pd.merge(df, df_max_min_date[key + ['max_date', 'min_date']], on=key)
    return df


def create_key_cumcount_df(df_cumdata, key):
    """
    create cumcount df for different key then cumdata
    :param df_cumdata: cumdata for lowest aggregation (shop_id, item_id)
    :param key: new key (e.g. shop_id)
    :return: df ready for calculating aggregates
    """
    df_key_cum = df_cumdata.groupby(key + ['year', 'month'], as_index=False)['cumcount'].sum()
    df_key_cum['date'] = pd.to_datetime(df_key_cum['year'].astype(str) + '-' + df_key_cum['month'].astype(str) + '-1')
    df_key_cum = join_max_min_date(df_key_cum, key)
    df_key_cum = join_max_cumcount(df_key_cum, key)

    return df_key_cum


def get_aggregates(df_ids):
    """
    create features for df_ids based on df historical data

    :param: df_cumdata - dataframe with historical data cumulatives
    :param: df_ids - dataframe with columns ['shop_id', 'item_id', 'year', 'month'],
            aggregates will be calculated for these rows

    """
    df_ids = pd.merge(df_ids, df_items, on='item_id', copy=False)
    df_ids = pd.merge(df_ids, df_categories, on='item_category_id', copy=False)
    df_ids = pd.merge(df_ids, df_shops, on='shop_id', copy=False)

    dates = aggregates_dates()
    for key in keys:
        df_cumdata = pd.read_hdf('data/processed/cumdata.hdf', '_'.join(key))

        df_key_cum = create_key_cumcount_df(df_cumdata, key)

        # calc aggregates
        for date_name, date in dates.items():
            aggr_name = 'count_aggr_' + date_name + '_' + '_'.join(key)
            print(df_ids.isnull().sum().sum())
            df_ids = count_diff(df_key_cum, df_ids, key, date, aggr_name)
        print('key: ', key)
        print('ids_nulls: ', df_ids.isnull().sum().sum())
    return df_ids


def weekly_aggregates_for_key(df_ids, keys):
    df_train_raw = pd.read_csv('data/raw/sales_train.csv', parse_dates=['date'])
    df_train_raw = pd.merge(df_train_raw, df_items, on='item_id', copy=False)
    df_train_raw = pd.merge(df_train_raw, df_categories, on='item_category_id', copy=False)
    df_train_raw = pd.merge(df_train_raw, df_shops, on='shop_id', copy=False)


    df_ids['date'] = pd.to_datetime(df_ids['year'].astype(str) + '-' + df_ids['month'].astype(str) + '-1')
    df_train_raw['next_month'] = df_train_raw['date'] + pd.DateOffset(months=1)
    df_train_raw['next_month_date'] = pd.to_datetime(
    df_train_raw['next_month'].dt.year.astype(str) + '-' + df_train_raw['next_month'].dt.month.astype(str) + '-1'
)
    df_train_raw['month_end_date'] = df_train_raw['next_month_date'] - pd.DateOffset(days=7)
    df_train_raw = df_train_raw[df_train_raw['date'] >= df_train_raw['month_end_date']]
    weekly_sales = df_train_raw.groupby(keys + ['next_month_date'], as_index=False)['item_cnt_day'].sum().rename(
        columns={'item_cnt_day' : 'count_last_week_' + '_'.join(keys)}
    )
    df_train_weekly = pd.merge(df_ids, weekly_sales,
             left_on=keys + ['date'],
             right_on=keys + ['next_month_date'],
             how='left'
            ).drop('next_month_date', axis=1).fillna(0)
    return df_train_weekly

def weekly_aggregates(df_ids):
    for key in keys:
        df_ids = weekly_aggregates_for_key(df_ids, key)
    return df_ids