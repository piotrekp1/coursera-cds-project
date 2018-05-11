import pandas as pd
import numpy as np


def get_filled_indexes(df_gb_key):
    """
    Create array of indexes in format: ['shop_id', 'item_id', 'year', 'month']
    :param df_gb_key: dataframe with min and max date per key
    :return: np array of indexes with every date between min and max
    """

    def tolist(possibly_tuple):
        return list(possibly_tuple) if isinstance(possibly_tuple, tuple) else [possibly_tuple]

    arr = []
    for name, row in df_gb_key.iterrows():
        for date in pd.date_range(row['min'] - pd.DateOffset(months=1), row['max'], freq='M') + pd.DateOffset(days=1):
            arr.append(tolist(name) + [date.year, date.month])
    arr = np.asarray(arr, dtype=np.object)
    return arr


def create_cumdata(key_cols):
    date_cols = ['year', 'month']

    df = pd.read_hdf('data/processed/train/total.hdf')

    df_items = pd.read_hdf('data/processed/dimensions/items.hdf')
    df_categories = pd.read_hdf('data/processed/dimensions/item_categories.hdf')
    df_shops = pd.read_hdf('data/processed/dimensions/shops.hdf')

    df = pd.merge(df, df_items, on='item_id', copy=False)
    df = pd.merge(df, df_categories, on='item_category_id', copy=False)
    df = pd.merge(df, df_shops, on='shop_id', copy=False)

    df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str) + '-1')
    df_gb_key = df.groupby(key_cols)['date'].agg(['min', 'max'])

    new_indexes = get_filled_indexes(df_gb_key)

    df_gb_all = df.groupby(key_cols + date_cols)['item_cnt_day'].sum()
    df_reindexed = df_gb_all.reindex(index=new_indexes.T.tolist(), fill_value=0)
    df_reindexed = df_reindexed.to_frame().reset_index()

    df_reindexed['date'] = pd.to_datetime(
        df_reindexed['year'].astype(str) + '-' + df_reindexed['month'].astype(str) + '-1')
    df_reindexed['cumcount'] = df_reindexed.groupby(key_cols)['item_cnt_day'].cumsum()

    # dataframe has cumulative date about all the transactions before the given point
    df_reindexed['date'] += pd.DateOffset(months=1)
    df_reindexed['year'], df_reindexed['month'] = df_reindexed['date'].dt.year, df_reindexed['date'].dt.month
    df_reindexed.drop(['date', 'item_cnt_day'], axis=1, inplace=True)

    df_reindexed.to_hdf('data/processed/cumdata.hdf', '_'.join(key_cols))


from itertools import product

raw_keys = [
    ['shop_id', 'mall', 'city'],
    ['item_id', 'item_category_id', 'first_big_category', 'last_big_category']
]

keys = [
    [key] for key_sets in raw_keys for key in key_sets
] + list(map(list, product(*raw_keys)))

for key_cols in keys:
    print('_'.join(key_cols))
    create_cumdata(key_cols)
