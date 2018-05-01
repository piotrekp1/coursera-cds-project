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

    df_items = pd.read_csv('data/raw/items.csv')
    df_categories = pd.read_csv('data/raw/item_categories.csv')
    df_categories['big_category'] = df_categories['item_category_name'].str.split().apply(lambda x: x[0])
    df = pd.merge(df, df_items, on='item_id')
    df = pd.merge(df, df_categories, on='item_category_id')


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

keys = [
    ['shop_id'],
    ['item_id'], ['item_category_id'], ['big_category'],
    ['shop_id', 'item_id'],
    ['shop_id', 'item_category_id'],
    ['shop_id', 'big_category']
]
for key_cols in keys:
    print('_'.join(key_cols))
    create_cumdata(key_cols)
