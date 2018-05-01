import pandas as pd
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta

df_train = pd.read_hdf('data/processed/train/train_case1.hdf')
df_valid = pd.read_hdf('data/processed/validation/valid_case1.hdf')
df_items = pd.read_csv('data/raw/items.csv')
df_categories = pd.read_csv('data/raw/item_categories.csv')
df_categories['big_category'] = df_categories['item_category_name'].apply(lambda x: x.split()[0])

item_categories = df_items.set_index('item_id')['item_category_id']
item_big_categories = df_categories.set_index('item_category_id')['big_category']
df_train = pd.merge(df_train, df_items, on='item_id')
df_train = pd.merge(df_train, df_categories, on='item_category_id')


def tolist(agg): return [agg] if type(agg) == str else list(agg)


def get_aggregates(df,
                   year, month,
                   from_reldelta, to_reldelta=relativedelta(days=0)
                   ):
    agg_functions = {
        'item_price': ['std'],
        'item_cnt_day': ['sum', 'std']
    }
    agg_objects = list(agg_functions.keys())
    shops_granulation = 'shop_id'
    items_granulations = ['item_id', 'item_category_id', 'big_category']

    from_time = date(year, month, 1) - from_reldelta
    to_time = date(year, month, 1) - to_reldelta

    time_constraints = (df['date'] >= from_time) & (df['date'] < to_time)
    df_time_constrained = df[time_constraints]

    object_aggregates = [
        shops_granulation,
        *items_granulations +
         [(shops_granulation, items_granulation) for items_granulation in items_granulations]
    ]

    aggregates = {
        agg_by: df_time_constrained.groupby(tolist(agg_by), as_index=False)[agg_objects].agg(agg_functions)
        for agg_by in object_aggregates
    }
    return aggregates


now = relativedelta(days=0)
dates = {
    '1_month': (relativedelta(months=1), now),
    '3_months': (relativedelta(months=3), now),
    '6_months': (relativedelta(months=6), now),
    #    'year': (relativedelta(months=12), now),
    '1_month_ago': (relativedelta(months=2), relativedelta(months=1))
    #    '2_quartals_ago': (relativedelta(months=9), relativedelta(months=6)),
}


def date_condition(year, month): return ((month.isin([7,8,9])) & (year == 2014)) | (month.isin([6, 7, 8]) & (year == 2015))


df_agg_train = df_train[
    ['shop_id', 'item_id', 'item_category_id', 'big_category', 'month', 'year', 'date', 'item_cnt_day']]
x_cols = ['shop_id', 'item_id', 'item_category_id', 'big_category', 'month', 'year']
df_agg_train = df_agg_train.groupby(x_cols, as_index=False)['item_cnt_day'].sum()
df_agg_train = df_agg_train[date_condition(df_agg_train['year'], df_agg_train['month'])]
print(df_agg_train.shape)

counter = 0

for date_name, curdate in dates.items():
    counter += 1
    aggs_df = dict()
    for _, el in df_agg_train[['year', 'month']].drop_duplicates().iterrows():

        for agg_keys, agg_df in get_aggregates(df_train, el['year'], el['month'], *curdate).items():
            agg_df.columns = ['_'.join(list(col) + [date_name]).rstrip('_') if col[1] != '' else col[0] for col in
                              agg_df.columns]
            agg_df['year'] = el['year']
            agg_df['month'] = el['month']

            aggs_df[agg_keys] = pd.concat([aggs_df.get(agg_keys, pd.DataFrame()), agg_df])
            del agg_df

    for agg_keys, agg_df in aggs_df.items():
        df_agg_train = pd.merge(df_agg_train,
                                agg_df,
                                on=tolist(agg_keys) + ['year', 'month'],
                                how='left',
                                suffixes=('', '_'+ '_'.join(tolist(agg_keys)))
                                )
        del agg_df
print(df_agg_train.shape)
print('nulls %: ', df_agg_train.isnull().mean().mean())
df_agg_train.to_hdf('data/processed/train/aggr_case1.hdf', 'hard_aggregates', mode='w')