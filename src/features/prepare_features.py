import pandas as pd
import numpy as np

from datetime import date
from dateutil.relativedelta import relativedelta

from collections import defaultdict
from itertools import chain

df_items = pd.read_csv('data/raw/items.csv')
df_categories = pd.read_csv('data/raw/item_categories.csv')
df_categories['big_category'] = df_categories['item_category_name'].str.split().apply(lambda x: x[0])


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


def get_aggregates_for_many_months(df, periods, from_reldelta, to_reldelta=relativedelta(days=0)):
    """
    Create aggregates for more then one month with the help of get_aggregates function
    :param df: dataframe with data
    :param periods: list of pairs ( year, month) that aggregates should be created for
    :param from_reldelta: relative time of aggregate start
    :param to_reldelta: relative time of aggregate finish
    :return: Dict with dataframes, where dict key is key related DataFrame (represents aggregation level)
    """

    aggrs = [get_aggregates(df, *period, from_reldelta, to_reldelta) for period in periods]
    ret_dict = defaultdict(pd.DataFrame)
    for key, df in chain(aggr.items() for aggr in aggrs):
        ret_dict[key] = pd.concat(ret_dict[key], df)
        del df
    return ret_dict


def aggregates_dates():
    now = relativedelta(days=0)
    dates = {
        '1_month': (relativedelta(months=1), now),
        '3_months': (relativedelta(months=3), now),
        '6_months': (relativedelta(months=6), now),
        #    'year': (relativedelta(months=12), now),
        '1_month_ago': (relativedelta(months=2), relativedelta(months=1))
        #    '2_quartals_ago': (relativedelta(months=9), relativedelta(months=6)),
    }
    return dates


def prepare_features(df_data, df):
    """
    Preprocessing function for data
    :param df_data: dataframe with transaction history
    :param df: dataframe with columns: ['shop_id', 'item_id', 'year', 'month'], for these shops items and dates features
               will be generated
    :return: features ready for the model
    """
    df_data = pd.merge(df_data, df_items, on='item_id')
    df_data = pd.merge(df_data, df_categories, on='item_category_id')

    df = pd.merge(df, df_items, on='item_id')
    df = pd.merge(df, df_categories, on='item_category_id')

    dates = aggregates_dates()
    for date_name, curdate in dates.items():
        for key, aggregates in get_aggregates_for_many_months(df_data,
                                                              df[['year', 'month']].drop_duplicates().as_matrix(),
                                                              *curdate):
            df = pd.merge(df,
                          aggregates,
                          on=tolist(key) + ['year', 'month'],
                          how='left',
                          suffixes=('', '_'+'_'.join(tolist(key)))
                          )
            del aggregates
        return df