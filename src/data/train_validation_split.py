import pandas as pd
from datetime import date
df_train_whole = pd.read_csv('../data/processed/df_train_w_date_features.csv',
                             keep_date_col=True, parse_dates=['date'])

def current_month(series, date):
    return (series.dt.year == date.year) & (series.dt.month == date.month)

case1_date = date(2015, 9, 1)
df_train1 = df_train_whole[df_train_whole['date'] < case1_date]
df_valid1 = df_train_whole[current_month(df_train_whole['date'], case1_date)]

case2_date = date(2015, 10, 1)
df_train2 = df_train_whole[df_train_whole['date'] < case2_date]
df_valid2 = df_train_whole[current_month(df_train_whole['date'], case2_date)]

df_train1.to_hdf('../data/processed/train/train_case1.hdf', 'tr_c1', mode='w')
df_train2.to_hdf('../data/processed/train/train_case2.hdf', 'tr_c2', mode='w')
df_valid1.to_hdf('../data/processed/validation/valid_case1.hdf', 'val_c1', mode='w')
df_valid2.to_hdf('../data/processed/validation/valid_case2.hdf', 'val_c2', mode='w')