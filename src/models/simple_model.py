import pandas as pd
from math import sqrt
from datetime import datetime

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

df = pd.read_hdf('data/processed/train/aggr_train.hdf')
df_valid = pd.read_hdf('data/processed/validation/aggr_valid.hdf')

threshold = datetime(2015, 10, 1)

df_train = df[df['date'] < threshold]
df_valid_y_data = df[df['date'] == threshold][['shop_id', 'item_id', 'item_cnt_day']]

df_valid = pd.merge(
    df_valid, df_valid_y_data,
    on=['shop_id', 'item_id'],
    how='left'
).fillna(0)


def X(df):
    df['day_of_year'] = df['date'].dt.dayofyear
    one_hot_cols = ['month', 'year',
                    'shop_id', 'item_category_id', 'big_category'
                    ]
    df_X = df[
        [
            col for col in df.columns
            if col.startswith('count')
        ] + ['day_of_year']
         + one_hot_cols
        ]

    df_X = pd.get_dummies(df_X, columns=['shop_id', 'item_category_id', 'big_category'])
    #df_X = pd.get_dummies(df_X, columns=one_hot_cols)
    return df_X


def XY(df):
    df_X = X(df)
    df_y = df['item_cnt_day']
    return df_X, df_y


def get_preds(model, df_X):
    df_X = df_X.reindex(columns=df_X_train.columns, fill_value=0)
    df_y_preds_if_buys = model.predict(df_X).clip(0, 20)
    df_y_existed_before = (df_X['count_aggr_1_month_ago_shop_id_item_id'] != 0)
    df_y_preds = df_y_existed_before * df_y_preds_if_buys
    return df_y_preds


def previous_value_preds(df_X):
    return df_X['count_aggr_1_month_ago_shop_id_item_id'].clip(0, 20)
def validate(model, df_y_valid, df_X_valid):
    df_y_valid = df_y_valid.copy().clip(0, 20)
    df_y_valid_preds = get_preds(model, df_X_valid)
    #df_y_valid_preds = previous_value_preds(df_X_valid)
    print('r2 score: ', r2_score(df_y_valid, df_y_valid_preds))
    print('RMSE: ', sqrt(mean_squared_error(df_y_valid, df_y_valid_preds)))



df_X_train, df_y_train = XY(df_train)
df_X_valid, df_y_valid = XY(df_valid)

df_X_valid = df_X_valid.reindex(columns=df_X_train.columns, fill_value=0)


scaler = StandardScaler()
# df_X_train_scaled = scaler.fit_transform(df_X_train)
# df_X_valid_scaled = scaler.transform(df_X_valid)
model = LinearRegression()
model = RandomForestRegressor(n_estimators=60, n_jobs=-1, max_depth=9)
model.fit(df_X_train, df_y_train)
validate(model, df_y_valid, df_X_valid)
print('train score: ', r2_score(df_y_train, model.predict(df_X_train)))

df_test = pd.read_hdf('data/processed/test/aggr_test.hdf')
df_test_X = X(df_test)
df_test['item_cnt_month'] = get_preds(model, df_test_X)
df_test[['ID', 'item_cnt_month']].to_csv('data/submissions/if_sells_submission.csv', index=False)
