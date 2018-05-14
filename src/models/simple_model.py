import pandas as pd
from math import sqrt
from datetime import datetime
from sklearn import preprocessing
import pickle

import xgboost as xgb

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

df = pd.read_hdf('data/processed/train/aggr_train_zeros.hdf')
# df = pd.read_hdf('data/processed/train/aggr_train_zeros.hdf')
df_valid = pd.read_hdf('data/processed/validation/aggr_valid.hdf')

threshold = datetime(2015, 9, 1)

df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str) + '-1')
df_valid['date'] = pd.to_datetime(df_valid['year'].astype(str) + '-' + df_valid['month'].astype(str) + '-1')

df_train = df[df['date'] < threshold]
df_valid = df[df['date'] >= threshold]


def X(df):
    df['day_of_year'] = df['date'].dt.dayofyear
    date_cols = [
        'month', 'year',
        'fridays', 'day_of_year', 'days_in_a_month'
    ]
    one_hot_cols = [
        'shop_id', 'mall', 'city',
        'item_id', 'item_category_id', 'first_big_category', 'last_big_category',
    ]
    additional = [
        'expected_sales'
    ]

    df_X = df[
        [
            col for col in df.columns
            if col.startswith('count')
        ]
        + date_cols + one_hot_cols + additional
        ]

    # df_X = pd.get_dummies(df_X, columns=one_hot_cols)
    # df_X = pd.get_dummies(df_X, columns=one_hot_cols + date_cols)
    for f in df_X.columns:
        if df_X[f].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(df_X[f].values))
            df_X[f] = lbl.transform(list(df_X[f].values))
    return df_X


def XY(df):
    df_X = X(df)
    df_y = df['item_cnt_day']
    return df_X, df_y


def get_preds_trained_on_bought(model, df_X):
    df_X = df_X.reindex(columns=df_X_train.columns, fill_value=0)
    df_y_preds_if_buys = model.predict(df_X).clip(0, 20)
    df_y_existed_before = (df_X['count_aggr_1_month_ago_shop_id_item_id'] != 0)
    df_y_preds = df_y_existed_before * df_y_preds_if_buys
    return df_y_preds


def get_preds_trained_on_all(model, df_X):
    df_X = df_X.reindex(columns=df_X_train.columns, fill_value=0)
    return model.predict(df_X).clip(0, 20)


def previous_value_preds(df_X):
    return df_X['count_aggr_1_month_ago_shop_id_item_id'].clip(0, 20)


def validate(model, df_y_valid, df_X_valid):
    df_y_valid = df_y_valid.copy().clip(0, 20)
    df_y_valid_preds = get_preds_trained_on_all(model, df_X_valid)
    # df_y_valid_preds = previous_value_preds(df_X_valid)
    print('r2 score: ', r2_score(df_y_valid, df_y_valid_preds))
    print('RMSE: ', sqrt(mean_squared_error(df_y_valid, df_y_valid_preds)))


df_X_train, df_y_train = XY(df_train)
df_X_valid, df_y_valid = XY(df_valid)

scaler = StandardScaler()
# df_X_train_scaled = scaler.fit_transform(df_X_train)
# df_X_valid_scaled = scaler.transform(df_X_valid)
# model = LinearRegression()
# model = RandomForestRegressor(n_estimators=60, n_jobs=-1, max_depth=6)
# model.fit(df_X_train, df_y_train)



params = {'max_depth': 6, 'eta': 0.1, 'silent': 1}
num_round = 100
dtrain = xgb.DMatrix(df_X_train, df_y_train)
model = xgb.train(params, dtrain, num_round)

pickle.dump(model, open('models/model_xgb.pck', 'wb'))

validate(model, df_y_valid, df_X_valid)
print('train score: ', r2_score(df_y_train, model.predict(df_X_train)))

df_test = pd.read_hdf('data/processed/test/aggr_test.hdf')
df_test_X = X(df_test)
df_test['item_cnt_month'] = get_preds_trained_on_all(model, df_test_X)
df_test[['ID', 'item_cnt_month']].to_csv('data/submissions/submission_trained_on_year_rf.csv', index=False)

linreg_test = pd.read_csv('data/submissions/if_sells_submission_linreg.csv')
print('linreg RMSE: ', sqrt(mean_squared_error(linreg_test['item_cnt_month'], df_test['item_cnt_month'])))
rf_test = pd.read_csv('data/submissions/if_sells_submission_rf.csv')
print('rf RMSE: ', sqrt(mean_squared_error(rf_test['item_cnt_month'], df_test['item_cnt_month'])))
