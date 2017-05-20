import pandas as pd
from sklearn import cross_validation
import xgboost as xgb
from sklearn.utils import shuffle
#=========================================================================================================================
#datetime element of data is pretreated and formatted
#Only date element of datetime is used in this algorithm

data_ss = pd.DataFrame.from_csv("/home/isaacalabi/.virtualenvs/kaggleprojects/lib/python3.5/site-packages/sessions.csv")
data_ss['posted_date'] = pd.DatetimeIndex(data_ss['posted_datetime']).date
data_ss['date_posted'] = data_ss['posted_date'].apply(lambda x: x.strftime('%d%m%Y'))

data_ss['start_date'] = pd.DatetimeIndex(data_ss['start_datetime']).date
data_ss['start_date'].apply(lambda x: x.strftime('%d%m%Y'))
data_ss['date_started'] = data_ss['start_date'].apply(lambda x: x.strftime('%d%m%Y'))

data_ss['end_date'] = pd.DatetimeIndex(data_ss['end_datetime']).date
data_ss['end_date'].apply(lambda x: x.strftime('%d%m%Y'))
data_ss['date_ended'] = data_ss['end_date'].apply(lambda x: x.strftime('%d%m%Y'))

del data_ss['posted_datetime']
del data_ss['start_datetime']
del data_ss['end_datetime']
del data_ss['posted_date']
del data_ss['start_date']
del data_ss['end_date']

data_x = data_ss

data_ccg = pd.DataFrame.from_csv("/home/isaacalabi/.virtualenvs/kaggleprojects/lib/python3.5/site-packages/ccgs.csv")
data_pra = pd.DataFrame.from_csv("/home/isaacalabi/.virtualenvs/kaggleprojects/lib/python3.5/site-packages/practices.csv")
#===========================================================================================================================
data_ccg['ccg_id'] = data_ccg.index
data_pra['practice_id']=data_pra.index
data_ccg_pra = pd.merge(data_ccg, data_pra,how='left', on='ccg_id')
data_new = pd.merge(data_ccg_pra, data_x, how='left', on='practice_id')

data_new = data_new.replace({'completed': '1', 'filled': '2', 'withdrawn': '3', 'expired': 'na', 'system_invalidated':'na'})
data =  data_new.dropna()

value_list = ['1']
data_1 = data[data.status.isin(value_list)]
data_12 = data_1[1:3445]

value_list1 = ['2']
data_2 = data[data.status.isin(value_list1)]

value_list2 = ['3']
data_3 = data[data.status.isin(value_list2)]

df =pd.concat([data_12, data_2, data_3], axis=0)

y = df['status']
y = y.astype(float)

del df['status']
del df['ccg_name']
del df['locum_id']

X = df.astype(float)
#X = X_1[1:9001]
#test =X_1[9001:len(X_1)]
#Y = y[1:9001]
#Y_test=y[9001:len(X_1)]

X, y = shuffle(X, y, random_state=0)


#############################################################################################################################
#This section uses an appropriate algorithm to train a model

# Split data into train and validation sets

X_train, X_val, y_train, y_val = cross_validation.train_test_split(X, y, train_size=0.80, random_state=0)
#===========================================================================================================================
#Algorithm will train a model using the training data

dtrain = xgb.DMatrix(X_train, y_train)

dvalid = xgb.DMatrix(X_val, y_val)

params = {
    "objective": "multi:softprob",
    "num_class": 4,
    "booster": "gbtree",
    "max_depth":4,#controls model complexity, higher values may cause overfitting, higher variance
    "eval_metric": "mlogloss",
    "eta": 0.1,# learning rate, you can reduce eta to prevent overfitting but remember to increase num_round
    "silent": 1,
    "alpha": 0,#L1 regularization on weights, increase to make model more conservative
    "seed": 0,
    "lambda": 6,#L2 regularization on weights, increase to make model more conservative
    "sample_type": "uniform",
    "normalize_type":"weighted",
    "subsample": 1,#adds randomness to make training robust to noise. 0.5 means half of data instances collected and noise added
    "colsample_bytree": 0.5,#adds randomness to make training robust to noise. subsamples ratio of columns,not rows
    "max_delta_step":1,
    "num_round": 1000

}

watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

gbm = xgb.train(params, dtrain, 2000, evals=watchlist,
                early_stopping_rounds=100, verbose_eval=True)

print("Training step")

dtrain = xgb.DMatrix(X, y)

gbm = xgb.train(params, dtrain, 2000, verbose_eval=True)

importance = gbm.get_fscore()

print("The importance of the features:", importance)
