import pandas as pd
from sklearn import cross_validation
import xgboost as xgb

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

data =  data_new.dropna()

y = data['status']
del data['status']

del data['ccg_name']

X = data.astype(float)

X_new = X[1:46501]
test =X[46501:len(X)] #created a test set to compute classification report

y_new = y.replace({'completed': '1', 'filled': '2', 'withdrawn': '3', 'expired': '4', 'system_invalidated':'5'})

Y = y_new.astype(float)
Y_new = Y[1:46501]
Y_test=Y[46501:len(Y)] #labels to be used to compute classification report

#############################################################################################################################
#This section uses an appropriate algorithm to train a model

# Split data into train and validation sets

X_train, X_val, y_train, y_val = cross_validation.train_test_split(X_new, Y_new, train_size=0.90, stratify = Y_new, random_state=42)
#===========================================================================================================================
#Algorithm will train a model using the training data

dtrain = xgb.DMatrix(X_train, y_train)

dvalid = xgb.DMatrix(X_val, y_val)

params = {
    "objective": "multi:softprob",
    "num_class": 6,
    "booster": "gbtree",
    "max_depth":2,#controls model complexity, higher values may cause overfitting, higher variance
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

dtrain = xgb.DMatrix(X_new, Y_new)

gbm = xgb.train(params, dtrain, 2000, verbose_eval=True)

importance = gbm.get_fscore()

print("The importance of the features:", importance)

Y_pred = gbm.predict(xgb.DMatrix(test),ntree_limit=gbm.best_iteration)

#Classification report to evalute true positive, false positive, true negative, false negative

from sklearn.metrics import classification_report

print(classification_report(Y_test, Y_pred[:,1].round(0)))
