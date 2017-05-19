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

data_new = data_new.replace({'completed': '1', 'filled': '0', 'withdrawn': '0', 'expired': '0', 'system_invalidated':'0'})
#status designated as 1 can always change. I have used status= completed this time as it makes more business sense
data =  data_new.dropna()

value_list = ['1']
data_1 = data[data.status.isin(value_list)]
data_12 = data_1[1:6695]

value_list1 = ['0']
data_0 = data[data.status.isin(value_list1)]

data_comb =pd.concat([data_12, data_0], axis=0)
data_comb1 = shuffle(data_comb, random_state=0)

y = data_comb1['status']

del data_comb1['locum_id']
del data_comb1['status']
del data_comb1['ccg_name']

X_1 = data_comb1.astype(float)
#X_1 = shuffle(X, random_state=0)

X_new = X_1[1:13001]
test =X_1[13001:len(X_1)] #created a test set to compute classification report

Y = y.astype(float)
Y_new = Y[1:13001]
Y_test=Y[13001:len(Y)] #labels to be used to compute classification report

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
    "num_class": 2,
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

dtrain = xgb.DMatrix(X_new, Y_new)

gbm = xgb.train(params, dtrain, 2000, verbose_eval=True)

importance = gbm.get_fscore()

print("The importance of the features:", importance)

Y_pred = gbm.predict(xgb.DMatrix(test),ntree_limit=gbm.best_iteration)

#Classification report to evalute true positive, false positive, true negative, false negative

from sklearn.metrics import classification_report

print(classification_report(Y_test, Y_pred[:,1].round(0)))

from sklearn.metrics import matthews_corrcoef

print("mcc:", matthews_corrcoef(Y_test, Y_pred[:,1].round(0)))
# The Matthews correlation coefficient (+1 represents a perfect prediction,
# 0 an average random prediction and -1 and inverse prediction)
