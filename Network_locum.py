import pandas as pd

data_ss = pd.DataFrame.from_csv("C:/Users/Home/Desktop/sessions.csv")
data_ccg = pd.DataFrame.from_csv("C:/Users/Home/Desktop/ccgs.csv")
data_pra = pd.DataFrame.from_csv("C:/Users/Home/Desktop/practices.csv")

#Merge data using ccg_id and practice_id
data_ccg['ccg_id'] = data_ccg.index
data_pra['practice_id']=data_pra.index
data_ccg_pra = pd.merge(data_ccg, data_pra,how='left', on='ccg_id') #merge datasets using ccg_id
data_new = pd.merge(data_ccg_pra, data_ss, how='left', on='practice_id') #merge datasets using practice_id

#Total number of unique status for all ccg_id

print(data_new['status'].value_counts())

####################################################################################################
x = data_new.groupby(['ccg_id']).agg({'posted_datetime':'count'}) #count the number of posted jobs per ccg_id

print(x['posted_datetime'].describe().transpose()) #describes the key stats for the total number of postings

#select ccg_id and status column, select ccg_id columns where status is completed
#and then count the number of completed for each ccg_id

y = data_new[['ccg_id','status']] 
value_list = ['filled']
y_new = y[y.status.isin(value_list)]
y_comp = y_new.groupby(['ccg_id']).agg({'status':'count'}) #count number of filled jobs per ccg_id

z = data_new.groupby(['ccg_id']).agg({'hourly_rate':'mean'})
print(z['hourly_rate'].describe().transpose()) #describes the key stats for the hourly_rate column

data1 = pd.concat([x, y_comp, z], axis=1)
data = data1.dropna()

data['n_posted_jobs'] = data['posted_datetime']
del data['posted_datetime']

data['n_filled_jobs'] = data['status']
del data['status']

data['ave_hourly_rate'] = data['hourly_rate']
del data['hourly_rate']

data['fill_rate'] = (data['n_filled_jobs']/data['n_posted_jobs'])*100 #calculate fill rate

data['ccg_id'] = data.index

data = data[['ccg_id', 'ave_hourly_rate', 'n_posted_jobs', 'n_filled_jobs', 'fill_rate']]
data.dropna(axis=0, how='any')

###############################################################################################
#This part of the code computes the correlation between ave_hourly_rate and the fill_rate
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.stats import pearsonr
import matplotlib
matplotlib.style.use('ggplot')
from scipy import stats

#Pearson correlation coefficient measures the linear relationship between two datasets
#Correlations of -1 or +1 imply an exact linear relationship
#Positive correlations imply that as x increases, so does y
#Negative correlations imply that as x increases, y decreases

print(pearsonr(data['ave_hourly_rate'], data['fill_rate']))

colors = (0,0,0)
area = np.pi*3
plt.subplot(2, 1, 1) 
x1 = data['ave_hourly_rate']
y1 = data['fill_rate']
plt.scatter(x1, y1, s=area, c=colors, alpha=0.8)
plt.title('Scatter plot showing correlation between ave_hourly_rate and fill_rate')
plt.xlabel('Average hourly_rate')
plt.ylabel('Fill rate')

plt.subplot(2, 1, 2)
s = data[['ccg_id', 'fill_rate']]
plt.scatter(s['ccg_id'], s['fill_rate'], s=area, c=colors, alpha=0.8)
plt.title('Scatter plot showing correlation between ccg_id and fill_rate')
plt.xlabel('ccg_id')
plt.ylabel('Fill rate')
plt.show()

correlationMatrix = data.corr().abs()
sns.heatmap(correlationMatrix,annot=True)
plt.show()

#################################################################################################
#Calculate r-squared value for the correlation between average hourly rate and fill rate
slope, intercept, r_value, p_value, std_err = stats.linregress(data['ave_hourly_rate'], data['fill_rate'])

print("r-squared:", r_value**2)

###########################################################################################################
#visualize how the fill rate varies for different ccg_id
s.diff().hist(color='k', alpha=0.5, bins=10)
plt.show()

s['fill_rate'].plot.hist(alpha=0.5)
plt.show()

###############################################################################################
#visualize how number of posted jobs varies with number of filled jobs
c = pd.concat([x, y_comp], axis=1)
c['n_posted_jobs'] = c['posted_datetime']
c['n_filled_jobs'] = c['status']
del c['posted_datetime']
del c['status']
c.dropna()
c.diff().hist(color='k', alpha=0.5, bins=10)
plt.show()









