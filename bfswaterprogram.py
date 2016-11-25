import pandas  as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
import numpy as np
from datetime import timedelta, datetime
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
from math import sqrt


df1 = pd.read_excel('data/changrenchiao waterlevel 14-15.xlsx')
df2 = pd.read_excel('data/wulilin waterlevel 14-15.xlsx')
df3 = pd.read_excel('data/chiaoto rainfall14-15.xlsx')
df4 = pd.read_excel('data/das rainlevel14-15.xlsx')

columns_name = ['time','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24']

df1.columns= columns_name
df2.columns= columns_name
df3.columns= columns_name
df4.columns= columns_name

df1.fillna(method='ffill', inplace=True)
df2.fillna(method='ffill', inplace=True)
df3.fillna(method='ffill', inplace=True)
df4.fillna(method='ffill', inplace=True)

data = {'time':[],'crc':[], 'wll':[], 'ctrf':[], 'dasrf':[]}

starttime = datetime(2014, 1, 1,0,0,0)
interval = timedelta(hours=1)

for i in range(len(df1.index)):
	for item in df1:
		if item == 'time':
			pass
		else:
			data['time'].append(starttime)
			data['crc'].append(df1[item][i])
			data['wll'].append(df2[item][i])
			data['ctrf'].append(df3[item][i])
			data['dasrf'].append(df4[item][i])
			starttime += interval

df = pd.DataFrame(data)
df.set_index('time', inplace=True)

df[df < 0] = 0

for i in range(1,8):
	df['ctrf'+str(i)] = df['ctrf'].shift(i)
	#df['wll_feature'+str(i)] = df['wll'].shift(i)
df['wll_feature1'] = df['wll'].shift(1)
df['wll_feature2'] = df['wll'].shift(2)
df['wll_feature3'] = df['wll'].shift(3)
df.fillna(0, inplace=True)

ax1 = plt.subplot2grid((2,2),(0,0))
ax2 = plt.subplot2grid((2,2),(1,0), sharex=ax1)
ax3 = plt.subplot2grid((2,2),(0,1))
ax4 = plt.subplot2grid((2,2),(1,1))

df[['crc','wll']].plot(ax= ax1, linewidth = 1, color=['r','b'])
df['ctrf'].plot(ax= ax2, label="rf1", linewidth = 1, color='g')

ax1.set_ylabel('waterlevel')
ax2.set_ylabel('CT rainfall')

df_corr = df.corr()
print (df_corr['wll'])

df_feature = df[['wll_feature1','ctrf2','ctrf3','ctrf4','ctrf5','ctrf6','ctrf7']]
X = np.array(df_feature)
#X = preprocessing.scale(X)
y = np.array(df['wll'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.1)

clf = LinearRegression(n_jobs=10)
#clf = svm.SVR(kernel='poly')
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)

print ('two hours in advance')
print ('accuracy,',accuracy)

def read_predict_file(file_name, start, stop_condition):
	df = pd.read_csv(file_name)
	df.columns = ['time2','waterlevel','rainfall']
	df.drop(['time2'], 1, inplace=True)
	df[df < 0] = 0
	df['daysvalue'] = df['rainfall'].rolling(window = stop_condition).mean()
	df.fillna(0.00000001,inplace=True)
	#print (df)
	df = df[df['daysvalue']>0]
	#print (df)
	time_index = []
	startpoint = datetime(start[0],start[1],start[2],start[3],0,0)
	for i in range(len(df.index)):
		time_index.append(startpoint)
		startpoint += interval
	df['time'] = pd.Series(time_index, index = df.index)
	df.set_index('time', inplace=True)
	df[df < 0] = 0
	for i in range(2,8):
		df['rainfall'+str(i)] = df['rainfall'].shift(i)

	df['waterlevel1'] = df['waterlevel'].shift(1)
	df['waterlevel2'] = df['waterlevel'].shift(2)
	df['waterlevel3'] = df['waterlevel'].shift(3)
	df.dropna(inplace=True)


	return df

df_0610 = read_predict_file('0610rain.csv',[2016,6,10,0], 48)
df_0706 = read_predict_file('0706rain.csv',[2016,7,6,0], 48)


def predict_df(df):
	df_feature_predict_me = df[['waterlevel1','rainfall2','rainfall3','rainfall4','rainfall5','rainfall6','rainfall7']]
	X_predict_me = np.array(df_feature_predict_me)
	forecast_set = clf.predict(X_predict_me)
	df['predict_wl'] = pd.Series(forecast_set, index = df.index)
	df_corr = df.corr()
	error_sum = 0
	error_com_sum = 0
	error_square_sum = 0
	com_error_square_sum = 0
	for i in range(len(df.index)):
		error_square = ((df['predict_wl'][i] - df['waterlevel'][i]) ** 2)
		error_square_sum += error_square
		com_error_square = ((df['waterlevel2'][i] - df['waterlevel'][i]) ** 2)
		com_error_square_sum += com_error_square

	rmse = sqrt(error_square_sum / len(df.index))
	brmse = sqrt(com_error_square_sum / len(df.index))
	print ('root mean square error', rmse)

print ('6/10')
predict_df(df_0610)
df_0610[['waterlevel','predict_wl']].plot(ax= ax3, linewidth = 1)
ax3.set_ylabel('0610_Waterlevel')

print ('7/6')
predict_df(df_0706)
df_0706[['waterlevel','predict_wl']].plot(ax= ax4, linewidth = 1)
ax4.set_ylabel('0706_Waterlevel')

plt.show()