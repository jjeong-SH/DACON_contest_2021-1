import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

'''
for visualized outputs, check out dacon_final.ipynb file
'''


def plot_variable_distributions(train_x):
	Longtrain_x = train_x.reshape((train_x.shape[0] * train_x.shape[1], train_x.shape[2]))
	plt.figure()
	xaxis = None
	for i in range(Longtrain_x.shape[1]):
		ax = plt.subplot(Longtrain_x.shape[1], 1, i+1, sharex=xaxis)
		ax.set_xlim(-100, 100)
		if i == 0:
			xaxis = ax
		plt.hist(Longtrain_x[:, i], bins=100)
	plt.show()

  
def standardize(train_x, test_x):
  Longtrain_x = train_x.reshape((train_x.shape[0] * train_x.shape[1], train_x.shape[2]))
  Longtest_x = test_x.reshape((test_x.shape[0] * test_x.shape[1], test_x.shape[2]))

  scaler=StandardScaler()
  scaler.fit(Longtrain_x)
  Flattrain_x=scaler.transform(Longtrain_x)
  Flattest_x=scaler.transform(Longtest_x)

  train_scaled=pd.DataFrame(data=Flattrain_x, columns=feature_names)
  print('scaling 후 feature 평균')
  print(train_scaled.mean())
  print('\nscaling 후 feature 분산')
  print(train_scaled.var())

  return Flattrain_x, Flattest_x


if __name__ == '__main__':
  plot_variable_distributions(train_df)
  train_st, test_st = standardize(train_df, test_df)
  train_st = pd.DataFrame(data=train_st, columns=feature_names)
  test_st = pd.DataFrame(data=test_st, columns=feature_names)
