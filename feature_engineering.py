import pandas as pd
import numpy as np

'''
processed into py file after submission (no other changes but making into functions)
for original code, check out dacon_final.ipynb file
'''


def sum_of_squares(train, test):
  # The sum of the squares of acceleration and gyroscope sensor data
  train['acc_t']  = (train['acc_x']**2+train['acc_y']**2+train['acc_z']**2)**(1/2)
  test['acc_t']  = (test['acc_x']**2+test['acc_y']**2+test['acc_z']**2)**(1/2)
  train['gy_t']  = (train['gy_x']**2+train['gy_y']**2+train['gy_z']**2)**(1/2)
  test['gy_t']  = (test['gy_x']**2+test['gy_y']**2+test['gy_z']**2)**(1/2)
  
  return train, test


def variation(df, row, timestep, new_col, ref_col):
  # The amount of change from the previous timestep
  for i in range (row):
    for j in range(timestep):
        if j != 0:
            train_df[i][j][new_col] = train_df[i][j][ref_col] - train_df[i][j-1][ref_col]
            train_df[i][j][new_col+1] = train_df[i][j][ref_col+1] - train_df[i][j-1][ref_col+1]
            train_df[i][j][new_col+2] = train_df[i][j][ref_col+2] - train_df[i][j-1][ref_col+2] 
        if j == 0:
            train_df[i][j][new_col] = train_df[i][j][ref_col]
            train_df[i][j][new_col+1] = train_df[i][j][ref_col]
            train_df[i][j][new_col+2] = train_df[i][j][ref_col]
            
   return df
      
  
if __name__ == '__main__':
  feature_names = ['acc_x', 'acc_y', 'acc_z', 'gy_x', 'gy_y', 'gy_z', 'acc_t', 'gy_t', 'gy_x_diff', 'gy_y_diff', 'gy_z_diff']
  train_features, test_features = sum_of_squares(train_features, test_features)
  
  # gyroscope variation (train data)
  train_features['gy_x_diff'] = 0
  train_features['gy_y_diff'] = 0
  train_features['gy_z_diff'] = 0
  train_df = train_features.iloc[:, 2:].to_numpy()
  train_df = train_df.reshape(3125, 600, 11)
  print(train_df.shape)
  train_df = variation(train_df, train_df.shape[0], train_df.shape[1], 8, 3)
  print("output after feature engineering (train data)")
  pd.DataFrame(train_df.reshape(train_df.shape[0]*train_df.shape[1], train_df.shape[2]), columns=feature_names)
  
  # gyroscope variation (test data)
  test_features['gy_x_diff'] = 0
  test_features['gy_y_diff'] = 0
  test_features['gy_z_diff'] = 0
  test_df = test_features.iloc[:, 2:].to_numpy()
  test_df = test_df.reshape(782, 600, 11)
  print(test_df.shape)
  test_df = variation(test_df, test_df.shape[0], test_df.shape[1], 8, 3)
  print("output after feature engineering (test data)")
  pd.DataFrame(test_df.reshape(test_df.shape[0]*test_df.shape[1], test_df.shape[2]), columns=feature_names)
