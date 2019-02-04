import pandas as pd

data_set = pd.read_csv('C:/Users/jhunjhun/Downloads/update.csv', index_col=0)
mean_value = data_set['Units Sold'].mean()
print(data_set['Units Sold'])
print(mean_value)
