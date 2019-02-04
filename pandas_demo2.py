import pandas as pd

data_set = pd.read_csv("C:/Users/jhunjhun/Downloads/update.csv")
print(data_set[['Country','Unit Cost','Units Sold']])
print(data_set['Total Cost'])
print(data_set[['Country','Units Sold']][data_set['Units Sold']==data_set['Units Sold'].max()])
