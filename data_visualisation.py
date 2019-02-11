import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data_set = pd.read_csv('C:/Users/jhunjhun/Downloads/update.csv')
demo_data = data_set.groupby('Region').min()
y = data_set['Region'].unique()
x = demo_data['Units Sold']
print(x)
index = np.arange(len(y))
plt.xticks(index, y, fontsize=10, rotation=90)
#plt.plot(y,x)
print(plt.boxplot(x))
