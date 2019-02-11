
#Here I am comparing total revenue with total cost present in csv file by plotting scatter plot


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data_set = pd.read_csv('C:/Users/jhunjhun/Downloads/update.csv',index_col=0)
plt.scatter(data_set['Total Revenue'],data_set['Total Cost'], marker='x')
plt.xlabel('Total Revenue')
plt.ylabel('Total Cost')
plt.title('Total Revenue V/S Total Cost')
