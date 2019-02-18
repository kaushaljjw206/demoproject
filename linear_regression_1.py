''' Here linear regression line for both the training data and the alidation data is same'''

'''Output-
Coefficients: [1.25425323]
Intercept: 146047.9770499831
First r2 score :  0.9730782173654317
Second r2 score :  0.9756372137429781
Coefficients: [1.25425323]
Intercept: 146047.9770499831

'''


''' Output Graph- images/linear_regression_1.png 
                  images/linear_regression_2.png'''

import pandas as pd
import matplotlib.pyplot as plt
import random
data_set = pd.read_csv('C:/Users/jhunjhun/Downloads/update.csv',index_col=0)
main_data = data_set.iloc[0:800]
#main_data = data_set.iloc[0:750]
#validate_data = data_set.iloc[751:1000]
#X = main_data['Total Cost']
#y = main_data['Total Revenue']
#validate_x = validate_data['Total Cost']
#validate_y = validate_data['Total Revenue']
n = 1000 #number of rows in the file
s = 100 #desired sample size
skip = random.sample(range(n),n-s)
df = pd.read_csv('C:/Users/jhunjhun/Downloads/update.csv',index_col=0)
#validate_data = data_set.iloc[901:1000]
X = main_data['Total Cost']
y = main_data['Total Revenue']
validate_x = df['Total Cost']
validate_y = df['Total Revenue']
#print(X)
#print(validate_x)
#print(validate_x)
#validate_x = data_set['Total Cost'].tail(19)
#validate_y = data_set['Total Revenue'].tail(19)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

regression = LinearRegression()
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.33, random_state=20)
X_train = X_train.values.reshape(-1,1)
#print(X_train)
#print(y_train)
#print(X_test)
#print(y_test)
regression.fit(X_train,y_train)
y_pred = regression.predict(X_test.values.reshape(-1,1))
#print(y_pred)
# regression coefficients
print('Coefficients:',regression.coef_)
# regression intercept
print('Intercept:' ,regression.intercept_)
plt.scatter(X_test,y_test,color='blue',marker='x')
plt.plot(X_test,y_pred,color='green')
# Results of Linear Regression. 
#from sklearn.metrics import mean_squared_error 
#mse = mean_squared_error(y_test, y_pred) 
#print("Mean Square Error : ", mse)
#from sklearn.metrics import mean_absolute_error
#mae = mean_absolute_error(y_test,y_pred)
#print("Mean absolute error is : ",mae)
from sklearn.metrics import r2_score
r2score = r2_score(y_test, y_pred) 
print("First r2 score : ", r2score)
#r2score_validate = r2_score(validate_y, y_pred)
#print("r2 score of validation data : ", r2score_validate)
regression_model = LinearRegression()
regression_model.fit(X_train,y_train)
y_pred_validate = regression_model.predict(validate_x.values.reshape(-1,1))
plt.scatter(validate_x,validate_y,color='green',marker='.')
plt.plot(validate_x,y_pred_validate,color='black')
r2score_validate = r2_score(validate_y,y_pred_validate)
print("Second r2 score : ", r2score_validate)
plt.xlabel('Total Cost')
plt.ylabel('Total Revenue')
plt.title('Total Cost v/s Total Revenue')
# regression coefficients
print('Coefficients:',regression_model.coef_)
# regression intercept
print('Intercept:' ,regression_model.intercept_)
