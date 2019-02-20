'''
Output:- 
                          OLS Regression Results                            
==============================================================================
Dep. Variable:          Total Revenue   R-squared:                       0.980
Model:                            OLS   Adj. R-squared:                  0.979
Method:                 Least Squares   F-statistic:                 1.587e+04
Date:                Wed, 20 Feb 2019   Prob (F-statistic):               0.00
Time:                        12:37:49   Log-Likelihood:                -13686.
No. Observations:                1000   AIC:                         2.738e+04
Df Residuals:                     996   BIC:                         2.740e+04
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       1.094e+04   1.96e+04      0.558      0.577   -2.75e+04    4.94e+04
Units Sold    34.8765      3.384     10.307      0.000      28.237      41.517
Unit Cost    -39.8842     74.770     -0.533      0.594    -186.608     106.840
Total Cost     1.2258      0.013     95.404      0.000       1.201       1.251
==============================================================================
Omnibus:                      119.619   Durbin-Watson:                   1.924
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              284.366
Skew:                           0.669   Prob(JB):                     1.78e-62
Kurtosis:                       5.244   Cond. No.                     4.34e+06
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 4.34e+06. This might indicate that there are
strong multicollinearity or other numerical problems.

'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data_set = pd.read_csv('C:/Users/jhunjhun/Downloads/update.csv',index_col=0)
X = data_set[['Units Sold','Unit Cost','Total Cost']]
y = data_set['Total Revenue']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X, y)

y_pred = regressor.predict(X_test)
#print(y_pred)

#print(regressor.coef_)
import statsmodels.api as sm

X = sm.add_constant(X)
est = sm.OLS(y,X).fit()
print(est.summary())
