
''' Prediction for total revenue by using total cost '''

'''Output:-
Enter value2233445
(1,)
(1, 1)
[2947353.58406281]
'''

tc = int(input("Enter value"))
#from numpy import array
#tc = [tc]
#t = array(tc)
#print(t.shape)
#t = t.reshape((t.shape[0],1))
#print(t.shape)
#t = t.astype(np.float64)
total_revenue = regression_model.predict([[tc]])
print(total_revenue)
