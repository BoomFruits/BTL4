import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
df = pd.read_csv(".vscode/.vscode/ThiHetMon/Student_Performance.csv")
encode_values = {
    "Extracurricular Activities":{"Yes":1,"No":0},
}
data = df.replace(encode_values)
dt_train,dt_test = train_test_split(data , test_size=0.3,shuffle=True)
X_train = dt_train.iloc[:,:5]
y_train = dt_train.iloc[:,5]
X_test = dt_test.iloc[:,:5]
y_test = dt_test.iloc[:,5]
# from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
model = Lasso(alpha=2.0,tol=1e-4,fit_intercept=False).fit(X_train,y_train)
y_pred_test = model.predict(X_test)
print(y_pred_test)
print(y_test)
def NSE(y_test, y_pred):
    return (1 - (np.sum((y_pred - y_test) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)))
print("LassoRegression scores")
print('R^2: ', r2_score(y_test, y_pred_test))
print('NSE: ',NSE(y_test,y_pred_test))
print('MAE: ',mean_absolute_error(y_test, y_pred_test))
print('RMSE: ', np.sqrt(mean_squared_error(y_test, y_pred_test)),'\n')
