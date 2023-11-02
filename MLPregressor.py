import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
df = pd.read_csv(".vscode/.vscode/ThiHetMon/Student_Performance.csv")
print(df.isna().sum())
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
from sklearn.neural_network import MLPRegressor
# model = model = MLPRegressor(max_iter=5000).fit(X_train,y_train)
# params = {'alpha':[0.1, 0.5, 0.8],
#              'hidden_layer_sizes': [(20,20),(40,40,40),(80,80),(70,70),(50,50),(60,60)]}
# mlp_cv_model = GridSearchCV(model, params, cv = 5)
# mlp_cv_model.fit(X_train, y_train)
# print(mlp_cv_model.best_params_)
# alpha = 0.5 , hidden_layer_sizes = (20,20)
model = MLPRegressor(alpha=0.5,hidden_layer_sizes=(20,20),max_iter=5000).fit(X_train,y_train)
y_pred_test = model.predict(X_test)
print(y_pred_test)
print(y_test)
def NSE(y_test, y_pred):
    return (1 - (np.sum((y_pred - y_test) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)))
r2 = r2_score(y_test, y_pred_test)
nse = NSE(y_test,y_pred_test)
mae = mean_absolute_error(y_test, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
print("MLPregressor scores")
print('R^2: ', r2)
print('NSE: ',nse)
print('MAE: ',mae)
print('RMSE: ', rmse,'\n')

