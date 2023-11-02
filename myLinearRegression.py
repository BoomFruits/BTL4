import numpy as np   
class LinearRegression():
    def __init__(self,learning_rate = 0.0001,tol=1e-5,iter=10000):
        self.learning_rate = learning_rate #Tốc độ học của GD
        self.iter = iter # số lần lăpk
        self.W = None 
        self.bias = None
        self.tol = tol
    def fit(self,X,Y): #X_train(8000,5) , y_train
        n_samples,n_features = X.shape #n_sample mẫu và 5 thuộc tính
        self.W = np.zeros(n_features) #khởi toạ vector trọng số W = [0,0,0,0,0]
        self.bias = 0 # bias = 0
        #Cập nhập lại trọng số bias và w cho mô hình
        last_cost = float('inf')
        for i in range(self.iter):
            y_pred = np.dot(X,self.W) + self.bias
            dw = (1/n_samples)*np.dot(X.T,(y_pred-Y)) # Tính đạo hàm riêng của hàm mất mát theo W
                                # dot là phép nhân ma trận giữa X.T và hiệu của nhãn dự đoán và nhãn thực tế 
            db = (1/n_samples)*np.sum(y_pred-Y)# 2tính tổng tất cả các phần tử hiệu nhãn dự đoán - nhãn thực tế
            cost = np.sum(np.square(y_pred-Y))/(2*n_samples)
            #Cập nhập trọng số theo gradient descent
            self.W = self.W - self.learning_rate*dw 
            self.bias = self.bias - self.learning_rate
            # print(i)
            if last_cost - cost < 1e-4: # điều kiện dừng
                break
            else:
                last_cost = cost
        return self # trả về hàm tuyến tính với trọng số W và bias
    def predict( self, X ) : 
        y_pred =  np.dot( X,self.W ) + self.bias # tính nhãn dự đoán theo hàm tuyến tính
        return y_pred
    # def fit(self,X,Y):
    #     self.w = np.linalg.pinv(X@X.T)@X@Y
    #     return self
    # def predict(self,X):
    #     y_predict = X.T@self.w
    #     return y_predict
