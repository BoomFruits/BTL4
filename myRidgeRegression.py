import numpy as np   
class RidgeRegression():
    def __init__(self,alpha=1.0,learning_rate = 0.01,iter = 1000,tol=1e-4): 
        self.alpha = alpha #tham số điều chỉnh độ lớn của regularization term
        self.iter = iter # số lần lặp
        self.learning_rate = learning_rate # tốc độ học của GD
        self.tol = tol # Ngưỡng hội tụ của GD
        self.W = None #Vector trọng số
        self.bias = None
    def fit(self,X,Y): #X_train(8000,5) , y_train
        n_samples,n_features = X.shape #n_sample mẫu và 5 thuộc tính
        self.W = np.zeros(n_features) #khởi toạ vector trọng số W = [0,0,0,0,0]
        self.bias = 0 # bias = 0
        #Cập nhập lại trọng số bias và w cho mô hình
        regularization_matrix = self.alpha*np.sum(self.W**2) #np.eye là ma trận đường chéo chính
        for i in range(self.iter):
            y_pred = np.dot(X,self.W) + self.bias
            dw = (1 / n_samples) * (np.dot(X.T, y_pred - Y) + np.dot(regularization_matrix, self.W))# Tính đạo hàm riêng của hàm mất mát theo W
            db = (1/n_samples)*np.sum(y_pred-Y) # dot là phép nhân ma trận giữa X.T và hiệu của nhãn dự đoán và nhãn thực tế 
            # tính tổng tất cả các phần tử hiệu nhãn dự đoán - nhãn thực tế
            #Cập nhập trọng số theo gradient descent
            self.W = self.W - self.learning_rate*dw 
            self.bias = self.bias - self.learning_rate*db
            if np.linalg.norm(self.learning_rate*dw) < self.tol:
                break
        return self # trả về hàm tuyến tính với trọng số W và bias
    def predict( self, X ) : 
        y_pred =  np.dot( X,self.W ) + self.bias # tính nhãn dự đoán theo hàm tuyến tính
        return y_pred
# Code này chạy ok nhưng kết quả đang bị nan chả hiểu kiểu gì