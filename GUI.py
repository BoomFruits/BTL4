import tkinter
from tkinter import StringVar
from tkinter import ttk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from LinearRegression import model
from RidgeRegression import model as ridgeModel
from LassoRegression import model as lassoModel
from MLPregressor import model as mlpregressorModel
from MLPregressor import r2,mae,nse,rmse
window = tkinter.Tk()#This method creates a blank window with close, maximize, and minimize buttons on the top as a usual GUI should have
window.title("LinearRegression GUI")
encode_values = {
    "Extracurricular Activities":{"Yes":1,"No":0},
}
myHeader = ["Hours Studied","Previous Scores","Extracurricular Activities","Sleep Hours","Sample Question Papers Practiced"]
varList = [StringVar() for i in range(0,len(myHeader))]
for i in range(0,len(myHeader)):
    if (i != 2):
        tkinter.Label(window,text=myHeader[i]).grid(row=i,column=0)
        tkinter.Entry(window,textvariable=varList[i]).grid(row=i,column=1)
    else:
        tkinter.Label(window,text=myHeader[i]).grid(row=i,column=0)
        cbchoosen = ttk.Combobox(window, values=['Yes','No'],state='readonly' ,width = 17, textvariable = varList[i]).grid(column=1,row=i)
def get_input_data():
    res = []
    for i in range(0,len(varList)):
        if(i == 2):
            value = varList[i].get()
            if(value == "Yes"):
                varList[i].set("1")
            else:
                varList[i].set("0")
        value = int(varList[i].get())       
        res.append(value)
        print(value)
    rowData = pd.DataFrame([res])
    rowData.columns = myHeader
    rowDataEncode = rowData.replace(encode_values)
    print(rowDataEncode)
    return rowDataEncode
def retrive_input_lr():
    rowDataEncode = get_input_data()
    y_predict = model.predict(rowDataEncode)
    lbl =tkinter.Label(window, text = 'Điểm dự đoán của học sinh đó là '+str(y_predict)).grid(row=17,column=0,padx=20,pady=20)
def retrive_input_ridge():
    rowDataEncode = get_input_data()
    y_predict = ridgeModel.predict(rowDataEncode)
    lbl =tkinter.Label(window, text = 'Điểm dự đoán của học sinh đó là '+str(y_predict)).grid(row=17,column=0,padx=20,pady=20)
def retrive_input_lasso():
    rowDataEncode = get_input_data()
    y_predict = lassoModel.predict(rowDataEncode)
    lbl =tkinter.Label(window, text = 'Điểm dự đoán của học sinh đó là '+str(y_predict)).grid(row=17,column=0,padx=20,pady=20)
def retrive_input_mlp():
    rowDataEncode = get_input_data()
    y_predict = mlpregressorModel.predict(rowDataEncode)
    lbl =tkinter.Label(window, text = 'Điểm dự đoán của học sinh đó là '+str(y_predict)).grid(row=17,column=0,padx=20,pady=20)
tkinter.Button(window,text="Dự đoán LinearRegression",command=retrive_input_lr).grid(row=15,column=0)
tkinter.Button(window,text="Dự đoán RidgeRegression",command = retrive_input_ridge).grid(row=15,column=1,padx=20)
tkinter.Button(window,text="Dự đoán LassoRegression",command=retrive_input_lasso).grid(row=16,column=0,pady=20)
tkinter.Button(window,text="Dự đoán Multi Layer Perceptron Regression",command = retrive_input_mlp).grid(row=16,column=1,padx=20,pady=20)
tkinter.Label(window,text = "Độ đo của MLPregressor là tốt nhất").grid(row=0,column=2,padx=10)
tkinter.Label(window,text = "R^2 score: "+str(r2)).grid(row=1,column=2,padx=10)
tkinter.Label(window,text = "MAE score: "+str(mae)).grid(row=2,column=2,padx=10)
tkinter.Label(window,text = "NSE score: "+str(nse)).grid(row=3,column=2,padx=10)
tkinter.Label(window,text = "RMSE score: "+str(rmse)).grid(row=4,column=2,padx=10)
# tkinter.Label(window,text = "Độ đo Silhouette: "+str(kmsSil_sc)).grid(row =0,column=2 , padx=10)
# tkinter.Label(window,text = "Độ đo Davies-Bouldin.: "+str(kmsDav_sc)).grid(row = 1 ,column=2,padx=10)
window.mainloop()
