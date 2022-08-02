###################################################
# Coder: 1LT Chris Aliperti and CDT Joshua Wong
# Title: Model GUI
# Date: 07JUL2022
# Version: 1.1
###################################################
# This GUI is used to demonstrate a capability of a predictive model. It allows Commanders to input certain data about a 'new'
# crew and get a predicted score based on one of the models created in M1Models.py.
# Given the scarcity of data these models are not very accurate yet
# (The model building and performance is a discussion for another document).

# Functions for executing GUI specific actions (ie filling text boxes with buttons) are not explicitly 
# explained for neatness sake.

#%% Import Libraries 
import pandas as pd
import tkinter as tk
import M1Models as mod

#%% Setup GUI
root= tk.Tk()

canvas = tk.Canvas(root, width = 1200, height = 500)
canvas.configure(background = 'blue')
canvas.pack()

#%% Build GUI body 
# Label 
label = tk.Label(root, text='Predict GTVI Score')
label.config(font=('helvetica', 14))
canvas.create_window(600, 25, window=label)

# Time Input
label1 = tk.Label(root, text='Time in Sim (hrs):')
label1.config(font=('helvetica', 10))
canvas.create_window(200, 100, window=label1)

entry1 = tk.Entry(root)
canvas.create_window(200, 140, window=entry1)

# Tot Hits Input
label2 = tk.Label(root, text='Total Hits in Sim:')
label2.config(font=('helvetica', 10))
canvas.create_window(400, 100, window=label2)

entry2 = tk.Entry(root)
canvas.create_window(400, 140, window=entry2)

# AVG Time To ID Input
label3 = tk.Label(root, text='AVG Time to ID (s):')
label3.config(font=('helvetica', 10))
canvas.create_window(600, 100, window=label3)

entry3 = tk.Entry(root)
canvas.create_window(600, 140, window=entry3)

# AVG Time Of Eng Input
label4 = tk.Label(root, text='AVG Time Of Eng (s):')
label4.config(font=('helvetica', 10))
canvas.create_window(800, 100, window=label4)

entry4 = tk.Entry(root)
canvas.create_window(800, 140, window=entry4)

# Prediction Button
def getPred():  

    df = pd.read_csv('M1_clean.csv')

    regr1, _ = mod.m1_reg_model(df)
    predVals = [entry1.get(),entry2.get(),entry3.get(),entry4.get()]
    predVals = [float(x) for x in predVals]
    
    label5 = tk.Label(root, text='The Predicted Score is:') 
    label5.config(font=('helvetica', 10))
    canvas.create_window(600, 210, window=label5)

    label6 = tk.Label(root, text=f'{mod.m1_reg_predict(regr1,predVals)[0][0]:.0f}') 
    label6.config(font=('helvetica', 10))
    canvas.create_window(600, 230, window=label6)

#%% Exit Button
def Quit_Func():
    canvas.destroy()

button1 = tk.Button(text='Get Prediction', command=getPred)
canvas.create_window(600, 180, window=button1)
# Quit = tk.Button(canvas, text = "QUIT", bg = 'red', command = Quit_Func).grid(row=310, column=10)

#%% Execute GUI
root.mainloop()

