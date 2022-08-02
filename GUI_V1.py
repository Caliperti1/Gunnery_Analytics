###################################################
# Coder: 1LT Chris Aliperti and CDT Joshua Wong
# Title: Gunnery Dashboard GUI
# Date: 07JUL2022
# Version: 1.1
###################################################
# GUI_V1 generates a prototype Commander's dashboard that allows COmmanders to interface
# with the data by selecting a crew of their choice and viewing all associated data. GUI also
# contains various plots to visualize the crew's performance compared to peers.
# GUI_V1 is also valuable as it incorporates and runs the other files and funcitons in this project
# but does not require any interaction with raw code if not desired. This is very useful for 
# demonstrating the project's capabilites to non-technical audiences.

# Functions for executing GUI specific actions (ie filling text boxes with buttons) are not explicitly 
# explained for neatness sake.

#%% Disable Warnings 
import warnings
warnings.filterwarnings("ignore")

#%% Import Libraries 
import tkinter 
from tkinter import *
from tkinter import ttk

import numpy as mp
import matplotlib as plt
import seaborn as sns
import pandas as pd

from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

#%% Import Functions 
import MainFunction as MF
import DataVisuals as DV 

#%% Import Test Dataframe 
M1 = pd.DataFrame(pd.read_csv('M1_Clean.csv'))

#%% Set up GUI
# initiate Program 
Dash = Tk()

# Set Geometry 
Dash.geometry("1250x500")

# Set Background Color 
Dash.configure(background = 'blue')

# Set Frame
frame = Frame(Dash, relief = RIDGE,bg = "blue")
frame.grid()

# Ttile 
Dash.title("Commander's Gunnery Dashboard V1.0")

# Run Main 
MF.main()

#%% Build Crew Input  

# Unit Select  
BNs = ['1-64 AR','3-69 AR','2-7 IN']
        
Unit_Selected = StringVar() 

Unit = OptionMenu(Dash,Unit_Selected, *BNs).place(x = 25, y = 25)

Unit_Label =Label(frame,text = "Select your Battalion",bg = "white",fg="black").grid(row = 1, column = 1)

# Company  Select  
Cos = ['A','B','C']
        
Co_Selected = StringVar() 

Co = OptionMenu(Dash,Co_Selected, *Cos).place(x = 145, y = 25)

Co_Label =Label(frame,text = "Select your Company",bg = "white",fg="black").grid(row = 1, column = 2)

# TrackTrack Select
Tracks = ['11','12','13','14','21','22','23','24','31','32','33','34','65','66']
   
Track_Selected = StringVar() 

Track = OptionMenu(Dash,Track_Selected, *Tracks).place(x = 285, y = 25)

Track_Label =Label(frame,text = "Select your crew",bg = "white",fg="black").grid(row = 1, column = 3)
 
# Display Selected Crew 
Crew_Selected = Text(frame, height = 1, width = 15, wrap = WORD)
Crew_Selected.grid(row = 1, column = 6)
Crew_Selected_Label = Label(frame,text = "Selected Crew",bg = "brown",fg="white").grid(row = 1, column = 5)

# Funciton to Set Crew Input 
Master_Key =''
Crew_Data = pd.DataFrame()

def Build_Crew():
    global Master_Key
    global Crew_Data
    global MK_Dict
    
    # Clear Crew Display Panel
    Crew_Selected.delete(0.0, END)
    
    # Get Inputs from menu
    Track_Input = Track_Selected.get()
    Unit_Input = Unit_Selected.get()
    Co_Input = Co_Selected.get()
    
    # Combine inputs into master key variable 
    Master_Key = str(str(Unit_Input) + ' ' + str(Co_Input) + str(Track_Input))
    
    # display Master key in selected Crew panel
    Crew_Selected.insert(END, Master_Key)
    
    # Build Dictionary of MAster Keys to Indicies 
    MK_Dict = {}
    for i in range(len(M1['Master Key'])):
        MK_Dict[M1.loc[i,'Master Key']] = i    
    
    # find index of Master Key 
    Crew_Data = M1.loc[M1['Master Key'] == Master_Key]    
    

#%% Build "snapshot" outputs and labels 

# Make spacers 
Row2_Spacer = Label(frame, bg = 'blue').grid(row = 2, column = 10)
Row3_Spacer = Label(frame, bg = 'blue').grid(row = 3, column = 10)
Row4_Spacer = Label(frame, bg = 'blue').grid(row = 4, column = 10)
Row7_Spacer = Label(frame, bg = 'blue').grid(row = 7, column = 10)

# TC Name 
TC_Name = StringVar()  
TC_Name_Label = Label(frame, text = 'TC Name', bg = 'white', fg = 'black').grid(row = 5, column = 1)
TC_Name_Output = Text(frame, height = 1, width  = 15, wrap = WORD)
TC_Name_Output.grid(row = 6, column = 1)

def TC_Name_Fill():
    TC_Name_Output.delete(1.0, END)
    TC_Name = Crew_Data.loc[MK_Dict[Master_Key],'tc_name']  # THIS WILL BE PULLED FROM M1 DF LATER 
    TC_Name_Output.insert(END, TC_Name)
    
# GNR Name 
GNR_Name = StringVar() 
GNR_Name_Label = Label(frame, text = 'GNR Name', bg = 'white', fg = 'black').grid(row = 5, column = 2)
GNR_Name_Output = Text(frame, height = 1, width  = 15, wrap = WORD)
GNR_Name_Output.grid(row = 6, column = 2)

def GNR_Name_Fill():
    GNR_Name_Output.delete(1.0, END)
    GNR_Name = Crew_Data.loc[MK_Dict[Master_Key],'gn_name']  # THIS WILL BE PULLED FROM M1 DF LATER 
    GNR_Name_Output.insert(END, GNR_Name)
    
# LDR Name 
LDR_Name = StringVar()  
LDR_Name_Label = Label(frame, text = 'LDR Name', bg = 'white', fg = 'black').grid(row = 5, column = 3)
LDR_Name_Output = Text(frame, height = 1, width  = 15, wrap = WORD)
LDR_Name_Output.grid(row = 6, column = 3)

def LDR_Name_Fill():
    LDR_Name_Output.delete(1.0, END)
    LDR_Name = Crew_Data.loc[MK_Dict[Master_Key],'ldr_name']  # THIS WILL BE PULLED FROM M1 DF LATER 
    LDR_Name_Output.insert(END, LDR_Name)
    
# Time crew has been together (Crew Age)
Crew_Time = StringVar()   
Crew_Time_Label = Label(frame, text = 'Crew Age', bg = 'white', fg = 'black').grid(row = 5, column = 4)
Crew_Time_Output = Text(frame, height = 1, width  = 15, wrap = WORD)
Crew_Time_Output.grid(row = 6, column = 4)

def Crew_Time_Fill():
    Crew_Time_Output.delete(1.0, END)
    Crew_Time = Crew_Data.loc[MK_Dict[Master_Key],'crew_time']  # THIS WILL BE PULLED FROM M1 DF LATER
    Crew_Time_Output.insert(END, Crew_Time)    
    
# Last Gunnery Date 
Gun_Date = StringVar()  
Gun_Date_Label = Label(frame, text = 'Last Gunnery Date', bg = 'white', fg = 'black').grid(row = 5, column = 5)
Gun_Date_Output = Text(frame, height = 1, width  = 15, wrap = WORD)
Gun_Date_Output.grid(row = 6, column = 5)

def Gun_Date_Fill():
    Gun_Date_Output.delete(1.0, END)
    Gun_Date = Crew_Data.loc[MK_Dict[Master_Key],'gun_date']  # THIS WILL BE PULLED FROM M1 DF LATER 
    Gun_Date_Output.insert(END, Gun_Date)    
    
# Last Gunnery Score
Gun_Score = StringVar() 
Gun_Score_Label = Label(frame, text = 'Last Gunnery Score', bg = 'white', fg = 'black').grid(row = 5, column = 6)
Gun_Score_Output = Text(frame, height = 1, width  = 15, wrap = WORD)
Gun_Score_Output.grid(row = 6, column = 6)

def Gun_Score_Fill():
    Gun_Score_Output.delete(1.0, END)
    Gun_Score = Crew_Data.loc[MK_Dict[Master_Key],'GTVI Score']   # THIS WILL BE PULLED FROM M1 DF LATER 
    Gun_Score_Output.insert(END, Gun_Score)    
    
# Last Gunnery Rating 
Gun_Rate = StringVar() 
Gun_Rate_Label = Label(frame, text = 'Last Gunnery Rating', bg = 'white', fg = 'black').grid(row = 5, column = 7)
Gun_Rate_Output = Text(frame, height = 1, width  = 15, wrap = WORD)
Gun_Rate_Output.grid(row = 6, column = 7)

def Gun_Rate_Fill():
    Gun_Rate_Output.delete(1.0, END)
    Gun_Rate = Crew_Data.loc[MK_Dict[Master_Key],'GTVI Rating']  # THIS WILL BE PULLED FROM M1 DF LATER 
    Gun_Rate_Output.insert(END, Gun_Rate)   
    
# Last Gunnery Score Percentile 
Gun_PCT = StringVar() 
Gun_PCT_Label = Label(frame, text = 'Score Percentile', bg = 'white', fg = 'black').grid(row = 5, column = 8)
Gun_PCT_Output = Text(frame, height = 1, width  = 15, wrap = WORD)
Gun_PCT_Output.grid(row = 6, column = 8)

def Gun_PCT_Fill():
    Gun_PCT_Output.delete(1.0, END)
    Gun_PCT = round(Crew_Data.loc[MK_Dict[Master_Key],'GTVI Percentile'],1)  # THIS WILL BE PULLED FROM M1 DF LATER 
    Gun_PCT_Output.insert(END, Gun_PCT)
    
# time in sim
AGTS_Time = StringVar() 
AGTS_Time_Label = Label(frame, text = 'AGTS Time', bg = 'white', fg = 'black').grid(row = 5, column = 9)
AGTS_Time_Output = Text(frame, height = 1, width  = 15, wrap = WORD)
AGTS_Time_Output.grid(row = 6, column = 9)

def AGTS_Time_Fill():
    AGTS_Time_Output.delete(1.0, END)
    AGTS_Time = round(Crew_Data.loc[MK_Dict[Master_Key],'Time'],1)  # THIS WILL BE PULLED FROM M1 DF LATER 
    AGTS_Time_Output.insert(END, AGTS_Time)

# time in sim percentile 
AGTS_Time_PCT = StringVar() 
AGTS_Time_PCT_Label = Label(frame, text = 'AGTS Time Percentile', bg = 'white', fg = 'black').grid(row = 5, column = 10)
AGTS_Time_PCT_Output = Text(frame, height = 1, width  = 15, wrap = WORD)
AGTS_Time_PCT_Output.grid(row = 6, column = 10)

def AGTS_Time_PCT_Fill():
    AGTS_Time_PCT_Output.delete(1.0, END)
    AGTS_Time_PCT = round(Crew_Data.loc[MK_Dict[Master_Key],'Time Percentile'],1)  # THIS WILL BE PULLED FROM M1 DF LATER 
    AGTS_Time_PCT_Output.insert(END, AGTS_Time_PCT)
        
#%% Execute Button 
def Master_Func():
    Build_Crew()
    #TC_Name_Fill()
    #GNR_Name_Fill()
    #LDR_Name_Fill()
    #Crew_Time_Fill()
    #Gun_Date_Fill()
    Gun_Score_Fill()
    Gun_Rate_Fill()
    Gun_PCT_Fill()
    AGTS_Time_Fill()
    AGTS_Time_PCT_Fill()
    
    
Execute_Button = Button(Dash, bg = 'green', text = 'Select Crew', command = Master_Func).place(x = 360, y = 25)
 
#%% Show plots 
def Score_Plot():
    DV.hist_box(M1['GTVI Score'],Crew_Data.loc[MK_Dict[Master_Key],'GTVI Score'],'Gunnery Table 6 Scores',7,1)

def Time_Plot():
    DV.hist_box(M1['Time'],Crew_Data.loc[MK_Dict[Master_Key],'Time'],'Time Spent in AGTS Trainer this year',8,2)   
    
def Eng_Plot():
    DV.hist_box(M1['AVG Time Of Eng'],Crew_Data.loc[MK_Dict[Master_Key],'AVG Time Of Eng'],'Average time of ngagement in AGTS this year',8,2) 
    
Score_Plot_Button = Button(Dash, bg = 'green', text = 'Compare GTVI Scores', command = Score_Plot).place(x = 560, y = 25)
Time_Plot_Button = Button(Dash, bg = 'green', text = 'Compare Time in AGTS', command = Time_Plot).place(x = 720, y = 25)
Eng_Plot_Button = Button(Dash, bg = 'green', text = 'Compare  AVG ENG Time in AGTS', command = Eng_Plot).place(x = 880, y = 25)

#%% Exit Button

# Exit Button
def Quit_Func():
    Dash.destroy()
    TC_Name_Output.delete(1.0, END)
    Crew_Selected.delete(0.0, END)
    
Quit = Button(frame, text = "QUIT", bg = 'red', command = Quit_Func).grid(row =10, column = 10)

#%% Run Program
Dash.mainloop()