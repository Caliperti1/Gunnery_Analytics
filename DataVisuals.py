###################################################
# Coder: 1LT Chris Aliperti and CDT Joshua Wong
# Title: Data Visualization
# Date: 07JUL2022
# Version: 1.1
###################################################

# File containing all data visualization functions. A critical funciton of the gunnery analytics program is
# communicating complex data to a non-technical audience so they can use it to mae infomred decisions.
# many of these functions are specifically designed to insert the figures into the Dashboard GUI.

#%% Import Libraries and Functions 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

#%% hist_box
# Generates a histogram and box and whisker plot on one figure to be input into a TKinter GUI.
 
#   Inputs: 
#       data - dataframe containing data for variable of interest 
#       score - variable value for single crew of interest (will be used to illustrate how a crew compares to peers)       
#       title - string containing title of plot to be printed on figure 
#       row - row that figure will be inserted into on tkinter GUI frame 
#       col - column that figure will be inserted into on tkinter GUI frame       

#    outputs: 
#       figure - histogram and boxplot on one figure with a line illustrating where the selected crew falls 
#           in comparision to the rest of the data. This figure will be inserted into a GUI as part of this function.

def hist_box(data, score, title, row, col):

    # Declare Dash as global so this function can referece GUI created externally named 'Dash'    
    global Dash
        
    # set a grey background (use sns.set_theme() if seabourn version 0.11.0 or above)
    sns.set(style='darkgrid')

    # creating a figure composed of two matplotlib.Axes objects (ax_box and ax_hist)
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': (.15, .85)})

    # assigning a graph to each ax
    sns.boxplot(data, ax=ax_box)

    # assigning width of bins
    sns.histplot(data, bins=15, ax=ax_hist)

    # draw line representing one score
    plt.axvline(score, color='k', linestyle='dashed', linewidth=1)

    # Remove x axis name for the boxplot
    ax_box.set(xlabel='')

    # add title 
    plt.title(title)
    
    # Create TKinter Canvas containing figure
    canvas = FigureCanvasTkAgg(f)
    canvas.draw()
    
    # Place canvas on TKinter window 
    canvas.get_tk_widget().grid(row = 6, column = 0)
    
#%% Heat map
    # Version 1.1 does not currently include a heat map function. The heat map will be a graphic depiction of a coreelation matrix.
def heat_map(data1,data2,data3, score):

    pass



