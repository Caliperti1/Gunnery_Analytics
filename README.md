Gunnery Data Analytics 
Product of Marne Innovations 
By 1LT Chris Aliperti and CDT Joshua Wong
Current as of: 18 JUL 22

Version 1.1 

Project Overview: 
    Commanders in the 3rd Infantry Division (US Army) currently lack the access and capability to
    make data driven decisions about their crews' performance or training. This project beigns to create a prototype 
    for a Gunnery analytics program that organizes, visualizes and analyzes all data associated with Gunnery to provide
    Commanders the appropriate tools to make data driven decisions and insights about their formations; increasing 
    the lethality of the 3rd Infantry Division. 
        
    This project in its current state takes data from local records, Vantage and simualtion trainers and creates a single
    observation for each crew, consisting of all related data. These observations are then used to create visual products 
    and a few preliminary models. To demonstrate the potential of this program some basic GUIs are created, allowing a 
    commadner to interface with the database, select their specific crews, and conduct some basic analysis on their 
    performance. Work on this prototype program is being completed by the Marne Innovation team and the United States
    Military Academy Math Department.        
    
Version Notes:
    Version 1.1 (18 JUL 22) - The first attempt at creating a single program to conduct gunnery analytics. Notes are made throughtou
        for areas of improvement in future version updates. Due to limitations in available data most analysis is only run on M1
        however the framework is laid to easily replicate functions for M2 in near future.
        
    Version 1.2 (IN PROGRESS) - Update to MainFunction.py that will no longer require 'Raw Gunnery Data' excel sheet that is manually popualted.
        Will include functions to build this from DTMS data. Limiting the amount of manual work by units up front.If M2 data becomes available
        functions will be created to extract M2 COFT simulator data from a pdf, filter M2 dataframe and run analysis.
        
Python Files in Project:
    MainFunction.py 1.1- The central data cleaning file that cleans and combines simulation data with the gunnery data. 
        Widens data from individual Soldiers and creates crews. Data cleaning is broken down into functions to allow for
        easily manipulation of the end datasetschanging the functions run in main at the bottom will ultimately affect
        which cleaning functions are performed on the raw data.
                
        Functions:
            simcombine - Takes in a folder of AGTS sim data and combines them into one large AGTS sim file.
               
                Inputs: 
                   folder: fodler  titled 'DirtySimData' of .csv or .xlsx files from AGTS (with blank charaters from linux removed beforehand)
                Outputs: 
                   combined_csv - 1 dataframe containing data from all files in folder. titled 'combined_csv'
               
            simcleaning - Takes in an AGTS file and outputs a cleaned version with each crew having only one row with 7 variables:
                 Master Key, BN, Crew, Time, Tot Hits, AVG Time To ID, AVG Time Of Eng
                 Similar function will need to be made when COFT data is available for Bradleys 
               
                 Inputs:
                    AGTS_3135 - concatenated dataframe of AGTS Data (combined_csv from simcombine func)            
                 Outputs: 
                    AGTS_Cons - dataframe with each row consisting of all simulaiton data for 1 Tank crew, with 7 columns.
           
            gunnerycleaning - Primary function for cleaning raw data from Vantage (300+ variables on each individual Soldier), combining Soldiers into crews,
                
                Inputs:
                    df - dataframe with each row repersenting 1 Soldier and containing all Vantage data, crew identifier and previous Gunnery performance (see Gunnery Data V1)
                Outputs: 
                   M1_df - dataframe with cleaned Tank crew data 
                   M2_df - dataframe with cleaned Bradley crew data
                   crews_df - M1_df and M2_df combined in 1 dataframe ( useful for some modeling that required alot of observations)
                   
            bradleyfill - Determine data that is still missing from Bradley dataframe and eithe drop variables or impute missing data if 
                missing data below tolerance. Tolerance set high to retain  as many variables and observations as possible.
                
                Inputs: 
                     M2_df - dataframe with cleaned Bradley crew data
                Outputs:
                    M2_df - dataframe with missing Bradley data fille
                    
            tankfill - Determine data that is still missing from Tank dataframe and either drop variables or impute missing data if 
                missing data below tolerance. Tolerance set high to retain  as many variables and observations as possible.
                
                Inputs: 
                     M1_df - dataframe with cleaned Tank crew data
                Outputs:
                    M1_df - dataframe with missing Tank data filled
                    
            agts_to_tank - Combine cleaned AGTS sim data with cleaned and filled Tank datafrme
                This same function will need to be made for Bradleys when COFT data becomes available 
                
                Inputs:
                    M1_df - dataframe with missing Tank data filled
                    AGTS -  dataframe with each row consisting of all simulaiton data for 1 Tank crew, with 7 columns.
                Outpus 
                    M1_df - dataframe with all Tank crew data and AGTS sim data joined into 1 
                    
            calc_percentile - calculates the percetile that a given scores fall into. This is used to build a couple of new variables 
                that are useful for analysis.
                
                Inputs:
                    data - set of values for any 1 variable
                    score - One observation's score for that variable
                Outputs:
                    percentile score - between 0 - 100 repersenting what percent of the data the given score is greater than.
                    
            tank_data_filter - Additional cleaning and agregation of data for M1 dataframe.
                
                Inputs:
                    df - Tank dataframe from previous cleaning steps.
                Outputs:
                    df - Tank dataframe with only variables used for analysis 
            
            main - Main runs selected data filtering funcitons to produce final cleaned dataframes for Tank and Bradley.
                Main can be manipulated to include any combination of the baove funcitons and allows for the data cleaning to be customized based on 
                the desired analysis. 

                Inputs:
                    NONE (functions inside of main are called with all required inputs)

                Outputs:
                    M1_clean.csv - csv file containing cleaned dataframe for M1. csv will be saved in the working directory 
                    M2_clean.csv - csv file containing cleaned dataframe for M2. csv will be saved in the working directory 
                    data_clean.csv - csv file containing combinedc leaned dataframe for M1 and M2, this is useful for analysis while there are limited observations. csv will be saved in the working directory  

    DataVisuals.py 1.1 - File containing all data visualization functions. A critical funciton of the gunnery analytics program is
        communicating complex data to a non-technical audience so they can use it to mae infomred decisions.
        many of these functions are specifically designed to insert the figures into the Dashboard GUI.
        
        Functions:
            hist_box - Generates a histogram and box and whisker plot on one figure to be input into a TKinter GUI.
             
                Inputs: 
                    data - dataframe containing data for variable of interest 
                    score - variable value for single crew of interest (will be used to illustrate how a crew compares to peers)       
                    title - string containing title of plot to be printed on figure 
                    row - row that figure will be inserted into on tkinter GUI frame 
                    col - column that figure will be inserted into on tkinter GUI frame       
                Outputs: 
                    figure - histogram and boxplot on one figure with a line illustrating where the selected crew falls 
                        in comparision to the rest of the data. This figure will be inserted into a GUI as part of this function.

            heat_map -  Version 1.1 does not currently include a heat map function. The heat map will be a graphic depiction of a coreelation matrix.
    
    M1Models.py 1.1 - M1Models.py is a collection of models that attempt to determine relationships between
        the variables and the crew's gunnery performance and eventually develop a predictive model
        to give Commanders insight into how a new crew is predicted to perform based on previous data.

        In Version 1.1 the models are only developed using M1 data due to the lack of sim data for the Bradley.
        These functions provide a framework for models but more data is needed to properly train them and 
        determine which model is the most appropriate for this problem.

        This file is more or less a scratch book for the team to experiemnt with various models without having 
        to rewrite code to make minor adjustments or wory about changing the model code when the data is updated.
        
        Functions:
            m1_pca - principle component analysis using the sim data for tanks. Principle component analysis 
                reduces the explantaroy varaibles into 2 or 3 'priniple components.' This allows the higher dimension data
                to be visualized in 2 or 3 dimensions. If the data is seperable this will reveal some clear delineation between
                groups of data which can then be used to create a model.
    
                Inputs:
                    df - cleaned M1 dataframe

                Outputs:
                    finalDf - dataframe containing the 3 principle components and the dependent variable  
                    explained_variance_ratio_ - EVR is a meaure of how much variance in the dependent variable is explained
                        by each of the principle components. 
                    components_ - shows how much of an impact each of the orignal variables has on each component. 
                        (This is essentially the eigen vectors for each principle component)
            
            m1_pc_selection - Funcluded in a model. Creates an elbow plot that will allow the modeler to make 
                a decision about how many principle components to retain. This funciton should be used before the 
                previous function in order to determine how many PCs (line 66)

                Inputs:
                    df - clean M1 dataframe
                Outputs:
                    perc_var - percent variation explained by each additional principle component added.
                    figure - elbow plot that shows 'elbow' at the optimal primciple component number 
                    
            m1_pcr_model - Creates a principle component regression model. A PCR is similar to a linear regression
            where x_i is a principle component.

                Inputs: 
                    df - clean M1 dataframe 
                Outputs:
                    regr - principle component regression oject 
                    rmse - root mean square error of model 
                    
            m1_reg_model - Simple linear regression model (with current features using only sim data).

                Inputs:
                    df - clean M1 dataframe 
                Outputs:
                    regr - linear regression object 
                    rmse - root mean squre error of model
            multi_reg - Mutliple regression that allows features and targets to be maniplualted in finction input 

                Inputs:
                    df - Clean M1 dataframe 
                    target - string containing target variable name 
                    features - list of strings containing feature names 

                Outpus:
                    regr - multiple regression object 
                    regr.coef_  - list of coefficeints for all features    
                    
            m1_reg_predict - predicts scores using a regression model object (generated in previous functions)

                Inputs:
                    regr - regression object 
                    predVals - default predicted values (inital guess)
                Outputs:
                    predictedScore - predicted score 
                    
            m1_pcr_predict - - predicts scores using a principle component regression model object (generated in previous functions)

                Inputs:
                    regr - regression object 
                    predVals - default predicted values (inital guess)
                Outputs:
                    pred - predicted score 
                    
            main -  runs models currently generated in M1 models and provides scores to allow analyst to choose best model to move forward with.

                Inputs:
                    NONE

                Outputs:
                    elbow plot to choose number of principle components 
                    regression RMSE
                    PCR regression RMSE
                    predicted score by simple regression
                    predicted score by PC regression
                    princple components 

    GUI_V1.py 1.1 - # GUI_V1 generates a prototype Commander's dashboard that allows COmmanders to interface
        with the data by selecting a crew of their choice and viewing all associated data. GUI also
        contains various plots to visualize the crew's performance compared to peers.
        GUI_V1 is also valuable as it incorporates and runs the other files and funcitons in this project
        but does not require any interaction with raw code if not desired. This is very useful for 
        demonstrating the project's capabilites to non-technical audiences.Functions for executing GUI 
        specific actions (ie filling text boxes with buttons) are not explicitly explained for neatness sake.

    Model_GUI.py 1.1 - This GUI is used to demonstrate a capability of a predictive model. It allows Commanders to input certain data about a 'new'
        crew and get a predicted score based on one of the models created in M1Models.py. Given the scarcity of data these models are not very accurate yet
        (The model building and performance is a discussion for another document).

Formatted Input Needed: 
    'Gunnery Data V1.csv' - For version 1.1 this is a csv file with each rowc containing the DOD ID, Crew ID, Gunnery scores and Vantage data for a single Soldier.
         An example of this file is in the working directory folder. 
    
    AGTS Sim data - individual csv files pulled from the AGTS trainer. We recommend saving each file as 'AGTS_"trainer number"_"Date"'. Store all files in the dirty sim folder.
        For version 1.1 this file still requires some external preprocessing. Becuase the AGTS is a linux based system, the fields that ars supposed to be NaN are filled with 
        characters that python does not recognize. To correct this open the csv file with excel, copy the contents of one of the seemingly empty cells, and use the replace all funcion 
        to replace all of these with actual empty cells. Future versions will explore ways of automating this.
        
    COFT Sim Data - Version 1.1 does not support COFT data yet.
    
How to Run the Program:
    (1) Save 'Gunnery Data V1.csv' to working directory 
    (2) Preprocess and properly name AGTS usage logs and save them to 'DirtySimData' folderin working directory 
    (3) Run GUI_V1.py 
    (4) Select Crew and view associated data. 
        (4a) Please send feedback on what data you would like to see displayed for future versions

Python Code Packages:
    Python Version: 3.9
    1) numpy (1.17.4)
    2) matplotlib (3.1.3)
    3) pandas (0.25.3)
    4) sklearn (0.0)
    5) seaborn (0.11.2)
    6) tk (0.1.0)

How to Contribute:
    please send any questions or recomendations to the Marne Innovations Deputy Chief Chris Aliperti at christopher.c.aliperti.mil@army.mil 

Credits:
    LTC James Starling (USMA)
    SFC Bill Wilder (3ID)
    CDT Seth Benson (USMA)
    CDT Mary Bell (USMA)
    CDT Ben Wetstein (USMA)
