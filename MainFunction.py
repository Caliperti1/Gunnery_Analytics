###################################################
# Coder: 1LT Chris Aliperti and CDT Joshua Wong
# Title: Main Gunnery Data Cleaning
# Date: 07JUL2022
# Version: 1.1
###################################################

# The central data cleaning file that cleans and combines simulation data with the gunnery data. 
# Widens data from individual Soldiers and creates crews.
# Data cleaning is broken down into functions to allow for easily manipulation of the end datasets
# changing the functions run in main() at the bottom will ultimately affect whihc cleaning functions are performed on the raw data

#%% Import Libraries and Functions
import glob
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

#%% Simcombine 
# Takes in a folder of AGTS sim data and combines them into one large AGTS sim file.
 
#   Inputs: 
#       folder: fodler  titled 'DirtySimData' of .csv or .xlsx files from AGTS (with blank charaters from linux removed beforehand)
    
#    outputs: 
#       combined_csv - 1 dataframe containing data from all files in folder. titled 'combined_csv'

def simcombine(folder):
    # using glob to find csv files and save to a list of file names
    extension = 'csv'
    all_filenames = [i for i in glob.glob(f'{folder}/*.{extension}')]

    # combine all files in the list
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])

    # return concatenated dataframe
    return combined_csv

#%% SimCleaning 
#Takes in an AGTS file and outputs a cleaned version with each crew having only one row with 7 variables:
#   Master Key, BN, Crew, Time, Tot Hits, AVG Time To ID, AVG Time Of Eng

# Similar function will need to be made when COFT data is available for Bradleys 

#   Inputs:
#       AGTS_3135 - concatenated dataframe of AGTS Data (combined_csv from simcombine func)
    
#    outputs: 
#       AGTS_Cons - dataframe with each row consisting of all simulaiton data for 1 Tank crew, with 7 columns.

# Manipulate Clean_Columns (line 73 ) to change which variables is retained in final dataframe

def simcleaning(AGTS_3135):

    # Drop all rows that are missing Commanders name, Gunners name and Vehcile ID.
    # This removes all runs that are not actually tables shot by crew (ie systems diagnostic runs have 
    # their own rows in the dirty data)
    AGTS_3135.dropna(subset = ['Commander Name'], inplace = True)
    AGTS_3135.dropna(subset = ['Gunner Name'], inplace = True)
    AGTS_3135.dropna(subset = ['Vehicle ID'], inplace = True)
    # If run alone this will print the first 5 row of dataframe to check work
    AGTS_3135.head()

    # Remove Runs with 0 for time to kill 
    AGTS_3135['Average Time To Kill'].astype('float')
    for i in range(len(AGTS_3135['Average Time To Kill'])):
            if AGTS_3135['Average Time To Kill'].iloc[i] == 0:
                    AGTS_3135['Average Time To Kill'].iloc[i] = np.nan
    AGTS_3135['Average Time To Kill'].dropna(how = 'any', axis = 0,inplace = True)

    # Consolidate all Crew data into single row for that crew  
    
    # Build empty data set for pulling in new data 
    Clean_Columns = ['VIC ID','BN','Crew','Tot Raw Time','hrs','mins','secs','Time','Tot Hits','AVG time to Kill']
    AGTS_Clean = pd.DataFrame(columns = Clean_Columns)

    # Pull in raw data 
    AGTS_3135.astype({'Vehicle ID': str})
    AGTS_3135.astype({'Total Exercise Training Time': str})
    AGTS_3135.astype({'Total Hits': float})
    AGTS_Clean['Tot Hits'] = AGTS_3135['Total Hits'].astype(float)
    AGTS_Clean['VIC ID'] = AGTS_3135['Vehicle ID']
    AGTS_Clean['AVG Time To Kill'] = AGTS_3135['Average Time To Kill'].astype(float)
    AGTS_Clean['AVG Time To ID'] = AGTS_3135['Average Time To ID'].astype(float)
    # AVG Time of engagement is not recorded by AGTS but was identified by SMEs as a potentially important variable to create
    AGTS_Clean['AVG Time Of Eng'] = AGTS_Clean['AVG Time To Kill'] - AGTS_Clean['AVG Time To ID']

    # Seperate Vehcile ID in to format of other data frame (BN then Crew)
    AGTS_Clean['BN'] = AGTS_3135['Vehicle ID'].str[:4]
    AGTS_Clean['Crew'] = AGTS_3135['Vehicle ID'].str[5:8]

    # Seperate Time to hours minutes and seconds then convert to float and add
    AGTS_Clean['Tot Raw Time'] = AGTS_3135['Total Exercise Training Time']
    AGTS_Clean['hrs'] = AGTS_3135['Total Exercise Training Time'].str[:2].astype(float)
    AGTS_Clean['mins'] = AGTS_3135['Total Exercise Training Time'].str[4:6].astype(float)
    AGTS_Clean['secs'] = AGTS_3135['Total Exercise Training Time'].str[8:10].astype(float)

    # Combine times 
    AGTS_Clean['Time'] = (AGTS_Clean['hrs'] * 60) + AGTS_Clean['mins'] + (AGTS_Clean['secs']/60)

    # Set key to allign with main dataset 
    AGTS_Clean['BN'] = AGTS_Clean['BN'].replace(['1/27','2/69','3/67','6/08','2/07','3/15','1/64','3/69','2/72','1/68'],
                                            ['1-27 IN','2-69 AR','3-67 AR','6-8 CAV','2-7 IN','3-15 IN', '1-64 AR','3-69 AR','3-67 AR','1-68 CAV'])
   
    # Master Key variable is the key used to identify crews and join data from different lcoations 
    AGTS_Clean['Master Key'] = AGTS_Clean['BN'] + ' ' + AGTS_Clean['Crew']

    # Combine Crews 
    AGTS_Cons = AGTS_Clean.groupby(['Master Key']).agg({'BN':'last','Crew':'last','Time':'sum','Tot Hits':'sum','AVG Time To ID':np.nanmean,'AVG Time Of Eng':np.nanmean})

    return AGTS_Cons

#%% GunneryCleaning 
# Primary function for cleaning raw data from Vantage (300+ variables on each individual Soldier), combining Soldiers into crews,
    
#   Inputs:
#       df - dataframe with each row repersenting 1 Soldier and containing all Vantage data, crew identifier and previous Gunnery performance (see Gunnery Data V1)

#    outputs: 
#       M1_df - dataframe with cleaned Tank crew data 
#       M2_df - dataframe with cleaned Bradley crew data
#       crews_df - M1_df and M2_df combined in 1 dataframe ( useful for some modeling that required alot of observations)

# manipuate Cols (Line 165) to change which variables are retained in final data set.

def gunnerycleaning(df):
    
    # Drop Crews that do not have Table VI Score 
    df['GTVI Score'].dropna(axis=0,inplace = True)

    # Rename some poorly named columns from original dataset
    df = df.rename(columns = {'Crew ' : 'Crew','qual_M4M16556' : 'M4','qual_M99mmBerettaPistol' : 'M99',
                            'qual_XM17' : 'M17'  })

    # Make Master Gunner Variable from string of ASIs 
    df['master_gunner'] = 0
    for i in range(len(df['all_duty_additional_skills'])):
        if 'K8' in str(df['all_duty_additional_skills'][i]) or 'J3' in str(df['all_duty_additional_skills'][i]) or 'R8' in str(df['all_duty_additional_skills'][i]):
            df['master_gunner'][i] = 1

    # Make Variables for other ASIs 
    df['maintanance_operations'] = 0
    for i in range(len(df['all_duty_additional_skills'])):
        if 'K4' in str(df['all_duty_additional_skills'][i]) or 'B9' in str(df['all_duty_additional_skills'][i]):
            df['maintanance_operations'][i] = 1

    # Make age variable out of age brackets
    df['age1'] = df['age_bracket'].str[0:2]
    df['age2'] = df['age_bracket'].str[3:5]
    df['age1'] = pd.to_numeric(df['age1'])
    df['age2'] = pd.to_numeric(df['age2'])
    df['avg_age'] = (df['age1'] + df['age2']) / 2
    df.drop('age1', axis = 1, inplace = True)
    df.drop('age2', axis = 1, inplace = True)

    # Make a skill variable (1-3 for E and 4 for O)
    df['skill_level'] = df['individual_skill_level'].str[12]
    df['O_E'] = df['grade_rank_code'].str[0]
    df['O_E'] = df['O_E'].fillna('M')
    for i in range(len(df['O_E'])):
        if df['O_E'] [i] == 'O':
            df['skill_level'][i] = '4'
    df['skill_level'] = pd.to_numeric(df['skill_level'])

    # Choose variables that are not nonsensical
    cols = ['Crew', 'Battalion', 'Vehicle', 'GTVI Score', 'GT VI Eng', 'Q1',
        'Position', 'VACANT', 'Sim Total Hrs','GTLF Score', 'GTLF Attmpt',
        'master_gunner', 'maintanance_operations', 'avg_age', 'apft_pass', 
        'apft_score_total','apft_score_su','apft_score_pu','apft_score_run','Sim Total Hrs','acft_score_total',
        'body_comp_pass', 'ethnic_group', 'height', 'skill_level',
        'O_E', 'M4', 'M99', 'M17', 'time_in_service_months',
        'weight']

    df = df[cols]

    # Drop empty columns 
    df = df.dropna(how = 'all', axis = 1)

    # Data Widening (Make Crews)

    # Make a crew df with crew wide variables 
    crew_cols = ['Crew','Battalion','Vehicle',
            'GTVI Score','GT VI Eng','Q1']
    crew_df = pd.DataFrame({'members' : df.groupby(crew_cols).size()}).reset_index()

    # Pull data into each member type 
    member_df = df.copy()
    driver_df = member_df.loc[member_df['Position'] == 'Driver']
    loader_df = member_df.loc[member_df['Position'] == 'Loader']
    gunner_df = member_df.loc[member_df['Position'] == 'Gunner']
    commander_df = member_df.loc[member_df['Position'] == 'Commander']

    # Combine crew and member data into one dataframe 
    dict = {'dr' : driver_df, 'ld' : loader_df, 'gn' : gunner_df, 'tc' : commander_df}
    for key in dict.keys():
        df = dict[key]
        cols_new = []
        for col in df.columns:
            if col not in crew_cols:
                cols_new.append(key + '_' + col)
            else:
                cols_new.append(col)
        df.columns = cols_new
        crew_df = pd.merge(crew_df, df, on = crew_cols, how = 'left')

    crew_df.columns

    # Drop Crews that have a vacant Gunner or Commander 
    crew_df['gn_VACANT'].dropna(axis=0,inplace = True)
    crew_df['tc_VACANT'].dropna(axis=0,inplace = True)

    # Split dataframe in Bradley and Tank and drop variables that do not apply to new dataframes then reset indicies 
    M1_df = crew_df.loc[crew_df['Vehicle'] =='Tank']
    M2_df = crew_df.loc[crew_df['Vehicle'] =='Bradley']

    M1_df = M1_df.dropna(how = 'all', axis = 1)
    M2_df = M2_df.dropna(how = 'all', axis = 1)

    crew_df = crew_df.reset_index()
    M1_df = M1_df.reset_index()
    M2_df = M2_df.reset_index()

    return M1_df, M2_df, crew_df


#%% BradleyFill
# Determine data that is still missing from Bradley dataframe and eithe drop variables or impute missing data if 
# missing data below tolerance. Tolerance set high to retain  as many variables and observations as possible.

#   Inputs: 
#        M2_df - dataframe with cleaned Bradley crew data

#   Outputs:
#       M2_df - dataframe with missing Bradley data filled

def bradleyfill(M2_df):

    # Determine missing data 
    # Set missing variable percentage tolerance (0-1)
    tol = .5 

    b_tol_missing = []
    b_missing_any = []
    tol_M2 = tol * len(M2_df['GTVI Score'])
    missing_data_M2 = M2_df.isnull()
    col = M2_df.columns
    count_M2 = 0

    for column in col:
        miss = missing_data_M2[column].tolist()
        trues = miss.count(True)
        if trues > tol_M2:
            count_M2 = count_M2 +1
            b_tol_missing.append(column)
            M2_df.drop(column, axis = 1, inplace = True)
        if ((trues > 0) & (trues <= tol_M2)):
            b_missing_any.append(column)
              
    # Remove Variables IDed above as having insufficient data 
    
    
    # Create function to impute missing quant variables with median and cat varibles with "Missing"
    mymedian_imputer = SimpleImputer(strategy='median')
    cat_b = []
    quant_b = []
    for col in b_missing_any:
        if M2_df[col].dtypes == 'object':
            M2_df[col] =M2_df[[col]].fillna(value=-1)
            M2_df = M2_df.astype({col : "str"})
            M2_df[col == '-1', col] = 'Missing'
            cat_b.append(col)
        else:
            mymedian_imputer.fit_transform(M2_df[[col]])
            M2_df[col] = mymedian_imputer.fit_transform(M2_df[[col]])
            quant_b.append(col)

    return M2_df


#%% Tank Fill 
# Determine data that is still missing from Tank dataframe and either drop variables or impute missing data if 
# missing data below tolerance. Tolerance set high to retain  as many variables and observations as possible.

#   Inputs: 
#        M1_df - dataframe with cleaned Tank crew data

#   Outputs:
#       M1_df - dataframe with missing Tank data filled


def tankfill(M1_df):

    # Determine missing data 
    # Set missing variable percentage tolerance (0-1)
    tol = .5 

    t_tol_missing = []
    t_missing_any = []
    tol_M1 = tol * len(M1_df['GTVI Score'])
    missing_data_M1 = M1_df.isnull()
    col = M1_df.columns
    count_M1 = 0

    for column in col:
        miss = missing_data_M1[column].tolist()
        trues = miss.count(True)
        if trues > tol_M1:
            count_M1 = count_M1 +1
            t_tol_missing.append(column)
            M1_df.drop(column, axis = 1, inplace = True)
            if ((trues > 0) & (trues <= tol_M1)):
                t_missing_any.append(column)
    # Remove Variables IDed above as having insufficient data 
    
    
    # Create function to impute missing quant variables with median and cat varibles with "Missing"
    mymedian_imputer = SimpleImputer(strategy='median')
    cat_t = []
    quant_t = []
    for col in t_missing_any:
        if M1_df[col].dtypes == 'object':
            M1_df[col] =M1_df[[col]].fillna(value=-1)
            M1_df = M1_df.astype({col : "str"})
            M1_df[col == '-1', col] = 'Missing'
            cat_t.append(col)
        else:
            mymedian_imputer.fit_transform(M1_df[[col]])
            M1_df[col] = mymedian_imputer.fit_transform(M1_df[[col]])
            quant_t.append(col)

    return M1_df


#%% agts_to_tank
# Combine cleaned AGTS sim data with cleaned and filled Tank datafrme
    # This same function will need to be made for Bradleys when COFT data becomes available 

#   Inputs:
#       M1_df - dataframe with missing Tank data filled
#       AGTS -  dataframe with each row consisting of all simulaiton data for 1 Tank crew, with 7 columns.

#   Outpus 
#       M1_df - dataframe with all Tank crew data and AGTS sim data joined into 1 

def agts_to_tank(M1_df, AGTS):
            
    # Create master keys

    M1_df['Master Key'] = M1_df['Battalion'] + ' ' + M1_df['Crew']

    # Left Join AGTS Data to Tank Dataset 

    M1_df = M1_df.merge(AGTS, on='Master Key', how = 'left')

    return M1_df

#%% calc_percentile 
# calculates the percetile that a given scores fall into. This is used to build a couple of new variables 
# that are useful for analysis.

#   Input:
#       data - set of values for any 1 variable
#       score - One observation's score for that variable

#   Output:
#       percentile score - between 0 - 100 repersenting what percent of the data the given score is greater than.

def calc_percentile(data, score):

    data = list(data.values)
    data.append(score)
    data = sorted(data)

    count = 0
    for i in range(len(data)):
        if data[i] == score:
            break
        else:
            count += 1

    return (count/len(data)) * 100


#%% Tank data filter 
# Additional cleaning and agregation of data for M1 dataframe.
# Manipulate cols (line 415) to change varaibles carried forward for analysis 

#   Input:
#       df - Tank dataframe from previous cleaning steps.

#   Output:
#       df - Tank dataframe with only variables used for analysis 

def tank_data_filter(df):

    # Turn Time from min to hours
    df['Time'] = df['Time']/60

    # Make new column GTVI rating which is a categorical variable
    new_scores = []
    for score in df['GTVI Score']:
        if score >= 900:
            new_scores.append('Distinguished')
        elif score >= 800:
            new_scores.append('Superior')
        elif score >= 700:
            new_scores.append('Qualified')
        else:
            new_scores.append('Fail')
    df['GTVI Rating'] = new_scores

    # Calculate and add GTVI Score Percentile (make new column)
    score_percentiles = []
    for score in df['GTVI Score']:
        score_percentiles.append(calc_percentile(df['GTVI Score'], score))
    df['GTVI Percentile'] = score_percentiles

    # Calculate and add Time in sim Percentile
    time_percentiles = []
    for time in df['Time']:
        time_percentiles.append(calc_percentile(df['Time'], time))
    df['Time Percentile'] = time_percentiles

    cols = ['Master Key','GTVI Score','GTVI Rating','GTVI Percentile','Time','Time Percentile','Tot Hits','AVG Time To ID','AVG Time Of Eng']
    df = df[cols]
    df = df.dropna()    
    df = df.reset_index()
    df.drop('index', inplace=True, axis=1)

    return df

#%% main 
# Main runs selected data filtering funcitons to produce final cleaned dataframes for Tank and Bradley.
# Main can be manipulated to include any combination of the baove funcitons and allows for the data cleaning to be customized based on 
# the desired analysis. 

#   Input:
#       NONE

#   Output:
#       M1_clean.csv - csv file containing cleaned dataframe for M1. csv will be saved in the working directory 
#       M2_clean.csv - csv file containing cleaned dataframe for M2. csv will be saved in the working directory 
#       data_clean.csv - csv file containing combinedc leaned dataframe for M1 and M2, this is useful for analysis while there are limited observations. csv will be saved in the working directory  

def main():

    # Combine Dirty Simulator data
    combined_csv = simcombine('DirtySimData')

    # clean sim data 
    clean_csv = simcleaning(combined_csv)

    # read in raw gunnnery data 
############THIS STEP IS BEING UPDATED IN V1.2 TO PULL FROM DTMS AND BUILD THIS INTERNALLY##########
    df = pd.read_excel('Gunnery Data V1.xlsx')

    # Run gunnery cleaning to split dataframes 
    M1_df, M2_df, crew_df = gunnerycleaning(df)

    # Add AGTS data to tank dataframe 
#########THIS STEP WILL BE REPLICATED IN FUTURE VERSIONS FOR M2 COFT DATA #####################
    M1_df = agts_to_tank(M1_df, clean_csv)

    # final filter on Tank data to choose variables for analysis 
#########THIS STEP WILL BE REPLICATED IN FUTURE VERSIONS FOR M2 DATA #####################  
    M1_df = tank_data_filter(M1_df)

    M1_df.to_csv('M1_clean.csv')
    M2_df.to_csv('M2_clean.csv')
    crew_df.to_csv('data_clean.csv')

