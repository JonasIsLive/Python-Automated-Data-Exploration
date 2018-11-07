#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 16:55:17 2018

@author: jonasbjerg
"""

import pandas as pd # data science essentials (read_excel, DataFrame)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns #alternative data visualization
import statistics #used to flag outliers based on z values
import statsmodels.api as sm
import statsmodels.formula.api as smf # regression modeling

#the below imports are currently unused, but here in just case the team 
#alters course and want to do something with them 
#import statsmodels.graphics as smg
#import numpy as np #trendline
#import scipy.stats as st #skew and CI
#from scipy.stats import kurtosis, skew #kurtosis skewness
#from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder

#set max values in pandas
pd.set_option('max_rows', 500) #set maximum rows to show at 500
pd.set_option('max_columns', 500) #set maximum columns to show at 500


#The WORLDTRENDLINE dataframe is from wbdata 
#It contains all data for all countries and specified columns from 2000 to 2017

#The out commented code below will only work if pip install of the WBdata API
"""
import wbdata #world data    #doesnt work because it needs pip install
import datetime
data_dates = (datetime.datetime(2001,1,1), datetime.datetime(2018,1,1))

wbdata.get_source()
data = wbdata.get_dataframe({'EN.ATM.CO2E.PP.GD.KD':'values'}, 
                            country=('BGR', 'BFA', 'XKX', 'LIE', 'MCO', 'SVK', 'LVA', 'AZE', 'CZE', 'POL', 'UKR', 'ITA', 'BIH', 'STP', 'LUX'), 
                            data_date=data_dates, 
                            convert_date=False, keep_levels=True)
"""

#Importing the two DataFrames (one is provided by Chase)
file ='WorldFull.xlsx'
original_df = pd.read_excel(file)


#check % missing data in original dataset and in europe
ori_ = original_df.copy()

ori_world = ori_.iloc[:44,:]
ori_worldFull = ori_world.append(ori_.iloc[59:-1,:])
ori_eu = ori_.iloc[44:59,:]


(ori_worldFull.isnull().sum()/len(ori_worldFull)).round(2)

(ori_worldFull.isnull().sum().sum())

(ori_eu.isnull().sum()/len(ori_eu)).round(2)

(ori_eu.isnull().sum().sum())


#The WORLDTRENDLINE contains the api downloaded DataFrame
#In it are trendlines for each row of missing data 
#where other years had additional data. 

#If a country only suplied a value one year, then that year was inserted that
#As it is assumed that a picture of a countries situation a couple year prior 
#Or more still is a better representation of the country, 
#than a simple mean or median based on other countries values.
#If more than one value had been supplied over multiple years, 
#then a trendline analysis was applied to provide a value for the desired year 
#Trendlines are attempting to indicate where the country would be situated today.
#This is done to minimize the number of countries where imputing can only be done
#with the mean or median.
        
#everywhere where a trendline was inserted a flag is made in the last column 
#of WORLDTRENDLINE - indicated by a 1 instead of a 0.

file2 ='WORLDTRENDLINE.xlsx'
worldDF = pd.read_excel(file2)


#Dublicate original dataset
dfMean  = pd.DataFrame.copy(original_df)
df_mean = pd.DataFrame.copy(original_df)

dfMedian = pd.DataFrame.copy(dfMean)
df_median = pd.DataFrame.copy(dfMean)

dfDropped = pd.DataFrame.copy(dfMedian)
df_dropped = pd.DataFrame.copy(dfMedian)

dfMeanMedian  = pd.DataFrame.copy(dfMedian)

#Then sort the original data by country_code
#original_df = original_df.sort_values(['country_code'], ascending = True)

#Selects the needed rows and columns to include (Tline is the aformentioned trendline flags)
WorldIn = worldDF.loc[:, ['Country Code','Series Name','YR2014','Tline']]

#Drop any missing values (since those are where no trendline could be made)
WorldDrop = WorldIn.dropna().sort_values(['Country Code'],ascending = True)

#Remap the data to have the column names from the originally provided dataframe
#be aware that CO2_emissions_per_capita had a missplaced ")" at the end 
#in the provided dataset. if that ")" is not removed, this will not run properly
WorldDrop['Series Name'] = WorldDrop['Series Name'].map(
        {'Row number for a country': 'country_index',
         'Access to electricity (% of population)':'access_to_electricity_pop',
         'Access to electricity, rural (% of rural population)':'access_to_electricity_rural',
         'Access to electricity, urban (% of urban population)':'access_to_electricity_urban',
         'CO2 emissions (metric tons per capita)':'CO2_emissions_per_capita',
         'Compulsory education, duration (years)':'compulsory_edu_yrs',
         'Contributing family workers, female (% of female employment) (modeled ILO estimate)':'pct_female_employment',
         'Contributing family workers, male (% of male employment) (modeled ILO estimate)':'pct_male_employment',
         'Employment in agriculture (% of total employment) (modeled ILO estimate)':'pct_agriculture_employment',
         'Employment in industry (% of total employment) (modeled ILO estimate)':'pct_industry_employment',
         'Employment in services (% of total employment) (modeled ILO estimate)':'pct_services_employment',
         'Exports of goods and services (% of GDP)':'exports_pct_gdp',
         'Foreign direct investment, net inflows (% of GDP)':'fdi_pct_gdp',
         'GDP (current US$)':'gdp_usd',
         'GDP growth (annual %)':'gdp_growth_pct',
         'Incidence of HIV (% of uninfected population ages 15-49)':'incidence_hiv',
         'Individuals using the Internet (% of population)':'internet_usage_pct',
         'Intentional homicides (per 100,000 people)':'homicides_per_100k',
         'Literacy rate, adult total (% of people ages 15 and above)':'adult_literacy_pct',
         'Mortality rate, under-5 (per 1,000 live births)':'child_mortality_per_1k',
         'PM2.5 air pollution, mean annual exposure (micrograms per cubic meter)':'avg_air_pollution',
         'Proportion of seats held by women in national parliaments (%)':'women_in_parliament',
         'Tax revenue (% of GDP)':'tax_revenue_pct_gdp',
         'Unemployment, total (% of total labor force) (modeled ILO estimate)':'unemployment_pct',
         'Urban population (% of total)':'urban_population_pct',
         'Urban population growth (annual %)':'urban_population_growth_pct'
         })

#Set the index to one each DataFrame can understand (country_code)
original_df = original_df.set_index('country_code')

"""
#List create to hold the NaN values in original_df(DataFrame) to check if any are
#would be overriten

comp = []

for n in enumerate(WorldDrop['Country Code']):
    
#Two values were found (which the team decides to overide, 
#because the both apear as the new values in WB data)
    comp.append(original_df.loc[WorldDrop.iloc[n[0],0]].at[WorldDrop.iloc[n[0],1]])

"""
#I insert the trendline values for 2014 into the provided DataFrame original_df
for n in enumerate(WorldDrop['Country Code']):
    original_df[WorldDrop.iloc[n[0],1]][WorldDrop.iloc[n[0],0]] = WorldDrop.iloc[n[0],2]
    
# Creating a loop to flag missing values
for col in original_df:

    """ Create columns that are 0s if a value was not missing and 1 if
    a value is missing. """
    
    if original_df[col].isnull().any():
        original_df['m_'+col] = original_df[col].isnull().astype(int)
#SettingWithCopyWarning here is because trying to set a value on a copy of a df
#it is only two values we are trying to overide, 
#that are different from the provided dataset, 
#but from a more recent download from the world bank

# checking to see if the for loop worked
#print(original_df.head())

#check skewness

"""
Acceptable skewness ±2
References:
Gravetter, F., & Wallnau, L. (2014). Essentials of statistics for the behavioral sciences (8th ed.). Belmont, CA: Wadsworth.
George, D., & Mallery, M. (2010). SPSS for Windows Step by Step: A Simple Guide and Reference, 17.0 update (10a ed.) Boston: Pearson.
Field, A. (2009). Discovering statistics using SPSS. London: SAGE
Trochim, W. M., & Donnelly, J. P. (2006). The research methods knowledge base (3rd ed.). Cincinnati, OH:Atomic Dog.
Field, A. (2000). Discovering statistics using spss for windows. London-Thousand Oaks- New Delhi: Sage publications.

"""

###### Missing Values
""" Imputing missing values using the mean or median dependend on skewness ±2 in every column from 5 and onwards
    unless there is missing more than 40% of the data, in that case it does nothing
    #idenitfy variables fitting whats normal
    """
for col in original_df.iloc[:,4:]:
            
    if original_df[col].isnull().any() and  -2 < (original_df[col].skew(axis=0,skipna=True,level=None,numeric_only=None)) < 2 and ((original_df[col].isnull().sum())/len(original_df))<0.45:
        
        colMean = original_df[col].mean()
        
        original_df[col] = original_df[col].fillna(colMean).round(2)
            
    elif original_df[col].isnull().any() and -2 > (original_df[col].skew(axis=0,skipna=True,level=None,numeric_only=None)) or 2 < (original_df[col].skew(axis=0,skipna=True,level=None,numeric_only=None)) and ((original_df[col].isnull().sum())/len(original_df))<0.45:
        
        colMedian = original_df[col].median()
        
        original_df[col] = original_df[col].fillna(colMedian).round(2)


#check % missing data
(original_df.isnull().sum()/len(original_df)).round(2)

(original_df.isnull().sum().sum())

#No missing values 

######## Outliers check manually version
#Loop that saves a png with a boxplot for each column.
#Mainly done this way, because the team needs to look at all column boxplots
#and plots load lazily, so saving them in a loop is the fastest way.
"""for n in original_df.iloc[:,4:]:
    fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
    original_df.boxplot(column = [n],
                   vert = False,
                   manage_xticks = True,
                   patch_artist = False,
                   meanline = True,
                   showmeans = True,)
    fig.savefig(n+'.png')
"""

#create df without the missing value flags and without europe data
#original_df = original_df.set_index('Hult_Team_Regions')
world = original_df.iloc[0:44, 4:29]
#append the remaining countries minus the "world" row at the bottom
world = world.append(original_df.iloc[59:-1, 4:29])

#should be 202 rows
world.shape

#lists to hold the worlds columns min and max acceptable z-values
world_z_low = []
world_z_max = []



#for loop to fill world's z lists with uper and lower limits
for n in enumerate(world):
    
    world_1 = world.iloc[:, n[0]]
    world_1 = world_1.astype(float)
    sd_1 = statistics.stdev(world_1)
    m_1 = statistics.mean(world_1)
    world_z_low.append(m_1 - sd_1)
    world_z_max.append(m_1 + sd_1)


#for loop to flag world's outliers
worldOutlier = pd.DataFrame()   

for n in enumerate(world):
    worldOutlier['o_'+n[1]] = 0    
    
    for r in enumerate(world[n[1]]):
       
        if r[1] > world_z_max[n[0]]:
            #outliers on high end are flagged 1
            worldOutlier.loc[r[0],'o_'+n[1]] = 1
            
        elif r[1] < world_z_low[n[0]]:
            #outliers on low end are flagged 1
            worldOutlier.loc[r[0],'o_'+n[1]] = 1
               
        else:
            worldOutlier.loc[r[0],'o_'+n[1]] = 0

#reset world's index, then save an "out version and a world version to join later       
world = world.reset_index()
worldOut = world.join(worldOutlier)
world = world.join(worldOutlier)


###
#Repeat the same flagging process for europe df based on the world flags
###

#df holding europes data without missing value flags
europe = original_df.iloc[44:59, 4:29]

#should be 15 rows
#europe.shape

#for loop to fill EU's z lists with uper and lower limits
for n in enumerate(europe):
    
    europe_1 = europe.iloc[:, n[0]]
    europe_1 = europe_1.astype(float)
    e_sd_1 = statistics.stdev(europe_1)
    e_m_1 = statistics.mean(europe_1)
    world_z_low.append(e_m_1 - e_sd_1)
    world_z_max.append(e_m_1 + e_sd_1)

#for loop to flag EU's outliers
europeOutlier = pd.DataFrame()    

for n in enumerate(europe):
    europeOutlier['o_'+n[1]] = 0    
    
    for r in enumerate(europe[n[1]]):
       
        if r[1] > world_z_max[n[0]]:
            #outliers on high end are flagged 1
            europeOutlier.loc[r[0],'o_'+n[1]] = 1
            
        elif r[1] < world_z_low[n[0]]:
            #outliers on low end are flagged 1
            europeOutlier.loc[r[0],'o_'+n[1]] = 1
               
        else:
            europeOutlier.loc[r[0],'o_'+n[1]] = 0

#reset europes's index, then save an "out version and a europe version to join later
europe = europe.reset_index()
europeOut = europe.join(europeOutlier)
europe = europe.join(europeOutlier)

#joing missing value flags
worldMissFlags = original_df.iloc[0:44, 30:]
worldMissFlags = worldMissFlags.append(original_df.iloc[59:-1, 30:])

#make sure they have the same indexies
world = world.set_index('country_code')

#join the two, so world holds all values including miss flags and outlier flags
world = world.join(worldMissFlags)

#repeat for europe
europeMissFlags = original_df.iloc[44:59, 30:]

#make sure they have the same indexies
europe = europe.set_index('country_code')

#join the two europe holds all flags
europe = europe.join(europeMissFlags)

#doublechecking % missing values
(world.isnull().sum()/len(world)).round(2)
(europe.isnull().sum()/len(europe)).round(2)

#saving both datafiles with flags of outliers and missing values
"""
world.to_excel('WorldOMflags.xlsx')
europe.to_excel('EuropeOMflags.xlsx')
"""

#declared versions without missing values but with outliers
#worldOut
#europeOut
#save to excel with two sheets holding each df
writer = pd.ExcelWriter('WorEUSheet.xlsx', engine='xlsxwriter')

# Write each dataframe to a different worksheet.
worldOut.to_excel(writer, sheet_name='Sheet1')
europeOut.to_excel(writer, sheet_name='Sheet2')

# Close the Pandas Excel writer and output the Excel file.
writer.save()

#save to individual
"""
worldOut.to_excel('WorldOut.xlsx')
europeOut.to_excel('EuropeOut.xlsx')
"""

#save to one file with column dedicated to tell europe from world
worldOut2 = worldOut.copy()
worldOut2.insert(0,'DataSet','World')

europeOut2 = europeOut.copy()
europeOut2.insert(0,'DataSet','Europe')

newWorld = europeOut2.append(worldOut2)

#newWorld.to_excel('NewWorld.xlsx')

#############################################
# Correlation matrix for world
#############################################
df_corr_world = world.iloc[:,:25].corr().round(2)

print(df_corr_world)

#Heatmap for world saved as png
plt.subplots(figsize=(20,15))
sns.heatmap(df_corr_world,
            cmap = 'Blues',
            square = True,
            annot = True,
            linecolor = 'black',
            linewidths = 0.5)
plt.savefig('AnoWorldHeatMap.png')


#############################################
# Correlation matrix for europe
#############################################
df_corr_europe = europe.iloc[:,:25].corr().round(2)

print(df_corr_europe)

#Heatmap for europe saved as png
plt.subplots(figsize=(20,15))
sns.heatmap(df_corr_europe,
            cmap = 'Blues',
            square = True,
            annot = True,
            linecolor = 'black',
            linewidths = 0.5)
plt.savefig('AnoEuropeHeatMap.png')



#########################################
# Analyzing outlier flags
#########################################

#seeing europes number of outliers compared to the rest of the world
for n in europeOut.iloc[:,26:]:
    
    v = europeOut[n].abs().sum()
    print(f'{n} = {v}')
    
###########################################
#Graphics for the presentation etc.
###########################################

#From here on the code is constantly being changed 
#depending on needs for the presentation 
#and is therefore not commented as thoroughly as the code above

                ####            ####
                ####            ####
        ###################################
        ###################################
                ####            ####
                ####            ####
                ####            ####        
        ###################################
        ###################################
                ####            ####
                ####            ####
    
#lmplot for child mortality and agriculture employment
sns.lmplot(x = 'child_mortality_per_1k',
                     y = 'pct_agriculture_employment',
                         data = world,
                         scatter_kws = {'color':'grey'})
plt.title('CORRELATION WITH TRENDLINE')
plt.xlabel('CHILD MORTALITY PER 1K')
plt.ylabel('AGRICULTURE EMPLOYMENT PCT')
plt.tight_layout()
plt.savefig('CorrelationChildAgricultureEmployment.png')
plt.show()

#lmplot for internet and service employment
sns.lmplot(x = 'internet_usage_pct',
                     y = 'pct_services_employment',
                         data = world,
                         scatter_kws = {'color':'grey'})
plt.title('CORRELATION WITH TRENDLINE')
plt.xlabel('INTERNET USAGE PCT')
plt.ylabel('SERVICE EMPLOYMENT PCT')
plt.tight_layout()
plt.savefig('CorrelationInternetServiceEmployment.png')
plt.show()



####

europeOut.info()
europe.mean()

europe_analyze = europeOut.copy()
europe_analyze.loc['europe_mean'] = europeOut.mean()
europe_analyze.loc['world_mean'] = worldOut.mean()
europe_analyze.loc['europe_mean', 'country_code'].rename({'EU'})
europe_analyze = europe_analyze.drop(labels=['world_mean', 'europe_mean'], axis=0)

europe_analyze = europe_analyze.set_index('country_code')
europe_analyze = europe_analyze.reset_index()


#Delete rows
'''
europe_analyze = europe_analyze.drop(labels=['mean', 'europe_mean'], axis=0)
'''


europe_analyze_1 = europe_analyze.loc[:, 'internet_usage_pct'].astype(int)


#Create bar graph for columns in EU with > 30% outliers
'''
plt.subplot(2, 1, 1)
plt.hist(x = worldOut_hist_internet,
         data = worldOut,
         bins = 100,
         cumulative = True,
         log = False,
         color = 'black',
         dtype = step
         )
plt.xlabel("Price")
plt.show()
'''

###INTERNET USAGE
worldOut_hist_internet = worldOut.loc[:, 'internet_usage_pct'].sort_values( ascending=True)


plt.subplot(2, 1, 1)
sns.distplot(worldOut_hist_internet,
             bins = 'fd',
             color = 'g',
             hist = True,
             kde = True,
             rug = False)

plt.xlabel('OBSERVATIONS')
plt.ylabel('DISTRIBUTION')
plt.title("WORLD INTERNET USAGE PERCENTAGE")

worldOut_hist_internet_limit_lo = world_z_low[15]
worldOut_hist_internet_limit_hi = world_z_max[15]


plt.axvline(x = worldOut_hist_internet_limit_lo,
            linestyle = '--',
            color = 'red')

plt.axvline(x = worldOut_hist_internet_limit_hi,
            linestyle = '--',
            color = 'red')


plt.subplot(2, 1, 2)
plt.bar(
         x = 'country_code',
         height = 'internet_usage_pct',
         data = europe_analyze,
         color = ['gray','red','red','red','red','red','red','red','gray',
                  'gray','gray','gray','gray','gray','red','orange',
                  'purple'],
        
         )

plt.xticks(rotation=90)
plt.title("EUROPE INTERNET USAGE PERCENTAGE")
plt.xlabel("COUNTRIES")
plt.ylabel("% OF THE POPULATION")


plt.tight_layout()
plt.savefig('INTERNET.png')

plt.show()

###URBAN GROWTH PCT

worldOut_hist_urbanpoppct = worldOut.loc[:, 'urban_population_growth_pct'].sort_values( ascending=True)


plt.subplot(2, 1, 1)
sns.distplot(worldOut_hist_urbanpoppct,
             bins = 'fd',
             color = 'g',
             hist = True,
             kde = True,
             rug = False)

plt.xlabel('OBSERVATIONS')
plt.ylabel('DISTRIBUTION')
plt.title("WORLD URBAN POPULATION GROWTH PERCENTAGE")

worldOut_hist_urbanpoppct_limit_lo = world_z_low[24]
worldOut_hist_urbanpoppct_limit_hi = world_z_max[24]


plt.axvline(x = worldOut_hist_urbanpoppct_limit_lo,
            linestyle = '--',
            color = 'red')

plt.axvline(x = worldOut_hist_urbanpoppct_limit_hi,
            linestyle = '--',
            color = 'red')


plt.subplot(2, 1, 2)
plt.bar(
         x = 'country_code',
         height = 'urban_population_growth_pct',
         data = europe_analyze,
         color =['gray','gray','gray','gray','red','red','gray','red','red','red',
                 'gray','red','red','gray','red','orange','purple'],
         )

plt.xticks(rotation=90)
plt.title("EUROPE URBAN POPULATION GROWTH PERCENTAGE")
plt.xlabel("COUNTRIES")
plt.ylabel("ANNUAL % INCREASE")


plt.tight_layout()
plt.savefig('URBANPOPPCT.png')

plt.show()


'''
plt.hist(x = 'price',
         data = diamonds,
         bins = 'fd',
         cumulative = False,
         histtype = 'barstacked',
         orientation = 'horizontal'
         )
plt.xlabel("Price")
plt.show()
'''
####ADULT LITERACY RATE
worldOut_hist_adultlit = worldOut.loc[:, 'adult_literacy_pct'].sort_values( ascending=True)


plt.subplot(2, 1, 1)
sns.distplot(worldOut_hist_adultlit,
             bins = 'fd',
             color = 'g',
             hist = True,
             kde = True,
             rug = False)

plt.xlabel('OBSERVATIONS')
plt.ylabel('DISTRIBUTION')
plt.title("WORLD ADULT LITERACY PERCENTAGE")

worldOut_hist_adultlit_limit_lo = world_z_low[17]
worldOut_hist_adultlit_limit_hi = world_z_max[17]


plt.axvline(x = worldOut_hist_adultlit_limit_lo,
            linestyle = '--',
            color = 'red')

plt.axvline(x = worldOut_hist_adultlit_limit_hi,
            linestyle = '--',
            color = 'red')


plt.subplot(2, 1, 2)
plt.bar(
         x = 'country_code',
         height = 'adult_literacy_pct',
         data = europe_analyze,
         color =['gray','gray','gray','gray','gray','red','red','gray',
                 'gray','red','red','gray','gray','gray','red',
                 'orange','purple'],
         )

plt.xticks(rotation=90)
plt.title("EUROPE ADULT LITERACY RATE PERCENTAGE")
plt.xlabel("COUNTRIES")
plt.ylabel("% OF POPULATION \n(> 15 YO)")


plt.tight_layout()

plt.savefig('ADULTLIT.png')
plt.show()


###INDUSTRY EMPLOYMENT PCT
worldOut_hist_industryemploy = worldOut.loc[:, 'pct_industry_employment'].sort_values( ascending=True)

plt.subplot(2, 1, 1)
sns.distplot(worldOut_hist_industryemploy,
             bins = 'fd',
             color = 'g',
             hist = True,
             kde = True,
             rug = False)

plt.xlabel('OBSERVATIONS')
plt.ylabel('DISTRIBUTION')
plt.title("WORLD INDUSTRY EMPLOYMENT PERCENTAGE")

worldOut_hist_industryemploy_limit_lo = world_z_low[8]
worldOut_hist_industryemploy_limit_hi = world_z_max[8]


plt.axvline(x = worldOut_hist_industryemploy_limit_lo,
            linestyle = '--',
            color = 'red')

plt.axvline(x = worldOut_hist_industryemploy_limit_hi,
            linestyle = '--',
            color = 'red')

plt.subplot(2, 1, 2)
plt.bar(
         x = 'country_code',
         height = 'pct_industry_employment',
         data = europe_analyze,
         color =['gray','gray','red','gray','red','gray','gray','red','red','gray',
                 'gray','red','red','gray','red','orange','purple'],
         )
plt.xticks(rotation=90)
plt.title("EUROPE INDUSTRY EMPLOYMENT PERCENTAGE")
plt.xlabel("COUNTRIES")
plt.ylabel("% OF TOTAL EMPLOYMENT")


plt.tight_layout()
plt.savefig('INDUSEMPLOY.png')

plt.show()

### PIE GRAPH FOR EUROPE TOTAL LABOR FORCE

indus_mean = europeOut.loc[:, 'pct_industry_employment'].mean()
service_mean = europeOut.loc[:, 'pct_services_employment'].mean()
agri_mean = europeOut.loc[:, 'pct_agriculture_employment'].mean()
emplyment_total = indus_mean + service_mean + agri_mean

unemploy_mean = europeOut.loc[:, 'unemployment_pct'].mean()
employment_total = 100 - unemploy_mean

employ_indus = ((indus_mean /100) * employment_total)
employ_service = ((service_mean /100) * employment_total)
employ_agri = ((agri_mean/100) * employment_total)

# Data to plot
labels = 'Industry', 'Services', 'Agriculture', 'Unemployed'
sizes = [employ_indus, employ_service , employ_agri, unemploy_mean]
colors = ['lightskyblue', 'blue', 'lightblue', 'red']
explode = (0, 0, 0, 0.2)  # explode 1st slice
 
# Plot

plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
 
plt.axis('equal')

plt.title('EUROPE TOTAL LABOR FORCE')

plt.savefig('EuropeTLF.png')
plt.show()

###BAR GRAPH FOR EUROPE VS WORLD TOTAL LABOR FORCE
'''
indus_mean = europeOut.loc[:, 'pct_industry_employment'].mean()
service_mean = europeOut.loc[:, 'pct_services_employment'].mean()
agri_mean = europeOut.loc[:, 'pct_agriculture_employment'].mean()
unemploy_mean = europeOut.loc[:, 'unemployment_pct'].mean()

indus_mean_world = worldOut.loc[:, 'pct_industry_employment'].mean()
service_mean_world = worldOut.loc[:, 'pct_services_employment'].mean()
agri_mean_world = worldOut.loc[:, 'pct_agriculture_employment'].mean()
unemploy_mean_world = worldOut.loc[:, 'unemployment_pct'].mean()

tlf = pd.DataFrame()

tlf_main = worldOut.iloc[:, 8:11]
tlf_main.loc['europe_mean'] = europeOut.iloc[:, 8:11].mean()
tlf_main.loc['world_mean'] = worldOut.iloc[:, 8:11].mean()


tlf_main_1 = worldOut.iloc[:, 23]
tlf_main_1.loc['europe_mean'] = europeOut.iloc[:, 23].mean()
tlf_main_1.loc['world_mean'] = worldOut.iloc[:, 23].mean()

tlf_main_2 = tlf_main.join(tlf_main_1)

tlf_main_3 = tlf_main_2.iloc[202:, :]



labels = 'Industry', 'Services', 'Agriculture', 'Unemployed'

plt.bar(
         x = 'Index',
         height = 'pct_agriculture_employment' ,
         data = tlf_main_3,
         color = 'red',        
         )
plt.xticks(rotation=90)
plt.title("EUROPE INTERNET USAGE PERCENTAGE")
plt.xlabel("COUNTRIES")
plt.ylabel("% OF THE POPULATION")
plt.tight_layout()
plt.show()
'''

### PIE GRAPH FOR EUROPE GDP

taxrev_mean = europeOut.loc[:, 'tax_revenue_pct_gdp'].mean()
fdi_mean = europeOut.loc[:, 'fdi_pct_gdp'].mean()
export_mean = europeOut.loc[:, 'exports_pct_gdp'].mean()
others = 100 - (taxrev_mean + fdi_mean + export_mean)

labels = 'Tax Revenue', 'FDI', 'Exports', 'Others'
sizes = [taxrev_mean, fdi_mean , export_mean, others]
colors = ['brown', 'red', 'pink', 'purple']
explode = (0.04, 0.04, 0.03, 0.04)  # explode 1st slice
 
# Plot

plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
 
plt.axis('equal')

plt.title('EUROPE GROSS DOMESTIC PRODUCT')

plt.savefig('EuropeGDP.png')
plt.show()

#######################################################
##### additional graphs, not used in final presentation

internet_sel = europeOut.loc[:,['internet_usage_pct']]

#pairplot for internet usage %
sns.pairplot(data = internet_sel)
plt.tight_layout()
plt.show()

#lmplot for internet usage % and child mortality
my_resid = sns.residplot(x = 'internet_usage_pct',
                         y = 'child_mortality_per_1k',
                         data = world,
                         lowess = True,
                         color = 'r',
                         line_kws = {'color':'black'})
plt.tight_layout()
#plt.savefig("internet usage ~ child mortality Plot.png")
plt.show()

#paitplot
sns.pairplot(data = world,
             x_vars = ['internet_usage_pct', 'child_mortality_per_1k'],
             y_vars = ['adult_literacy_pct']
             )
plt.tight_layout()
#plt.savefig('Adult lit Pairplot.png')
plt.show()

#reglot for child mortality and agriculture employment
sns.regplot(x = 'child_mortality_per_1k',
                     y = 'pct_agriculture_employment',
                     data = world,
                     x_estimator = pd.np.mean,
                     x_bins = 8)
plt.tight_layout()
plt.show()

              
#adding means as the new bottom row
europeOut.loc['mean'] = worldOut.mean()

print(europeOut.iloc[1,25])
print(europeOut.iloc[:15,:25])


#regplot between child mortality and adult literacy
my_plot = sns.regplot(x = 'child_mortality_per_1k',
                     y = 'adult_literacy_pct',
                     data = world)
plt.tight_layout()
plt.show()

#snail graph regplot for child mortality and adult literacy
my_plot2 = sns.regplot(x = 'child_mortality_per_1k',
                     y = 'adult_literacy_pct',
                     data = world,
                     x_estimator = pd.np.mean,
                     x_bins = 8)
plt.tight_layout()
plt.show()

#a bit of everything - plot
my_plot = sns.jointplot(x = 'child_mortality_per_1k',
                     y = 'adult_literacy_pct',
                     kind = 'reg',
                     joint_kws={'color':'blue'},
                     data = world,
                     x_estimator = pd.np.mean,
                     x_bins = 8)
plt.tight_layout()
plt.show()

#residuals plot for child mortality and adult literacy
my_resid = sns.residplot(x = 'child_mortality_per_1k',
                     y = 'adult_literacy_pct',
                         data = world,
                         lowess = True,
                         color = 'r',
                         line_kws = {'color':'black'})
plt.tight_layout()
plt.show()

#child mortality and adult literacy 
sns.violinplot(x = 'child_mortality_per_1k',
               y = 'adult_literacy_pct',
               data = world,
               orient = 'v')
plt.show()

###############################################################################
# Univariate OLS Regression
###############################################################################

# OLS linear regression can be run usning 'smf.ols'
lm_child_literacy = smf.ols(formula = 'child_mortality_per_1k ~ adult_literacy_pct',
                         data = world)

results = lm_child_literacy.fit()

print(results.summary())

# The summary2 function in results
print(results.summary2())

# Accessing the results directory
dir(results)

# Residuals
residuals = results.resid

print(residuals)

# Fitted values
predicted_child = results.fittedvalues

print(predicted_child)


# More about results
dir(results)

# Let's utlize results.params
#childmortality at index 20
world_z_low[20]
world_z_max[20]

child_weight = 20
pred_adult = results.params[0] + results.params[1] * child_weight

# A function based on our regression model
def price_pred():
    """Predicts price based on the carat weight."""
    
    import statsmodels.formula.api as smf
    
    results = smf.ols(formula = 'child_mortality_per_1k ~ adult_literacy_pct',
                      data = world).fit()
    
    child_weight = int(input("Input child mortality rate per 1k > "))

    pred_adult = results.params[0] + results.params[1] * child_weight

    print(f"""
      
A child mortality rate of that size leads to an adult literacy rate of approximately {round(pred_adult, 2)}.

      """)

price_pred()


# Creating a DataFrame with original, predicted, and residual values
predict_df = pd.DataFrame(world['child_mortality_per_1k'])

predict_df['Child mortality'] = pd.DataFrame(newWorld['child_mortality_per_1k'])

predict_df['Predicted'] = predicted_child.round(2)

predict_df['Residuals'] = residuals.round(2)

predict_df['Abs_Residuals'] = residuals.round(2).abs()

print(predict_df)

# Add's the absolute values of the residuals
predict_df = predict_df.sort_values(by = 'Abs_Residuals', ascending = False)

# Investigating
residuals_df = residuals.round(2).abs()

residuals_df = predict_df.sort_values(by = 'Abs_Residuals', ascending = False)

print(residuals_df)



fig, ax = plt.subplots(figsize=(12,8))
fig = sm.graphics.influence_plot(results,
                                 ax = ax,
                                 criterion = 'cooks')

####################################################
# More Regression Techniques 
####################################################

# Bonferroni outlier test
test = results.outlier_test()

print('Bad data points (bonf(p) < 0.05):')
print(test[test.iloc[:, 2] < 0.05])

newWorld2 = newWorld.copy()

# Creating binary matricies for categorical variable DataSet (Europe/World)
channel_dummies = pd.get_dummies(list(newWorld2['DataSet']))

print(channel_dummies)

# concatenating binaries matricies with the diamonds dataset
#newWorld3 = pd.concat(
#        [newWorld2.loc[:,:],
#         channel_dummies],
#         axis = 1)

df = newWorld.iloc[:,2:]
print(df)

#find correlations to adult literacy
df.corr()['adult_literacy_pct'].sort_values()


from sklearn import linear_model

x1 = df
y1 = df['adult_literacy_pct']

#function fits a linear model
lm = linear_model.LinearRegression()
model = lm.fit(x1,y1)


predictions = lm.predict(x1)
print(predictions)[0:5]


###### THE END ######