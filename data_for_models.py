import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from dateutil.relativedelta import relativedelta
from uszipcode import SearchEngine
search = SearchEngine(simple_zipcode = True)

### Pull CSV files ###
zori = pd.read_csv('./data/cleandata/clean_zori.csv').drop('Unnamed: 0', axis = 1)
hpi = pd.read_csv('./data/cleandata/clean_hpi.csv').drop('Unnamed: 0', axis = 1)
zhvi = pd.read_csv('./data/cleandata/clean_zhvi.csv').drop('Unnamed: 0', axis = 1)
airq = pd.read_csv('./data/cleandata/clean_airq.csv').drop('Unnamed: 0', axis = 1)
population = pd.read_csv('./data/cleandata/clean_population.csv').drop('Unnamed: 0', axis = 1)
unemployment = pd.read_csv('./data/cleandata/clean_unemployment.csv').drop('Unnamed: 0', axis = 1)
education = pd.read_csv('./data/cleandata/clean_education.csv').drop('Unnamed: 0', axis = 1)
permits = pd.read_csv('./data/cleandata/clean_permits.csv').drop('Unnamed: 0', axis = 1)
IandH = pd.read_csv('./data/cleandata/clean_IandH.csv').drop('Unnamed: 0', axis = 1)
pce = pd.read_csv('./data/cleandata/clean_pce.csv').drop('Unnamed: 0', axis = 1)
vacancy = pd.read_csv('./data/cleandata/clean_vacancy.csv').drop('Unnamed: 0', axis = 1)
jobs = pd.read_csv('./data/cleandata/clean_jos.csv').drop('Unnamed: 0', axis = 1)
commute_worker = pd.read_csv('./data/cleandata/clean_commute_worker.csv').drop('Unnamed: 0', axis = 1)
grapi = pd.read_csv('./data/cleandata/clean_grapi.csv').drop('Unnamed: 0', axis = 1)
population_density = pd.read_csv('./data/cleandata/clean_population_density.csv').drop('Unnamed: 0', axis = 1)
income_inequality = pd.read_csv('./data/cleandata/clean_income_inequality.csv').drop('Unnamed: 0', axis = 1)

### Intermediate matrices for merging dataframes before training ###
one_year_forecast_train = zori[['Year', 'Month', 'Year_Month']]
one_year_forecast_train = pd.concat([one_year_forecast_train,
                                     pd.DataFrame(one_year_forecast_train['Year'].map(lambda year: year - 1))],
                                    axis = 1)
one_year_forecast_train.columns = ['Year', 'Month', 'Year_Month', 'Year2']
one_year_forecast_train = pd.concat([one_year_forecast_train,
                                     pd.DataFrame(one_year_forecast_train['Year'].map(lambda year: year - 2))],
                                    axis = 1)
one_year_forecast_train.columns = ['Year', 'Month', 'Year_Month', 'Year2', 'Year3']
one_year_forecast_train['Year_Month2'] = one_year_forecast_train['Year2'].map(str) + '_' +\
                                         one_year_forecast_train['Month'].map(str)
one_year_forecast_train['Year_Month3'] = one_year_forecast_train['Year3'].map(str) + '_' +\
                                         one_year_forecast_train['Month'].map(str)
one_year_forecast_train = one_year_forecast_train[['Year_Month', 'Year_Month2', 'Year_Month3']]
one_year_forecast_train.drop_duplicates(inplace = True)

### Intermediate matrices for merging dataframes before predicting ###
prediction_start_month = 10
prediction_start_year = 2020
prediction_horizon_months = 12

one_year_forecast_pred = pd.DataFrame({'Year': np.array(prediction_start_year).repeat(prediction_horizon_months),
                                       'Month': np.arange(prediction_start_month,
                                                          prediction_start_month + prediction_horizon_months)})
one_year_forecast_pred['Year_Month'] = one_year_forecast_pred['Year'].map(str) + '_' +\
                                       one_year_forecast_pred['Month'].map(str)
one_year_forecast_pred['Year_Month'] = one_year_forecast_pred['Year_Month'].map(lambda x: str(int(x[:4]) + 1) + '_' +\
                                                                                str(int(x[x.find('_') + 1:]) - 12) if \
                                                                                int(x[x.find('_') + 1:]) > 12 else x)
one_year_forecast_pred['Year'] = one_year_forecast_pred['Year_Month'].map(lambda x: int(x[:4]))
one_year_forecast_pred['Month'] = one_year_forecast_pred['Year_Month'].map(lambda x: int(x[x.find('_') + 1:]))
one_year_forecast_pred = pd.concat([one_year_forecast_pred,
                                    pd.DataFrame(one_year_forecast_pred['Year'].map(lambda year: year - 1))],
                                   axis = 1)
one_year_forecast_pred.columns = ['Year', 'Month', 'Year_Month', 'Year2']
one_year_forecast_pred = pd.concat([one_year_forecast_pred,
                                    pd.DataFrame(one_year_forecast_pred['Year'].map(lambda year: year - 2))],
                                   axis = 1)
one_year_forecast_pred.columns = ['Year', 'Month', 'Year_Month', 'Year2', 'Year3']
one_year_forecast_pred['Year_Month2'] = one_year_forecast_pred['Year2'].map(str) + '_' +\
                                        one_year_forecast_pred['Month'].map(str)
one_year_forecast_pred['Year_Month3'] = one_year_forecast_pred['Year3'].map(str) + '_' +\
                                        one_year_forecast_pred['Month'].map(str)
one_year_forecast_pred = one_year_forecast_pred[['Year_Month', 'Year_Month2', 'Year_Month3']]

### Final Dataframe for Training (ZORI) ###
df = zori[zori['ZORI'] <= 4000].copy() # Exclude outliers
df = pd.merge(df, one_year_forecast_train, on = 'Year_Month')
df = pd.merge(df, zori[['Year_Month', 'ZipCode', 'ZORI']], left_on = ['Year_Month2', 'ZipCode'],
              right_on = ['Year_Month', 'ZipCode'], how = 'left')
df.drop('Year_Month_y', axis = 1, inplace = True)

df = pd.merge(df, zhvi[['Year_Month', 'ZipCode', 'ZHVI']], left_on = ['Year_Month2', 'ZipCode'],
              right_on = ['Year_Month', 'ZipCode'], how = 'left')
df.drop('Year_Month', axis = 1, inplace = True)

df = pd.merge(df, hpi[['Year_Month', 'ZipCode', 'HPI']], left_on = ['Year_Month3', 'ZipCode'],
              right_on = ['Year_Month', 'ZipCode'], how = 'left')
df.drop('Year_Month', axis = 1, inplace = True)

df = pd.merge(df, airq[['County', 'State', 'Year_Month', 'AQI']], left_on = ['County', 'State', 'Year_Month2'],
              right_on = ['County', 'State', 'Year_Month'], how = 'left')
df.drop('Year_Month', axis = 1, inplace = True)

df = pd.merge(df, population[['County', 'State', 'Year_Month', 'Population']],
              left_on = ['County', 'State', 'Year_Month3'], right_on = ['County', 'State', 'Year_Month'], how = 'left')
df.drop('Year_Month', axis = 1, inplace = True)

df = pd.merge(df, unemployment[['County', 'State', 'Year_Month', 'Unemployment']],
             left_on = ['County', 'State', 'Year_Month2'], right_on = ['County', 'State', 'Year_Month'], how = 'left')
df.drop('Year_Month', axis = 1, inplace = True)

df = pd.merge(df, education[['County', 'State', 'Year_Month', 'Percent Bachelors']],
              left_on = ['County', 'State', 'Year_Month3'], right_on = ['County', 'State', 'Year_Month'], how = 'left')
df.drop('Year_Month', axis = 1, inplace = True)

df = pd.merge(df, permits[['State', 'Units', 'Year_Month']], left_on = ['State', 'Year_Month2'],
              right_on = ['State', 'Year_Month'], how = 'left')
df.drop('Year_Month', axis = 1, inplace = True)

df = pd.merge(df, IandH[['County', 'State', 'Year_Month', 'Total_Households', 'Med_income']],
             left_on = ['County', 'State', 'Year_Month3'], right_on = ['County', 'State', 'Year_Month'], how = 'left')
df.drop('Year_Month', axis = 1, inplace = True)

df = pd.merge(df, pce[['PCE', 'Year_Month']], left_on = 'Year_Month2', right_on = 'Year_Month')
df.drop('Year_Month', axis = 1, inplace = True)

df = pd.merge(df, vacancy[['County', 'State', 'Year_Month', 'Rental Vacancy Rate']],
              left_on = ['County', 'State', 'Year_Month3'], right_on = ['County', 'State', 'Year_Month'], how = 'left')
df.drop('Year_Month', axis = 1, inplace = True)

df = pd.merge(df, jobs[['State', 'Job Openings', 'Year_Month']], left_on = ['State', 'Year_Month2'],
              right_on = ['State', 'Year_Month'], how = 'left')
df.drop('Year_Month', axis = 1, inplace = True)

df = pd.merge(df, commute_worker[['County', 'State', 'CommuteTime', 'Salwrkr', 'Govwrkr', 'Year_Month']],
              left_on = ['County', 'State', 'Year_Month3'], right_on = ['County', 'State', 'Year_Month'], how = 'left')
df.drop('Year_Month', axis = 1, inplace = True)

df = pd.merge(df, grapi[['County', 'State', 'GRAPI', 'Year_Month']], left_on = ['County', 'State', 'Year_Month3'],
              right_on = ['County', 'State', 'Year_Month'], how = 'left')
df.drop('Year_Month', axis = 1, inplace = True)

df = pd.merge(df, population_density[['County', 'State', 'P_Density', 'Year_Month']],
              left_on = ['County', 'State', 'Year_Month3'], right_on = ['County', 'State', 'Year_Month'], how = 'left')
df.drop('Year_Month', axis = 1, inplace = True)

df = pd.merge(df, income_inequality[['County', 'State', 'Gini_Index', 'Year_Month']],
              left_on = ['County', 'State', 'Year_Month3'], right_on = ['County', 'State', 'Year_Month'], how = 'left')
df.drop('Year_Month', axis = 1, inplace = True)

df = df[['Year', 'Month', 'ZipCode', 'ZORI_x', 'ZORI_y', 'ZHVI', 'HPI', 'AQI', 'Population', 'Unemployment',
         'Percent Bachelors', 'Units', 'Total_Households', 'Med_income', 'PCE', 'Rental Vacancy Rate', 'Job Openings',
        'CommuteTime', 'Salwrkr', 'Govwrkr', 'GRAPI', 'P_Density', 'Gini_Index']]

df.columns = ['Year', 'Month', 'ZipCode', 'ZORI', 'ZORI_lagged_1', 'ZHVI_lagged_1', 'HPI_lagged_2', 'AQI_lagged_1',
              'Population_lagged_2', 'Unemployment_lagged_1', 'Percent Bachelors_lagged_2', 'Permits_lagged_1',
              'Total_Households_lagged_2', 'Med_Income_lagged_2', 'PCE_lagged_1', 'Rental Vacancy Rate_lagged_2',
              'Job Openings_lagged_1', 'CommuteTime_lagged_2', 'Salwrkr_lagged_2', 'Govwrkr_lagged_2', 'GRAPI_lagged_2',
             'P_Density_lagged_2', 'Gini_Index_lagged_2']

# df_clean = df.drop('ZORI_lagged_1', axis = 1)
df_clean = df.copy()

# Dropping all NaN values because data is not available in certain zipcodes and counties
# These are missing data even after imputing.
df_clean = df_clean[~df_clean['ZORI_lagged_1'].isnull()]
df_clean = df_clean[~df_clean['ZHVI_lagged_1'].isnull()]
df_clean = df_clean[~df_clean['HPI_lagged_2'].isnull()]
df_clean = df_clean[~df_clean['AQI_lagged_1'].isnull()]
df_clean = df_clean[~df_clean['Percent Bachelors_lagged_2'].isnull()]
df_clean = df_clean[~df_clean['Rental Vacancy Rate_lagged_2'].isnull()]

### Final DataFrame for Prediction (ZORI) ###
df_pred = zori[zori['ZORI'] <= 4000].copy() # Exclude outliers
df_pred = pd.merge(df_pred, one_year_forecast_pred, left_on = 'Year_Month', right_on = 'Year_Month2')
df_pred.drop('Year_Month_x', axis = 1, inplace = True)
df_pred['Year'] = df_pred['Year'] + 1

df_pred = pd.merge(df_pred, zhvi[['Year_Month', 'ZipCode', 'ZHVI']], left_on = ['Year_Month2', 'ZipCode'],
                   right_on = ['Year_Month', 'ZipCode'], how = 'left')
df_pred.drop('Year_Month', axis = 1, inplace = True)

df_pred = pd.merge(df_pred, hpi[['Year_Month', 'ZipCode', 'HPI']], left_on = ['Year_Month3', 'ZipCode'],
                   right_on = ['Year_Month', 'ZipCode'], how = 'left')
df_pred.drop('Year_Month', axis = 1, inplace = True)

df_pred = pd.merge(df_pred, airq[['County', 'State', 'Year_Month', 'AQI']], left_on = ['County', 'State', 'Year_Month2'],
                   right_on = ['County', 'State', 'Year_Month'], how = 'left')
df_pred.drop('Year_Month', axis = 1, inplace = True)

df_pred = pd.merge(df_pred, population[['County', 'State', 'Year_Month', 'Population']],
                   left_on = ['County', 'State', 'Year_Month3'], right_on = ['County', 'State', 'Year_Month'], how = 'left')
df_pred.drop('Year_Month', axis = 1, inplace = True)

df_pred = pd.merge(df_pred, unemployment[['County', 'State', 'Year_Month', 'Unemployment']],
                   left_on = ['County', 'State', 'Year_Month2'], right_on = ['County', 'State', 'Year_Month'], how = 'left')
df_pred.drop('Year_Month', axis = 1, inplace = True)

df_pred = pd.merge(df_pred, education[['County', 'State', 'Year_Month', 'Percent Bachelors']],
                   left_on = ['County', 'State', 'Year_Month3'], right_on = ['County', 'State', 'Year_Month'], how = 'left')
df_pred.drop('Year_Month', axis = 1, inplace = True)

df_pred = pd.merge(df_pred, permits[['State', 'Units', 'Year_Month']], left_on = ['State', 'Year_Month2'],
                   right_on = ['State', 'Year_Month'], how = 'left')
df_pred.drop('Year_Month', axis = 1, inplace = True)

df_pred = pd.merge(df_pred, IandH[['County', 'State', 'Year_Month', 'Total_Households', 'Med_income']],
                   left_on = ['County', 'State', 'Year_Month3'], right_on = ['County', 'State', 'Year_Month'], how = 'left')
df_pred.drop('Year_Month', axis = 1, inplace = True)

df_pred = pd.merge(df_pred, pce[['PCE', 'Year_Month']], left_on = 'Year_Month2', right_on = 'Year_Month')
df_pred.drop('Year_Month', axis = 1, inplace = True)

df_pred = pd.merge(df_pred, vacancy[['County', 'State', 'Year_Month', 'Rental Vacancy Rate']],
                   left_on = ['County', 'State', 'Year_Month3'], right_on = ['County', 'State', 'Year_Month'], how = 'left')
df_pred.drop('Year_Month', axis = 1, inplace = True)

df_pred = pd.merge(df_pred, jobs[['State', 'Job Openings', 'Year_Month']], left_on = ['State', 'Year_Month2'],
                   right_on = ['State', 'Year_Month'], how = 'left')
df_pred.drop('Year_Month', axis = 1, inplace = True)

df_pred = pd.merge(df_pred, commute_worker[['County', 'State', 'CommuteTime', 'Salwrkr', 'Govwrkr', 'Year_Month']],
                   left_on = ['County', 'State', 'Year_Month3'], right_on = ['County', 'State', 'Year_Month'], how = 'left')
df_pred.drop('Year_Month', axis = 1, inplace = True)

df_pred = pd.merge(df_pred, grapi[['County', 'State', 'GRAPI', 'Year_Month']],
                   left_on = ['County', 'State', 'Year_Month3'], right_on = ['County', 'State', 'Year_Month'], how = 'left')
df_pred.drop('Year_Month', axis = 1, inplace = True)

df_pred = pd.merge(df_pred, population_density[['County', 'State', 'P_Density', 'Year_Month']],
                   left_on = ['County', 'State', 'Year_Month3'], right_on = ['County', 'State', 'Year_Month'], how = 'left')
df_pred.drop('Year_Month', axis = 1, inplace = True)

df_pred = pd.merge(df_pred, income_inequality[['County', 'State', 'Gini_Index', 'Year_Month']],
                   left_on = ['County', 'State', 'Year_Month3'], right_on = ['County', 'State', 'Year_Month'], how = 'left')
df_pred.drop('Year_Month', axis = 1, inplace = True)

df_pred = df_pred[['Year', 'Month', 'ZipCode', 'ZORI', 'ZHVI', 'HPI', 'AQI', 'Population', 'Unemployment',
                   'Percent Bachelors', 'Units', 'Total_Households', 'Med_income', 'PCE', 'Rental Vacancy Rate',
                   'Job Openings', 'CommuteTime', 'Salwrkr', 'Govwrkr', 'GRAPI', 'P_Density', 'Gini_Index']]

df_pred.columns = ['Year', 'Month', 'ZipCode', 'ZORI_lagged_1', 'ZHVI_lagged_1', 'HPI_lagged_2', 'AQI_lagged_1',
                  'Population_lagged_2', 'Unemployment_lagged_1', 'Percent Bachelors_lagged_2', 'Permits_lagged_1',
                  'Total_Households_lagged_2', 'Med_Income_lagged_2', 'PCE_lagged_1', 'Rental Vacancy Rate_lagged_2',
                  'Job Openings_lagged_1', 'CommuteTime_lagged_2', 'Salwrkr_lagged_2', 'Govwrkr_lagged_2',
                   'GRAPI_lagged_2', 'P_Density_lagged_2', 'Gini_Index_lagged_2']

df_pred = df_pred[~df_pred['ZORI_lagged_1'].isnull()]
df_pred = df_pred[~df_pred['ZHVI_lagged_1'].isnull()]
df_pred = df_pred[~df_pred['HPI_lagged_2'].isnull()]
df_pred = df_pred[~df_pred['HPI_lagged_2'].isnull()]
df_pred = df_pred[~df_pred['AQI_lagged_1'].isnull()]
df_pred = df_pred[~df_pred['Population_lagged_2'].isnull()]
df_pred = df_pred[~df_pred['Permits_lagged_1'].isnull()]
df_pred = df_pred[~df_pred['Rental Vacancy Rate_lagged_2'].isnull()]
df_pred = df_pred[~df_pred['Job Openings_lagged_1'].isnull()]

### Final Dataframe for Training (% Change in ZORI) ###
df_change = df.drop('ZORI_lagged_1', axis = 1).copy()
df_change['Year_Month'] = df_change['Year'].map(str) + '_' + df_change['Month'].map(str)
df_change = pd.merge(df_change, one_year_forecast_train, on = 'Year_Month')
df_change = pd.merge(df_change, df_change, left_on = ['Year_Month2', 'ZipCode'], right_on = ['Year_Month', 'ZipCode'],
                     how = 'left')

df_change['ZORI_x'] = (df_change['ZORI_x'] - df_change['ZORI_y']) / df_change['ZORI_y']
df_change['ZHVI_lagged_1_x'] = (df_change['ZHVI_lagged_1_x'] - df_change['ZHVI_lagged_1_y']) / df_change['ZHVI_lagged_1_y']
df_change['HPI_lagged_2_x'] = (df_change['HPI_lagged_2_x'] - df_change['HPI_lagged_2_y']) / df_change['HPI_lagged_2_y']
df_change['AQI_lagged_1_x'] = (df_change['AQI_lagged_1_x'] - df_change['AQI_lagged_1_y']) / df_change['AQI_lagged_1_y']
df_change['Population_lagged_2_x'] = (df_change['Population_lagged_2_x'] - df_change['Population_lagged_2_y']) /\
                                     df_change['Population_lagged_2_y']
df_change['Unemployment_lagged_1_x'] = (df_change['Unemployment_lagged_1_x'] - df_change['Unemployment_lagged_1_y']) /\
                                        df_change['Unemployment_lagged_1_y']
df_change['Percent Bachelors_lagged_2_x'] = (df_change['Percent Bachelors_lagged_2_x'] -\
                                            df_change['Percent Bachelors_lagged_2_y']) /\
                                            df_change['Percent Bachelors_lagged_2_y']
df_change['Permits_lagged_1_x'] = (df_change['Permits_lagged_1_x'] - df_change['Permits_lagged_1_y']) /\
                                  df_change['Permits_lagged_1_y']
df_change['Total_Households_lagged_2_x'] = (df_change['Total_Households_lagged_2_x'] -\
                                           df_change['Total_Households_lagged_2_y']) /\
                                           df_change['Total_Households_lagged_2_y']
df_change['Med_Income_lagged_2_x'] = (df_change['Med_Income_lagged_2_x'] - df_change['Med_Income_lagged_2_y']) /\
                                      df_change['Med_Income_lagged_2_y']
df_change['PCE_lagged_1_x'] = (df_change['PCE_lagged_1_x'] - df_change['PCE_lagged_1_y']) / df_change['PCE_lagged_1_y']
df_change['Rental Vacancy Rate_lagged_2_x'] = (df_change['Rental Vacancy Rate_lagged_2_x'] -\
                                              df_change['Rental Vacancy Rate_lagged_2_y']) /\
                                              df_change['Rental Vacancy Rate_lagged_2_y']
df_change['Job Openings_lagged_1_x'] = (df_change['Job Openings_lagged_1_x'] - df_change['Job Openings_lagged_1_y']) /\
                                        df_change['Job Openings_lagged_1_y']
df_change['CommuteTime_lagged_2_x'] = (df_change['CommuteTime_lagged_2_x'] - df_change['CommuteTime_lagged_2_y']) /\
                                        df_change['CommuteTime_lagged_2_y']
df_change['Salwrkr_lagged_2_x'] = (df_change['Salwrkr_lagged_2_x'] - df_change['Salwrkr_lagged_2_y']) /\
                                        df_change['Salwrkr_lagged_2_y']
df_change['Govwrkr_lagged_2_x'] = (df_change['Govwrkr_lagged_2_x'] - df_change['Govwrkr_lagged_2_y']) /\
                                        df_change['Govwrkr_lagged_2_y']
df_change['GRAPI_lagged_2_x'] = (df_change['GRAPI_lagged_2_x'] - df_change['GRAPI_lagged_2_y']) /\
                                        df_change['GRAPI_lagged_2_y']
df_change['P_Density_lagged_2_x'] = (df_change['P_Density_lagged_2_x'] - df_change['P_Density_lagged_2_y']) /\
                                        df_change['P_Density_lagged_2_y']
df_change['Gini_Index_lagged_2_x'] = (df_change['Gini_Index_lagged_2_x'] - df_change['Gini_Index_lagged_2_y']) /\
                                        df_change['Gini_Index_lagged_2_y']

df_change = df_change[['Year_x', 'Month_x', 'ZipCode', 'ZORI_x', 'ZHVI_lagged_1_x', 'HPI_lagged_2_x', 'AQI_lagged_1_x',
                      'Population_lagged_2_x', 'Unemployment_lagged_1_x', 'Percent Bachelors_lagged_2_x',
                      'Permits_lagged_1_x', 'Total_Households_lagged_2_x', 'Med_Income_lagged_2_x', 'PCE_lagged_1_x',
                      'Rental Vacancy Rate_lagged_2_x', 'Job Openings_lagged_1_x', 'CommuteTime_lagged_2_x',
                      'Salwrkr_lagged_2_x', 'Govwrkr_lagged_2_x', 'GRAPI_lagged_2_x', 'P_Density_lagged_2_x',
                      'Gini_Index_lagged_2_x']]

df_change.columns = ['Year', 'Month', 'ZipCode', 'ZORI_delta', 'ZHVI_lagged_1_delta', 'HPI_lagged_2_delta',
                     'AQI_lagged_1_delta', 'Population_lagged_2_delta', 'Unemployment_lagged_1_delta',
                     'Percent Bachelors_lagged_2_delta', 'Permits_lagged_1_delta', 'Total_Households_lagged_2_delta',
                     'Med_Income_lagged_2_delta', 'PCE_lagged_1_delta', 'Rental Vacancy Rate_lagged_2_delta',
                     'Job Openings_lagged_1_delta', 'CommuteTime_lagged_2_delta', 'Salwrkr_lagged_2_delta',
                     'Govwrkr_lagged_2_delta', 'GRAPI_lagged_2_delta', 'P_Density_lagged_2_delta',
                     'Gini_Index_lagged_2_delta']

# ZORI not available for 2013 and Jan 2014
df_change = df_change[(df_change['Year'] != 2014) & ~((df_change['Year'] == 2015) & (df_change['Month'] == 1))]

# Dropping NaN values
df_change = df_change[~df_change['ZORI_delta'].isnull()]
df_change = df_change[~df_change['ZHVI_lagged_1_delta'].isnull()]
df_change = df_change[~df_change['HPI_lagged_2_delta'].isnull()]
df_change = df_change[~df_change['AQI_lagged_1_delta'].isnull()]
df_change = df_change[~df_change['Population_lagged_2_delta'].isnull()]
df_change = df_change[~df_change['Permits_lagged_1_delta'].isnull()]
df_change = df_change[~df_change['Rental Vacancy Rate_lagged_2_delta'].isnull()]

# Drop np.inf values
df_change = df_change[df_change['Permits_lagged_1_delta'] != np.inf]
df_change = df_change[df_change['Rental Vacancy Rate_lagged_2_delta'] != np.inf]

### Final DataFrame for Prediction (% Change in ZORI) ###
df_pred_change = zori[zori['ZORI'] <= 4000].copy() # Exclude outliers
df_pred_change = pd.merge(df_pred_change, one_year_forecast_pred, left_on = 'Year_Month', right_on = 'Year_Month2')
df_pred_change.drop('Year_Month_x', axis = 1, inplace = True)

df_pred_change = pd.merge(df_pred_change, zhvi[['Year_Month', 'ZipCode', 'ZHVI']], left_on = ['Year_Month2', 'ZipCode'],
                   right_on = ['Year_Month', 'ZipCode'], how = 'left')
df_pred_change.drop('Year_Month', axis = 1, inplace = True)

df_pred_change = pd.merge(df_pred_change, hpi[['Year_Month', 'ZipCode', 'HPI']], left_on = ['Year_Month3', 'ZipCode'],
                   right_on = ['Year_Month', 'ZipCode'], how = 'left')
df_pred_change.drop('Year_Month', axis = 1, inplace = True)

df_pred_change = pd.merge(df_pred_change, airq[['County', 'State', 'Year_Month', 'AQI']],
                          left_on = ['County', 'State', 'Year_Month2'], right_on = ['County', 'State', 'Year_Month'],
                          how = 'left')
df_pred_change.drop('Year_Month', axis = 1, inplace = True)

df_pred_change = pd.merge(df_pred_change, population[['County', 'State', 'Year_Month', 'Population']],
                   left_on = ['County', 'State', 'Year_Month3'], right_on = ['County', 'State', 'Year_Month'], how = 'left')
df_pred_change.drop('Year_Month', axis = 1, inplace = True)

df_pred_change = pd.merge(df_pred_change, unemployment[['County', 'State', 'Year_Month', 'Unemployment']],
                   left_on = ['County', 'State', 'Year_Month2'], right_on = ['County', 'State', 'Year_Month'], how = 'left')
df_pred_change.drop('Year_Month', axis = 1, inplace = True)

df_pred_change = pd.merge(df_pred_change, education[['County', 'State', 'Year_Month', 'Percent Bachelors']],
                          left_on = ['County', 'State', 'Year_Month3'], right_on = ['County', 'State', 'Year_Month'],
                          how = 'left')
df_pred_change.drop('Year_Month', axis = 1, inplace = True)

df_pred_change = pd.merge(df_pred_change, permits[['State', 'Units', 'Year_Month']], left_on = ['State', 'Year_Month2'],
                          right_on = ['State', 'Year_Month'], how = 'left')
df_pred_change.drop('Year_Month', axis = 1, inplace = True)

df_pred_change = pd.merge(df_pred_change, IandH[['County', 'State', 'Year_Month', 'Total_Households', 'Med_income']],
                          left_on = ['County', 'State', 'Year_Month3'], right_on = ['County', 'State', 'Year_Month'],
                          how = 'left')
df_pred_change.drop('Year_Month', axis = 1, inplace = True)

df_pred_change = pd.merge(df_pred_change, pce[['PCE', 'Year_Month']], left_on = 'Year_Month2', right_on = 'Year_Month')
df_pred_change.drop('Year_Month', axis = 1, inplace = True)

df_pred_change = pd.merge(df_pred_change, vacancy[['County', 'State', 'Year_Month', 'Rental Vacancy Rate']],
                          left_on = ['County', 'State', 'Year_Month3'], right_on = ['County', 'State', 'Year_Month'],
                          how = 'left')
df_pred_change.drop('Year_Month', axis = 1, inplace = True)

df_pred_change = pd.merge(df_pred_change, jobs[['State', 'Job Openings', 'Year_Month']], left_on = ['State', 'Year_Month2'],
                          right_on = ['State', 'Year_Month'], how = 'left')
df_pred_change.drop('Year_Month', axis = 1, inplace = True)

df_pred_change = pd.merge(df_pred_change, commute_worker[['County', 'State', 'CommuteTime', 'Salwrkr', 'Govwrkr',
                                                          'Year_Month']],
                          left_on = ['County', 'State', 'Year_Month3'],
                          right_on = ['County', 'State', 'Year_Month'], how = 'left')
df_pred_change.drop('Year_Month', axis = 1, inplace = True)

df_pred_change = pd.merge(df_pred_change, grapi[['County', 'State', 'GRAPI', 'Year_Month']],
                          left_on = ['County', 'State', 'Year_Month3'],
                          right_on = ['County', 'State', 'Year_Month'], how = 'left')
df_pred_change.drop('Year_Month', axis = 1, inplace = True)

df_pred_change = pd.merge(df_pred_change, population_density[['County', 'State', 'P_Density', 'Year_Month']],
                          left_on = ['County', 'State', 'Year_Month3'],
                          right_on = ['County', 'State', 'Year_Month'], how = 'left')
df_pred_change.drop('Year_Month', axis = 1, inplace = True)

df_pred_change = pd.merge(df_pred_change, income_inequality[['County', 'State', 'Gini_Index', 'Year_Month']],
                          left_on = ['County', 'State', 'Year_Month3'],
                          right_on = ['County', 'State', 'Year_Month'], how = 'left')
df_pred_change.drop('Year_Month', axis = 1, inplace = True)

df_pred_change = df_pred_change[['Year', 'Month', 'ZipCode', 'Year_Month_y', 'Year_Month2', 'Year_Month3', 'ZHVI',
                                 'HPI', 'AQI', 'Population', 'Unemployment', 'Percent Bachelors', 'Units',
                                 'Total_Households', 'Med_income', 'PCE', 'Rental Vacancy Rate', 'Job Openings',
                                'CommuteTime', 'Salwrkr', 'Govwrkr', 'GRAPI', 'P_Density', 'Gini_Index']]
df_pred_change['Year'] = df_pred_change['Year'] + 1

df_temp = df.copy()
df_temp['Year_Month'] = df_temp['Year'].map(str) + '_' + df_temp['Month'].map(str)

df_pred_change = pd.merge(df_pred_change, df_temp, left_on = ['Year_Month2', 'ZipCode'],
                          right_on = ['Year_Month', 'ZipCode'], how = 'left')

df_pred_change['ZHVI'] = (df_pred_change['ZHVI'] - df_pred_change['ZHVI_lagged_1']) / df_pred_change['ZHVI_lagged_1']
df_pred_change['HPI'] = (df_pred_change['HPI'] - df_pred_change['HPI_lagged_2']) / df_pred_change['HPI_lagged_2']
df_pred_change['AQI'] = (df_pred_change['AQI'] - df_pred_change['AQI_lagged_1']) / df_pred_change['AQI_lagged_1']
df_pred_change['Population'] = (df_pred_change['Population'] - df_pred_change['Population_lagged_2']) /\
                                df_pred_change['Population_lagged_2']
df_pred_change['Unemployment'] = (df_pred_change['Unemployment'] - df_pred_change['Unemployment_lagged_1']) /\
                                  df_pred_change['Unemployment_lagged_1']
df_pred_change['Percent Bachelors'] = (df_pred_change['Percent Bachelors'] -\
                                       df_pred_change['Percent Bachelors_lagged_2']) /\
                                       df_pred_change['Percent Bachelors_lagged_2']
df_pred_change['Units'] = (df_pred_change['Units'] - df_pred_change['Permits_lagged_1']) /\
                           df_pred_change['Permits_lagged_1']
df_pred_change['Total_Households'] = (df_pred_change['Total_Households'] -\
                                      df_pred_change['Total_Households_lagged_2']) /\
                                      df_pred_change['Total_Households_lagged_2']
df_pred_change['Med_income'] = (df_pred_change['Med_income'] - df_pred_change['Med_Income_lagged_2']) /\
                                df_pred_change['Med_Income_lagged_2']
df_pred_change['PCE'] = (df_pred_change['PCE'] - df_pred_change['PCE_lagged_1']) / df_pred_change['PCE_lagged_1']
df_pred_change['Rental Vacancy Rate'] = (df_pred_change['Rental Vacancy Rate'] -\
                                         df_pred_change['Rental Vacancy Rate_lagged_2']) /\
                                         df_pred_change['Rental Vacancy Rate_lagged_2']
df_pred_change['Job Openings'] = (df_pred_change['Job Openings'] - df_pred_change['Job Openings_lagged_1']) /\
                                  df_pred_change['Job Openings_lagged_1']
df_pred_change['CommuteTime'] = (df_pred_change['CommuteTime'] - df_pred_change['CommuteTime_lagged_2']) /\
                                 df_pred_change['CommuteTime_lagged_2']
df_pred_change['Salwrkr'] = (df_pred_change['Salwrkr'] - df_pred_change['Salwrkr_lagged_2']) /\
                             df_pred_change['Salwrkr_lagged_2']
df_pred_change['Govwrkr'] = (df_pred_change['Govwrkr'] - df_pred_change['Govwrkr_lagged_2']) /\
                             df_pred_change['Govwrkr_lagged_2']
df_pred_change['GRAPI'] = (df_pred_change['GRAPI'] - df_pred_change['GRAPI_lagged_2']) /\
                             df_pred_change['GRAPI_lagged_2']
df_pred_change['P_Density'] = (df_pred_change['P_Density'] - df_pred_change['P_Density_lagged_2']) /\
                               df_pred_change['P_Density_lagged_2']
df_pred_change['Gini_Index'] = (df_pred_change['Gini_Index'] - df_pred_change['Gini_Index_lagged_2']) /\
                                df_pred_change['Gini_Index_lagged_2']

df_pred_change = df_pred_change[['Year_x', 'Month_x', 'ZipCode', 'ZHVI', 'HPI', 'AQI', 'Population', 'Unemployment',
                                 'Percent Bachelors', 'Units', 'Total_Households', 'Med_income', 'PCE',
                                 'Rental Vacancy Rate', 'Job Openings', 'CommuteTime', 'Salwrkr', 'Govwrkr', 'GRAPI',
                                 'P_Density', 'Gini_Index']]

df_pred_change.columns = ['Year', 'Month', 'ZipCode', 'ZHVI_lagged_1_delta', 'HPI_lagged_2_delta', 'AQI_lagged_1_delta',
                          'Population_lagged_2_delta', 'Unemployment_lagged_1_delta', 'Percent Bachelors_lagged_2_delta',
                          'Permits_lagged_1_delta', 'Total_Households_lagged_2_delta', 'Med_Income_lagged_2_delta',
                          'PCE_lagged_1_delta', 'Rental Vacancy Rate_lagged_2_delta', 'Job Openings_lagged_1_delta',
                          'CommuteTime_lagged_2_delta', 'Salwrkr_lagged_2_delta', 'Govwrkr_lagged_2_delta',
                          'GRAPI_lagged_2_delta', 'P_Density_lagged_2_delta', 'Gini_Index_lagged_2_delta']

# Dropping NaN values
df_pred_change = df_pred_change[~df_pred_change['ZHVI_lagged_1_delta'].isnull()]
df_pred_change = df_pred_change[~df_pred_change['HPI_lagged_2_delta'].isnull()]
df_pred_change = df_pred_change[~df_pred_change['AQI_lagged_1_delta'].isnull()]
df_pred_change = df_pred_change[~df_pred_change['Population_lagged_2_delta'].isnull()]
df_pred_change = df_pred_change[~df_pred_change['Permits_lagged_1_delta'].isnull()]
df_pred_change = df_pred_change[~df_pred_change['Rental Vacancy Rate_lagged_2_delta'].isnull()]
df_pred_change = df_pred_change[~df_pred_change['Job Openings_lagged_1_delta'].isnull()]

# Drop np.inf values
df_pred_change = df_pred_change[df_pred_change['Permits_lagged_1_delta'] != np.inf]
df_pred_change = df_pred_change[df_pred_change['Rental Vacancy Rate_lagged_2_delta'] != np.inf]
