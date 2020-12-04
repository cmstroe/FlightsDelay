#%%

import pandas as pd

flights_train = pd.read_csv(r'./data/flights_train.csv')
airports = pd.read_csv(r'./data/airports.csv')

###################################################

data = pd.merge(flights_train, airports.drop('AIRPORT', axis=1).add_prefix('ORIGIN_'), left_on='ORIGIN_AIRPORT', right_on='ORIGIN_IATA_CODE')

data = pd.merge(data, airports.drop('AIRPORT', axis=1).add_prefix('DESTINATION_'), left_on='DESTINATION_AIRPORT', right_on='DESTINATION_IATA_CODE')

###################################################

data['Datetime'] = pd.to_datetime(data['YEAR'].astype(str) + '-' + data['MONTH'].astype(str) + '-' + data['DAY'].astype(str), format='%Y-%m-%d')

# %%
