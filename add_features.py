import pandas as pd
import numpy as np
from tqdm import tqdm



def add_flight_features(flights_train, airports):

    """[Adding flights_train objects/features to the flights_train such as airport locations of the origin and destination and datetime]

    Returns:
        [DataFrame]: flights_train populated with the additional features
    """    

    # Merge the 'ORIGIN_AIRPORT' with meta_data of 'airports.csv'
    flights_train = pd.merge(flights_train, 
                            airports.drop('AIRPORT', axis=1).add_prefix('ORIGIN_'), 
                            left_on='ORIGIN_AIRPORT', right_on='ORIGIN_IATA_CODE')

    # Merge the 'DESTINATION_AIRPORT' with meta_data of 'airports.csv'
    flights_train = pd.merge(flights_train, 
                            airports.drop('AIRPORT', axis=1).add_prefix('DESTINATION_'), 
                            left_on='DESTINATION_AIRPORT', right_on='DESTINATION_IATA_CODE')

    ############################################################################

    # Convert the year, month and day column to a datetime object
    flights_train['DATETIME'] = pd.to_datetime(flights_train['YEAR'].astype(str) + '-' + flights_train['MONTH'].astype(str) + '-' + flights_train['DAY'].astype(str), format='%Y-%m-%d')

    # Pad the time with zeros in front to a maximum of 4 chars so it becomes a time in range (0000 - 2359)
    flights_train.loc[:, 'SCHEDULED_DEPARTURE_DATETIME'] = flights_train['SCHEDULED_DEPARTURE'].astype(str).str.pad(width=4, fillchar='0', side='left')

    # Convert the scheduled departure time to a datetime (year-month-date hours:minutes) 
    # object and round to the nearest hour for future data wrangling purposes
    flights_train.loc[:, 'SCHEDULED_DEPARTURE_DATETIME'] = pd.to_datetime(flights_train['YEAR'].astype(str) + 
                                                                    '-' + flights_train['MONTH'].astype(str) + 
                                                                    '-' + flights_train['DAY'].astype(str) + 
                                                                    ' ' + flights_train['SCHEDULED_DEPARTURE_DATETIME'].astype(str), 
                                                                    format='%Y-%m-%d %H%M').dt.round('1H'
                                                                    )

    return flights_train

################################################################################

def filter_and_concatenate_METAR_data(flights_train, airports):

    # Listing all unique airports in the flights_train dataset (origin/destination)
    unique_airports = np.unique(np.concatenate((flights_train['ORIGIN_AIRPORT'].unique(), flights_train['DESTINATION_AIRPORT'].unique())))

    ############################################################################

    dfs_METAR_airports = []

    # Loop over METAR data files of all airports in the world
    for txt_file in os.listdir('./METAR_data'):
        
        # Only iterate over .txt files
        if '.txt' in txt_file:

            print(txt_file)

            # Load the .txt file METAR data of a single day into a DataFrame, skip the first 5 rows and set 6th row as header
            METAR_day = pd.read_csv(f'./METAR_data/{txt_file}', sep=",", header=0, skiprows=5)

            # Filter DataFrame by only considering all origin and destination (US) airports in flights_train
            df_METAR_airports = METAR_day[METAR_day['station'].isin(unique_airports)]

            # Append the daily airport METAR data to a list for eventual concatenation into a single DataFrame (spanning all days)
            dfs_METAR_airports.append(df_METAR_airports)

    # Concatenate all METAR airport data for all days within the flights_train time period and reset its index
    df_METAR_airports = pd.concat(dfs_METAR_airports).reset_index()

    ############################################################################

    # For some reason Pandas cannot convert the 'valid' (timestamp of the METAR observation) 
    # column in one go but does when looping over the data per airport and then 
    # transforming the 'valid' column to a datetime object and rounding it 
    # to the nearest hour in the 'valid_hour' column

    valid_dfs = []

    # Loop over all US airports in the METAR data (location of METAR station)
    for airport in tqdm(df_METAR_airports['station'].unique()):

        data = df_METAR_airports[df_METAR_airports['station'] == airport]

        # Convert the timestamp of observation column to a datetime object
        data.loc[:, 'valid'] = pd.to_datetime(data['valid'])
        
        # Round the timestamp of observation column to the nearest hour and 
        # store as new column
        data.loc[:, 'valid_hour'] = data['valid'].dt.round(freq='1H')

        # Append the unique METAR airport data for the time-period to a list 
        # for eventual concatenation
        valid_dfs.append(data)

    # Concatenate all METAR airport data 
    df_METAR_airports = pd.concat(valid_dfs)

    ############################################################################

    # Drop duplicates by checking the rounded timestamps of observations of an
    # airport station and only keep the last observation if there are multiple
    df_METAR_airports = df_METAR_airports.drop_duplicates(subset=['station', 'valid_hour'], keep='last')

    # Write away the METAR data into .hdf file format for fast data loading
    df_METAR_airports.to_hdf('./METAR_data/df_METAR_airports.hdf', key='data', mode='w')

    return df_METAR_airports

################################################################################

def add_METAR_data_features(flights_train, df_METAR_airports):

    # relevant_METAR_cols = [
    #                     'station', # three or four character site identifier
    #                     'valid', # timestamp of the observation
    #                     'drct', # Wind Direction in degrees from north
    #                     'sknt', # Wind Speed in knots 
    #                     'vsby', # Visibility in miles
    #                     'gust', # Wind Gust in knots
    #                     'wxcodes', # Present Weather Codes (space seperated)
    #                     'ice_accretion_1hr', # Ice Accretion over 1 Hour (inches)
    #                     'ice_accretion_3hr', # Ice Accretion over 3 Hours (inches)
    #                     'ice_accretion_6hr', # Ice Accretion over 6 Hours (inches)
    #                     'peak_wind_gust', # Peak Wind Gust (from PK WND METAR remark) (knots)
    #                     'peak_wind_drct', # Peak Wind Gust Direction (from PK WND METAR remark) (deg)
    #                     'peak_wind_time:', # Peak Wind Gust Time (from PK WND METAR remark)
    #                     'metar', # unprocessed reported observation in METAR format
    # ]

    # Adding origin METAR data
    flights_train = flights_train.merge(df_METAR_airports.add_prefix('ORIGIN_'), 
                                        left_on=['ORIGIN_AIRPORT', 'SCHEDULED_DEPARTURE_DATETIME'], 
                                        right_on=['ORIGIN_station', 'ORIGIN_valid_hour'], 
                                        how='left')

    # Adding destination METAR data
    flights_train = flights_train.merge(df_METAR_airports.add_prefix('DESTINATION_'), 
                                        left_on=['DESTINATION_AIRPORT', 'SCHEDULED_DEPARTURE_DATETIME'], 
                                        right_on=['DESTINATION_station', 'DESTINATION_valid_hour'], 
                                        how='left')

    # Write away the flights_train data with additional features
    # into .hdf file format for fast data loading
    flights_train.to_hdf('./data/flights_train.hdf', key='data', mode='w')

    return flights_train

#%%

# Load the raw data
flights_train = pd.read_csv(r'./data/flights_train.csv')
airports = pd.read_csv(r'./data/airports.csv')

# Populate flights_train with data features
flights_train = add_flight_features(flights_train, airports)

# Produce METAR database for airports in flights_train
df_METAR_airports = filter_and_concatenate_METAR_data(flights_train, airports)

# Merge the METAR database with the origin and destination airport in flights_train
# to obtain METAR data for the origin and destination airport at the scheduled departure time
df_METAR_airports = pd.read_hdf('./METAR_data/df_METAR_airports.hdf', key='data')
flights_train = add_METAR_data_features(flights_train, df_METAR_airports)

# Syntax to read the final feature-engineered flights_train data (populated with features)
flights_train = pd.read_hdf('./data/flights_train.hdf', key='data')

################################################################################

# Checking whether all processing was performed correctly

# Checking whether the length of indices of unprocessed raw flights_train data 
# is the same as the processed flights_train data
print(f'Length of raw flights_train and processed flights_train equal: {pd.read_csv("./data/flights_train.csv").shape[0] == flights_train.shape[0]}')

# Checking the count of rows with no METAR information
print(flights_train[['SCHEDULED_DEPARTURE_DATETIME', 
                    'ORIGIN_AIRPORT', 
                    'ORIGIN_station',  
                    'ORIGIN_valid_hour', 
                    'DESTINATION_AIRPORT', 
                    'DESTINATION_station', 
                    'DESTINATION_valid_hour']].isna().sum())

# Checking which airports have no METAR data
def airports_no_METAR_data(flights_train, airports):

    # Listing all unique airports in the flights_train dataset first (origin/destination)
    unique_airports = np.unique(np.concatenate((flights_train['ORIGIN_AIRPORT'].unique(), flights_train['DESTINATION_AIRPORT'].unique())))

    airports_no_METAR_data = [airport for airport in unique_airports if airport not in df_METAR_airports['station'].unique()]
    
    airports_no_METAR_data = airports[airports['IATA_CODE'].isin(airports_no_METAR)][['IATA_CODE', 'AIRPORT', 'CITY', 'STATE', 'COUNTRY']]
    
    print(f'Airports with no METAR data: {airports_no_METAR_data}')

    return airports_no_METAR_data

airports_no_METAR_data = airports_no_METAR_data(flights_train, airports)

airports_no_METAR_data