def add_features(flights_train):

    """[Adding flights_train objects/features to the flights_train such as airport locations of the origin and destination and datetime]

    Returns:
        [DataFrame]: flights_train populated with the additional features
    """    

    ###################################################

    flights_train = pd.merge(flights_train, airports.drop('AIRPORT', axis=1).add_prefix('ORIGIN_'), left_on='ORIGIN_AIRPORT', right_on='ORIGIN_IATA_CODE')

    flights_train = pd.merge(flights_train, airports.drop('AIRPORT', axis=1).add_prefix('DESTINATION_'), left_on='DESTINATION_AIRPORT', right_on='DESTINATION_IATA_CODE')

    ###################################################

    flights_train['Datetime'] = pd.to_datetime(flights_train['YEAR'].astype(str) + '-' + flights_train['MONTH'].astype(str) + '-' + flights_train['DAY'].astype(str), format='%Y-%m-%d')

    return flights_train
# %%
