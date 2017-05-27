import numpy as np
import pandas as pd

def detrend_timeseries(series):

    X = series.values
    diff = list()
    for i in range(1, len(X)):
        value = X[i] - X[i - 1]
        diff.append(value)

def load_csv(file, group_columns = [], categorical_columns = [], meta_columns = []):
    data = pd.read_csv(file, sep=';')

    # Initial data-slicing
    data = data[(data.LinkTravelTime > 0) & (data.LineDirectionCode == 1)]

    # Data convertion
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    time = pd.DatetimeIndex(data['DateTime']) 
    data['TimeOfDayClass'] = 'NO_PEEK' 
    data['Hour'] = time.hour
    data.ix[((7 < time.hour) & (time.hour < 9) & (data['DayType'] == 1)), 'TimeOfDayClass'] = 'PEEK' 
    data.ix[((15 < time.hour) & (time.hour < 17) & (data['DayType'] == 1)), 'TimeOfDayClass'] = 'PEEK' 
       
    numerical_columns = []
    
    output_column = 'LinkTravelTime'

    # Calculate m lag headway and travel time for same link, earlier journeys
    m = 20
    grouping = data.groupby(['LinkRef'])
    for i in range(1, m + 1):
        data['HeadwayTime_L' + str(i)] = (data['DateTime'] - grouping['DateTime'].shift(i)) / np.timedelta64(1, 's')
        data['LinkTravelTime_L' + str(i)] = grouping['LinkTravelTime'].shift(i)
        #numerical_columns += ['HeadwayTime_L' + str(i), 'LinkTravelTime_L' + str(i)]
        numerical_columns += ['LinkTravelTime_L' + str(i)]
    
    # Slice out missing values
    for i in range(1, m + 1):
        data = data[(data['HeadwayTime_L' + str(i)] > 0) & (data['LinkTravelTime_L' + str(i)] > 0)]

    """
    # Calculate j lag headway and travel time for journey, upstream links
    j = 3
    grouping = data.groupby(['JourneyRef'])
    for i in range(1, j + 1):
        data['LinkTravelTime_J' + str(i)] = grouping['LinkTravelTime'].shift(i)
        numerical_columns += ['LinkTravelTime_J' + str(i)]
    
    # Slice out missing values
    for i in range(1, j + 1):
        data = data[(data['LinkTravelTime_J' + str(i)] > 0)]
    """
    data = data[(26 <= data.LineDirectionLinkOrder) & (data.LineDirectionLinkOrder <= 26)]

    print('Preprosessed data set size:', len(data))

    input_columns = categorical_columns + numerical_columns

    if len(group_columns) > 0:
        grouping = data.groupby(group_columns)
    else:
        grouping = [('all', data)]

    for key, group in grouping:
        with_dummies = pd.get_dummies(group[input_columns], columns = categorical_columns)
        print with_dummies.head()
        # Create dummy variables
        X = with_dummies.as_matrix()

        Y = group.as_matrix(columns = [output_column])

        yield (key, X, Y, group[(meta_columns + input_columns + [output_column])])
