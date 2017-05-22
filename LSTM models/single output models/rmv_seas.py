import numpy as np
import pandas as pd
import datetime

def remove_trend(origin, destination):

    data = np.load('../../data/OD_matrix.csv.npy')

    start_date = datetime.datetime.strptime(
        "2009-01-01 00:00", "%Y-%m-%d %H:%M")
    end_date = datetime.datetime.strptime("2016-06-30 00:00", "%Y-%m-%d %H:%M")
    date_index = pd.date_range(start_date, end_date, freq='30min').values

    ts = data[origin, destination, :]

    df = pd.DataFrame()
    df['counts'] = ts
    df = df.set_index(pd.DatetimeIndex(date_index))

    from_date = datetime.datetime.strptime(
        "2015-01-01 00:00", "%Y-%m-%d %H:%M")
    to_date = datetime.datetime.strptime("2015-06-30 23:30", "%Y-%m-%d %H:%M")

    df = df.loc[from_date: to_date]

    std = np.std(df['counts'].values)

    df2 = pd.DataFrame()
    df2['Counts'] = df['counts'].values
    df2['time_of_day'] = [(date.month, date.time()) for date in df.index]

    # group by time of day
    avg = pd.DataFrame(df2.groupby(df2.time_of_day).Counts.mean()).to_dict()

    stationary_ts = []
    rmv_std = []
    rmv_seasonality = []

    for index, item in df2['time_of_day'].iteritems():

        stationary_ts.append(
            (df2.loc[index]['Counts'] - avg['Counts'][item]) / std)
        rmv_std.append(std)
        rmv_seasonality.append(avg['Counts'][item])

    df_rmv_seas = pd.DataFrame()
    df_rmv_seas['time series'] = stationary_ts
    df_rmv_seas['removed seasonality'] = rmv_seasonality
    df_rmv_seas['removed std'] = rmv_std

    return df_rmv_seas