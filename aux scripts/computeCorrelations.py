import pandas as pd
import os
import shutil
import glob
import datetime
import time
import numpy as np
import itertools
import math


def splitDataByZones(inpath, outpath):
    """
    Split NYC taxi dataset in files by zones. Data for years 2009-2016.
    2 folders:
            - origin set, multiple destionations 
            - destination set, multiple origins 
    """
    num_rows_to_skip = 2
    data = {_key: [] for _key in [_ for _ in xrange(1, NUM_ZONES + 1)]}

    with open(inpath) as f:

        for _ in xrange(num_rows_to_skip):
            next(f)

        for line in f:
            line = [x.strip() for x in line.split('|')]
            try:
                data[int(line[0])].append(line)
            except:
                pass

    if os.path.exists(outpath):
        shutil.rmtree(outpath)
    os.makedirs(outpath)

    headers = ['pickup_zone', 'dropoff_zone',
               'date', 'hour', 'minute', 'pickups']

    for key, value in data.iteritems():
        pd.DataFrame(value, columns=headers).to_csv(
            outpath + 'zone_' + str(key) + '.csv', index=False, sep='\t')


def addMin(date, val):
    """
    Add minutes to time
    """
    if val == 1:
        return date + datetime.timedelta(minutes=30)
    else:
        return date


def addHour(date, val):
    """
    Add hour to time
    """
    if val != 0:
        return date + datetime.timedelta(hours=val)
    else:
        return date


def customStrptime(string):
    """
    Convert string to timestamp. Runs faster than strptime.
    """
    return datetime.datetime(
        int(string[0:4]),  # year
        int(string[5:7]),  # month
        int(string[8:10])  # day
    )


def fillPickUps(inpath):
    """
    Generate full time series. Zones in which there are no pickups
    at a given timestamp are missing in the data.
    """
    start_date = datetime.datetime.strptime(
        "2009-01-01 00:00", "%Y-%m-%d %H:%M")
    end_date = datetime.datetime.strptime("2016-06-30 00:00", "%Y-%m-%d %H:%M")
    index = pd.date_range(start_date, end_date, freq='30min').values

    zones = [_ for _ in xrange(1, NUM_ZONES + 1)]

    for file in glob.glob(inpath + '*'):

        fixed_zone = int(file.split('_')[1].split('.')[0])
        data = pd.read_csv(file, sep='\t')

        mult_df = []
        for zone in zones:

            df = data[(data.pickup_zone == fixed_zone)
                      & (data.dropoff_zone == zone)]
            df = df.set_index(np.arange(0, len(df), 1))

            datetimes = []
            for row in df.itertuples():

                inTime = customStrptime(row.date)
                updTime = addMin(addHour(inTime, row.hour), row.minute)
                datetimes.append(updTime)

            df.insert(5, "DateTime", datetimes)
            df = df.drop(['date', 'hour', 'minute'], axis=1)
            df = df.set_index(pd.DatetimeIndex(df['DateTime']))
            df = df.drop('DateTime', axis=1)
            df = df.reindex(index)

            if math.isnan(df.loc[df.index[0]].pickup_zone):
                df.loc[df.index[0]].pickup_zone = fixed_zone
                df.loc[df.index[0]].dropoff_zone = zone

            for col in ["pickup_zone", "dropoff_zone"]:
                df[col].ffill(inplace=True)

            df.reset_index(level=0, inplace=True)
            df = df.fillna(value=0)
            df.columns = ['DateTime', 'pickup_zone', 'dropoff_zone', 'pickups']
            df = df[['pickup_zone', 'dropoff_zone', 'DateTime', 'pickups']]
            df[['pickup_zone', 'dropoff_zone', 'pickups']] = df[[
                'pickup_zone', 'dropoff_zone', 'pickups']].applymap(np.int64)

            mult_df.append(df)

        pd.concat(mult_df).to_csv(file, index=False, sep='\t')


def build_OD_Matrix(inpath):
    """
    Build Origin-Destination Matrix (Origin fixed)
    - rows: Origins
    - cols : Destinations
    - depth: timeseries
    """
    OD_mat = np.zeros((NUM_ZONES, NUM_ZONES, LEN_TIMESERIES))
    dests = [_ for _ in xrange(1, NUM_ZONES + 1)]

    for file in glob.glob(inpath + '*'):
        ori = int(file.split('_')[1].split('.')[0])
        df = pd.read_csv(file, sep='\t')

        for dest in dests:
            dt = df[(df.pickup_zone == ori) & (df.dropoff_zone == dest)]
            OD_mat[ori - 1, dest - 1, :] = dt.pickups.values

    np.save("../data/OD_matrix.csv", OD_mat)

    return OD_mat


def compute_Correlations(outpath, mat):
    """
    Compute correlations between ODs (origin fixed)
    """
    num_rows = mat.shape[0]
    num_cols = mat.shape[1]

    corrs = []
    for row in xrange(1, num_rows + 1):
        corr = []
        for col_f in xrange(1, num_cols + 1):
            new_dict = {}
            for col_var in xrange(1, num_cols + 1):
                if col_f != col_var:
                    new_dict[col_var] = np.corrcoef(
                        mat[row - 1, col_f - 1, :], mat[row - 1, col_var - 1, :])[0, 1]
            corr.append(new_dict)
        corrs.append(corr)

    df = pd.DataFrame()
    for row in xrange(num_rows):
        for col in xrange(num_cols):
            df.set_value(row, col, [corrs[row][col]])

    df.to_csv(outpath + 'CorrMat.csv', index=False)


def main():

    splitDataByZones(inpath='../data/zone_od_30min.csv',
                     outpath='../data/ODs/')
    fillPickUps(inpath='../data/ODs/')
    OD_mat = build_OD_Matrix(inpath='../data/ODs/')
    compute_Correlations(outpath='../data/ODs/', mat=OD_mat)


#global variableS
NUM_ZONES = 29
LEN_TIMESERIES = 131377

if __name__ == "__main__":
    main()
