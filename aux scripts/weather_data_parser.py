import pandas as pd
from sklearn import preprocessing


def weather_parser():
    """ This function resamples the weather data
    to a 30min frequency and encodes the different
    features
    """
    df = pd.read_csv('data/daily_weather_NYC_2015.csv', sep=",")

    list_of_dataframes = [pd.Series(row['weather'], index=pd.date_range(
        row['date'], periods=48, freq='30min')) for index, row in df.iterrows()]

    df = pd.concat(list_of_dataframes)

    # merge some weather features & encode them
    for index, item in df.iteritems():

        if item in ["mostlycloudy", "partlycloudy", "hazy", "unknown", "cloudy"]:
            df.loc[index] = "clear"

        if item == "snow":
            df.loc[index] = "rain"

    le = preprocessing.LabelEncoder()
    le.fit(list(set(df.values)))

    encoded = pd.DataFrame()
    encoded['date'] = df.index
    encoded['weather'] = le.transform(df.values)

    encoded.to_csv('data/30min_freq_weather_NYC_2015.csv', index=False)


def main():
    weather_parser()


if __name__ == "__main__":
    main()
