import requests
import json
from datetime import date, timedelta
import pandas as pd
import calendar

locations = {
    "New York": "NY/New York",
}

API_KEY = "07dd951e248608cc"
year = 2015
months = [str(i) for i in range(1, 13)]


def get_json(url):
    print "getting weather from url:", url
    r = requests.get(url)
    data = json.loads(r.text)
    return data


def get_weather_info():

    err = False
    err_msg = ""

    dates = []
    weather = []

    for month in months:

        if len(month) == 1:
            month = '0' + month
        num_days = calendar.monthrange(year, int(month))[1]

        for day in range(1, num_days + 1):

            if len(str(day)) == 1:
                day = '0' + str(day)
            else:
                day = str(day)

            url = "http://api.wunderground.com/api/" + API_KEY + "/history_2015" + \
                month + day + "/q/" + locations["New York"] + ".json"

            try:
                data = get_json(url)

                dates.append(data['history']['date']['year'] + '-' + data['history']
                             ['date']['mon'] + '-' + data['history']['date']['mday'])
                weather.append(data['history']['observations'][2]['icon'])

            except:
                print "Could not get data for: %s \nResponse: %s" % (city, str(data))
                err = True
                err_msg += "Could not get data for: %s \nResponse: %s\n" % (
                    city, str(data))
                continue

    df = pd.DataFrame()

    df['date'] = dates
    df['weather'] = weather

    return df.to_csv('data/daily_weather_NYC_2015_.csv', index=False)


if __name__ == "__main__":
    get_weather_info()
