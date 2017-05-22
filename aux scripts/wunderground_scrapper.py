import sys
import requests
import json
import pickle
import time
import random
from pprint import pprint
from datetime import date, timedelta

locations = {
    "New York": "NY/New York",
}

#API_KEY = "fb647762af5c3954"
API_KEY = "74c48d0021b92246"


def get_json(url):
    print "getting weather from url:", url
    r = requests.get(url)
    data = json.loads(r.text)
    return data


def read_db(filename):
    print "reading db from file:", filename
    try:
        f = open(filename)
        #db = json.load(f)
        db = pickle.load(f)
        f.close()
    except IOError:
        db = {}
    return db


def save_db(db, filename):
    print "writing db to file:", filename
    f = open(filename, "w")
    #json.dump(db, f)
    pickle.dump(db, f)
    f.close()


data_path = '../data/'
if not data_path.endswith('/'):
    data_path += "/"

# get weather info
err = False
err_msg = ""
for city in locations:
    print "\ngetting weather for city:", city

    # read database
    db_filename = data_path + "db_" + city + ".pickle"
    db = read_db(db_filename)
    # print "db size:", len(db)

    url = "http://api.wunderground.com/api/" + API_KEY + \
        "/conditions/q/" + locations[city] + ".json"
    try:
        data = get_json(url)
        observation = data["current_observation"]
        # pprint(observation)
    except:
        print "Could not get data for: %s \nResponse: %s" % (city, str(data))
        err = True
        err_msg += "Could not get data for: %s \nResponse: %s\n" % (
            city, str(data))
        continue

    observation_id = observation["observation_epoch"]
    # print observation_id

    if observation_id not in db:
        print "got new weather conditions: %s (%s)" % (observation["weather"], observation["observation_time_rfc822"])
    else:
        print "updating weather conditions: %s (%s)" % (observation["weather"], observation["observation_time_rfc822"])
    db[observation_id] = observation

    print "new db size:", len(db)

    time.sleep(60 + 20 * random.random())

    # save database
    save_db(db, db_filename)

if err:
    raise Exception(err_msg)
