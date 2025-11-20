import os
from dotenv import load_dotenv
from garminconnect import Garmin
import datetime
from datetime import date
import json
import pandas as pd

load_dotenv()
# help(Garmin)

username = os.getenv("GARMIN_UNAME")
password = os.getenv("GARMIN_PWORD")

api = Garmin(username, password)
api.login()


all_activities = api.get_activities_by_date("2019-01-01", str(date.today()))
all_activities

# Convert the whole list of activity dictionaries into a DataFrame
df_activities = pd.DataFrame(all_activities)

df_activities.columns

df_activity_type = pd.json_normalize(df_activities["activityType"])
df_event_type = pd.json_normalize(df_activities["eventType"])

df_activities = pd.concat([df_activities, df_activity_type], axis=1)
df_activities = df_activities.drop(columns=["activityType", "eventType"])
df_activities.rename(columns={"typeKey": "Activity Type"}, inplace=True)

df_activities = df_activities.drop(
    columns=["parentTypeId", "isHidden", "restricted", "trimmable"]
)

df_activities = df_activities[df_activities["Activity Type"].str.contains("running")]
df_activities_missing_hr = df_activities[(df_activities["maxHR"].isnull()) | (df_activities["hrTimeInZone_1"].isnull()) | (df_activities["hrTimeInZone_2"].isnull()) | (df_activities["hrTimeInZone_3"].isnull()) | (df_activities["hrTimeInZone_4"].isnull()) | (df_activities["hrTimeInZone_5"].isnull())]

df_activities_missing_hr

for i in range(len(df_activities_missing_hr)):
    activity_id = df_activities_missing_hr.iloc[i]["activityId"]
    hr_zones = api.get_activity_hr_in_timezones(str(activity_id))
    print(hr_zones)
    for zone in hr_zones:
        zone_key = f"hrTimeInZone_{zone['zoneNumber']}"
        df_activities.loc[
            df_activities["activityId"] == activity_id, zone_key
        ] = zone["secsInZone"]



df_activities["distance_miles"] = df_activities["distance"] / 1609.34

df_activities["duration_minutes"] = df_activities["duration"] / 60

df_activities["elapsedDuration_minutes"] = df_activities["elapsedDuration"] / 60
pd.set_option("display.max_columns", None)
print(df_activities[df_activities["activityId"] == 6173191603])
cols = [
    "activityId",
    "startTimeLocal",
    "Activity Type",
    "distance_miles",
    "duration_minutes",
    "elapsedDuration_minutes",
    "maxHR",
    "averageRunningCadenceInStepsPerMinute",
    "maxRunningCadenceInStepsPerMinute",
    "steps",
    "beginTimestamp",
    "avgStrideLength",
    "maxElevation",
    "maxDoubleCadence",
    "lapCount",
    "endLatitude",
    "endLongitude",
    "minActivityLapDuration",
    "fastestSplit_1000",
    "fastestSplit_1609",
    "hrTimeInZone_1",
    "hrTimeInZone_2",
    "hrTimeInZone_3",
    "hrTimeInZone_4",
    "hrTimeInZone_5",
    "endTimeGMT",
    "pr",
    "fastestSplit_5000",
    "Activity Type",
]


columns_list = cols

columns_list

df_activity_fact = df_activities[columns_list]

print(df_activity_fact.info())
print(df_activity_fact.describe())
print(df_activity_fact["hrTimeInZone_1"].describe())

print("Missing Values Counts")
print(df_activity_fact.isnull().sum())


# print("Values with Null hrTimeInZone_1")
# print(df_activity_fact[df_activity_fact["hrTimeInZone_1"].isnull()])
# df_activity_fact[df_activity_fact["hrTimeInZone_1"].isnull()]

# df_activity_fact[df_activity_fact["hrTimeInZone_1"].isnull()]
df_activity_fact.to_csv("activity_fact.csv", index=False)

## I will need to get the heart rate in the specific zones
# api.get_heart_rates("2021-01-26")
# api.get_activity_hr_in_timezones("6173191603")

## Take SecsinZone and mod by 60 that will give you the amount of seconds.
## Take SecsinZone divide by 60 that will give you the amount of minutes.

# import logging

# Set the basic configuration for logging
# level=logging.INFO means it will show INFO messages and above (WARNING, ERROR, CRITICAL)
# logging.basicConfig(level=logging.DEBUG)

# Now, instead of print("Hello world"), you use:
# logging.info("Hello world")

# This will not show up because its level (DEBUG) is below our set level (INFO)
# logging.debug("This is a debug message.")

# This will show up
# logging.warning("Something might be wrong.")
