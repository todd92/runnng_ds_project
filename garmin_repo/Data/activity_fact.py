import os 
from dotenv import load_dotenv
from garminconnect import Garmin
import datetime
from datetime import date
import json
import pandas as pd

load_dotenv()

username = os.getenv("GARMIN_UNAME")
password = os.getenv("GARMIN_PWORD")

api = Garmin(username, password)
api.login()

all_activities = api.get_activities_by_date("2019-01-01", str(date.today()))


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

df_activities["distance_miles"] = df_activities["distance"] / 1609.34

df_activities["duration_minutes"] = df_activities["duration"] / 60

df_activities["elapsedDuration_minutes"] = df_activities["elapsedDuration"] / 60

cols = [
    "activityId",
    "startTimeLocal",
    "Activity Type",
    "distance_miles",
    "duration_minutes",
    "elapsedDuration_minutes",
    "maxHR",
    "averageRunningCadenceInStepsPerMinute",
    "maxRunningCadencInStepsPerMinute",
    "steps",
    "beginTimestamp",
    "avgStrideLength",
    "vO2MaxValueminElevation",
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
