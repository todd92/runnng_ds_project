import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
from datetime import datetime


garmin_act = pd.read_csv("C:/Users/toddglad/Downloads/garmin_activities.csv")

garmin_act.sort_values("Distance")

garmin_act[["Date", "Activity Type", "Favorite", "Calories", "Time"]]


garmin_act[garmin_act["Time"].isna()]

garmin_act.dtypes

print(garmin_act.info())


garmin_act["Calories"] = pd.to_numeric(garmin_act["Calories"], errors="coerce")


garmin_act.groupby("Activity Type")["Calories"].agg(["mean", "size"])


garmin_act["Avg HR"] = pd.to_numeric(garmin_act["Avg HR"], errors="coerce")


def get_unique_values(dataset, column_name):
    print(f"Unique values in column '{column_name}':")
    return dataset[column_name].unique()


get_unique_values(garmin_act, "Activity Type")

garmin_act.columns

for i in range(0, len(garmin_act.columns)):
    print(f"{i}: {garmin_act.columns[i]}")
    print(get_unique_values(garmin_act, garmin_act.columns[i]))


def convert_to_numeric(df, column_name):
    df[column_name + "_num"] = pd.to_numeric(df[column_name], errors="coerce")
    return df


for i in range(0, len(garmin_act.columns)):
    col_name = garmin_act.columns[i]
    if not is_numeric_dtype(garmin_act[col_name]):
        print(f"Converting column '{col_name}' to numeric.")
        garmin_act = convert_to_numeric(garmin_act, col_name)
        if garmin_act[col_name + "_num"].isna().all():
            print(
                f"Warning: Column '{col_name}' conversion resulted in all NaN values."
            )
            garmin_act = garmin_act.drop(col_name + "_num", axis="columns")
    else:
        print(f"Column '{col_name}' is already numeric.")

# pd.api.types.is_numeric_dtype(df['column_name']):


def create_agg_dataset(df, group_by_col, agg_col, agg_func):
    agg_df = df.groupby(group_by_col)[agg_col].agg(agg_func).reset_index()
    return agg_df


cadence_agg = garmin_act.groupby("Activity Type")["Avg Cadence_num"].mean()

cadence_agg = cadence_agg.reset_index().sort_values("Avg Cadence_num", ascending=False)

cadence_agg

plt.figure(figsize=(15, 6))
plt.bar(cadence_agg["Activity Type"], cadence_agg["Avg Cadence_num"])


fig, ax = plt.subplots(figsize=(15, 6))
ax.bar(cadence_agg["Activity Type"], cadence_agg["Avg Cadence_num"])
ax.set_title("Average Cadence by Activity Type", fontweight="bold")
ax.set_xlabel("Activity Type", fontsize=14)
ax.set_ylabel("Average Cadence", fontsize=14)
ax.grid(axis="y")
fig.patch.set_facecolor("lightgray")
ax.set_facecolor("whitesmoke")
plt.show()


fig, ax = plt.subplots(figsize=(15, 6))
ax.bar(cadence_agg["Activity Type"], cadence_agg["Avg Cadence_num"])
ax.set_title("Average Cadence by Activity Type", fontweight="bold")
ax.set_xlabel("Activity Type", fontsize=14)
ax.set_ylabel("Average Cadence", fontsize=14)
ax.grid(axis="y")
fig.patch.set_facecolor("lightgray")
ax.set_facecolor("whitesmoke")
plt.show()

garmin_act["Date_time"] = pd.to_datetime(garmin_act["Date"])
garmin_act["Month"] = garmin_act["Date_time"].dt.to_period("M").dt.to_timestamp()
garmin_act["Date"] = garmin_act["Date_time"].dt.date

garmin_act

date_cadence_agg = garmin_act.groupby("Month")["Avg Cadence_num"].mean()
date_cadence_agg = date_cadence_agg.reset_index().sort_values("Month", ascending=True)

date_cadence_agg


date_cad, ax1 = plt.subplots(figsize=(15, 6))
ax1.bar(date_cadence_agg["Month"], date_cadence_agg["Avg Cadence_num"])
ax1.set_title("Average Cadence by Date", fontweight="bold")
ax1.set_xlabel("Month", fontsize=14)
ax1.set_ylabel("Average Cadence", fontsize=14)
ax1.grid(axis="y")
date_cad.patch.set_facecolor("lightgray")
ax1.set_facecolor("whitesmoke")
plt.show()


#### Best ways to know that I'm ready for a marathon, or half marathon.
## Days running per week
## Mileage per week
## Number of runs per week
## longest run per week


garmin_act.columns

## treadmill or regular runs

garmin_act["Activity Type"].unique()

running = garmin_act[
    (garmin_act["Activity Type"] == "Running")
    | (garmin_act["Activity Type"] == "Treadmill Running")
]

distance_date = create_agg_dataset(running, "Date", "Distance_num", "sum")

distance_d, ax3 = plt.subplots(figsize=(15, 6))
ax3.plot(distance_date["Date"], distance_date["Distance_num"])
ax3.set_title("Distance by Date", fontweight="bold")
ax3.set_xlabel("Date", fontsize=14)
ax3.set_ylabel("Distance", fontsize=14)
ax3.grid(axis="y")
distance_d.patch.set_facecolor("lightgray")
ax1.set_facecolor("whitesmoke")
plt.show()


from garminconnect import Garmin
import datetime
import json

help(Garmin)
help(json)
# Replace with your credentials
username = "twadeal@gmail.com"
password = "1Nephi3:7"

try:
    api = Garmin(username, password)
    api.login()
    api.get_user_profile()
    api.get_last_activity()
    api.get_heart_rates("2025-10-13")
    api.get_fitnessage_data("2025-10-13")
    api.get_daily_steps("2025-10-10", "2025-10-13")
    activity_start_date = datetime.date(2023, 1, 1)
    activity_end_date = datetime.date(2023, 1, 7)
    activities = api.get_workouts(1, 100)
    all_activities = api.get_activities_by_date("2019-01-01", "2025-10-14")
    print(activities)
except Exception as e:
    print(f"An error occurred: {e}")

all_activities[0]["pr"]

first_activity = all_activities[0]

print(first_activity.keys())

print(json.dumps(first_activity, indent=4))

type(all_activities[0])

import pandas as pd

# Convert the whole list of activity dictionaries into a DataFrame
df_activities = pd.DataFrame(all_activities)

# Display the first 5 rows of your new table
df_activities

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

df_activities["Activity Type"].unique()

api.get_activity_splits(all_activities[0]["activityId"])

df_activities["pr"]

df_activities["pr"].value_counts()


api.get_activity_weather(all_activities[0]["activityId"])


from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
from datetime import datetime


running_activities = df_activities[df_activities["Activity Type"] == "running"]
running_activities["startTimeLocal"] = pd.to_datetime(
    running_activities["startTimeLocal"]
).dt.date

running_activities.describe()

plt.figure(figsize=(15, 6))
plt.scatter(
    running_activities["startTimeLocal"],
    running_activities["distance_miles"],
)
plt.title("Running Distance Over Time")
plt.xlabel("Date")
plt.ylabel("Distance (miles)")
plt.grid()
plt.show()

api.get_training_readiness("2025-10-10")

api.get_user_profile()
