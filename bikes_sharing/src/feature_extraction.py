import time
import datetime as dt
from sklearn import preprocessing

import math
import numpy as np
def extract_features(data, target_column='count'):
    datetimes = [time.strptime(datetime, '%Y-%m-%d %H:%M:%S')  for datetime in data['datetime']]
    data['date'] = [dt.datetime(*datetime[:6]).date() for datetime in datetimes]
    data['hour'] = [datetime.tm_hour for datetime in datetimes]

    work_hours = []
    daytime = []
    month = []
    day_of_week = []
    year = []
    year_day = []
    sunday = []
    frequent_days = []
    unfrequent_days = []
    daypart = []
    night_time = []
    small_humidity = []
    large_humidity = []
    
    cos_temp = []
    sin_temp = []
    cos_atemp = []

    month_day = []
    start_of_the_year = []
    for datetime in datetimes:
        daypart.append(int(datetime.tm_hour / 6))

        if 0 <= datetime.tm_hour <= 7:
            night_time.append(1)
        else:
            night_time.append(0)
            
        if 10 <= datetime.tm_hour <= 17:
            work_hours.append(1)
        else:
            work_hours.append(0)

        if 8 <= datetime.tm_hour <= 21:
            daytime.append(1)
        else:
            daytime.append(0)

        if datetime.tm_wday == 6:
            sunday.append(1)
        else:
            sunday.append(0)

        if datetime.tm_wday == 5 or datetime.tm_wday == 4 or datetime.tm_wday == 3:
            frequent_days.append(1)
            unfrequent_days.append(0)
        else:
            frequent_days.append(0)
            unfrequent_days.append(1)
        
        if 0 <= datetime.tm_mon <= 1:
            start_of_the_year.append(1)
        else:
            start_of_the_year.append(0)

        month.append(datetime.tm_mon)
        day_of_week.append(datetime.tm_wday)
        year.append(datetime.tm_year)
        year_day.append(datetime.tm_yday)

        month_day.append(datetime.tm_mday)
        
    for humidity in data['humidity']:
        if humidity <= 16:
            small_humidity.append(1)
            large_humidity.append(0)
        elif humidity >= 84:
            small_humidity.append(0)
            large_humidity.append(1)
        else:
            small_humidity.append(0)
            large_humidity.append(0)
          
    atemp_less_than_9 = []
    atemp_interval = []
    for atemp in data['atemp']:
        if atemp <= 9:
            atemp_less_than_9.append(1)
            atemp_interval.append(0)
        elif 36 <= atemp <= 38:
            atemp_less_than_9.append(0)
            atemp_interval.append(1)
        else:
            atemp_less_than_9.append(0)
            atemp_interval.append(0)

    
    # if target_column == 'count':
    data['work_hours'] = work_hours
    data['month'] = month
    data['day_of_week'] = day_of_week
    data['year'] = year
    data['daytime'] = daytime
    data['nighttime'] = night_time
    
    if target_column == 'casual':
        # data['small_humidity'] = small_humidity
        data['large_humidity'] = large_humidity
        data['start_of_the_year'] = start_of_the_year

        # data['atemp_less_than_9'] = atemp_less_than_9
        data['atemp_interval'] = atemp_interval
        
    
    # elif target_column == 'casual':
    #     data['work_hours'] = work_hours
    #     data['month'] = month
    #     data['day_of_week'] = day_of_week
    #     data['year'] = year
    #     data['daytime'] = daytime
    #     data[]
    # data['sunday'] = sunday
    # data['frequent_days'] = frequent_days
    # data['unfrequent_days'] = unfrequent_days
    # data['daypart'] = daypart
    # data['cos_temp'] = np.cos(data['temp'].values * np.pi * 2 / 365 * 24)
    # data['sin_temp'] = np.sin(data['temp'] * 2 * np.pi / 365 * 24)
    #
    # data['cos_atemp'] = np.cos(data['atemp'] * 2 * np.pi / 365 * 24)
    # data['sin_atemp'] = np.sin(data['atemp'] * 2 * np.pi / 365 * 24)
    #
    # data['cos_humidity'] = np.cos(data['humidity'] * 2 * np.pi / 365 * 24)
    # data['sin_humidity'] = np.sin(data['humidity'] * 2 * np.pi / 365 * 24)
    #
    # data['cos_windspeed'] = np.cos(data['windspeed'] * 2 * np.pi / 365 * 24)
    # data['sin_windspeed'] = np.sin(data['windspeed'] * 2 * np.pi / 365 * 24)
    #
    feature_columns = [col for col in data.columns if col not in ['datetime', 'date']]
    
# 2*pi*train_factor$temp/365*24),cos(2*pi*train_factor$temp/365*24),sin(2*pi*train_factor$atemp/365*24),cos(2*pi*train_factor$atemp/365*24),sin(2*pi*train_factor$humidity/365*24),cos(2*pi*train_factor$humidity/365*24),sin(2*pi*train_factor$windspeed/365*24),cos(2*pi*train_factor$windspeed/365*24)
    return data[feature_columns]