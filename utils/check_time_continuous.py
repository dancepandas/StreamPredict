from datetime import timedelta

import pandas as pd

def check_time_continuous(data_file):
    df = pd.read_csv(data_file)
    df['datetime']  = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S')
    df.set_index("datetime", inplace=True)
    all_datetime_list = df.index.tolist()

    for j in range(1,len(all_datetime_list)):
        time_diff = all_datetime_list[j] - all_datetime_list[j - 1]
        if time_diff != timedelta(minutes=5):
            return False

    return True