
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nomad import conf

def add_time_cols(df, datetime_col):
    '''Extract month, day, hour, minute from datetime.'''
    df['month'] = df[datetime_col].dt.month
    df['day'] = df[datetime_col].dt.day
    df['time'] = df[datetime_col].dt.time
    df['hour'] = df[datetime_col].dt.hour
    df['minute'] = df[datetime_col].dt.minute
    #return df 

def intraday_variation(df, frc, hour_start, hour_end):
    '''For a given frc and a single representative day: return a df of the mean travel time across all roads, by timestamp (at 5 min intervals).'''
    df = df.copy()
    df = df[df['frc'] == frc]
    condition = (df.day == 1) & (df.hour.between(hour_start,hour_end-1))  # day = 1 for a single representative day
    df_time = df[condition]
    df_grouped = df_time.groupby('measurement_tstamp')[['hour','minute','travel_time_seconds']].mean().reset_index()  # for the given frc, take the mean travel time at a timestamp
    return df_grouped

def day2day_variation(df, frc, hour, minute):
    '''For a given frc, hr, min combination, return a list of the reliability ratio.
       The reliability ratio is the ratio of max (avg across all roads, timestamps) travel time to min (avg across all roads, timestamps) travel time.
    '''
    df_subset = df[((df['frc'].values == frc) & (df['hour'].values == hour) & (df['minute'].values == minute))] # subset by frc
    df_day = df_subset.groupby('day')[['hour', 'minute', 'travel_time_seconds']].mean().sort_values(by='travel_time_seconds', ascending=True)
    minTT = df_day['travel_time_seconds'].min()
    maxTT = df_day['travel_time_seconds'].max()
    reliability_ratio = maxTT/minTT
    return([frc, hour, minute, reliability_ratio])


def inrix_to_ratios(inrix_travel_time_inpath, inrix_roadID_inpath, start_time, end_time, tt_ratio_outpath, rel_ratio_outpath):
    '''Read inrix data from csv files, where each row is unique by roadID and timestamp.
       --Only consider timestamps between start_time and end_time
       --Estimate a "reliability ratio" on the frc level, which is the ratio of maximum (avg across roads) to minimum (avg across roads) travel time across days in the sample.
       --Estimate "travel time ratio" on the frc level, which is the ratio of travel time (avg across roads) at a given timestamp to travel time (avg across roads) at 7am (assumed free flow).
    '''
    # data_path = '/home/lgraff/Desktop/Data_Testing/'
    # filepath = os.path.join(data_path, 'Allegheny_sample_xd_part1', 'Allegheny_sample_xd_part1.csv')
    # df = pd.read_csv(filepath)
    # filepath = os.path.join(data_path, 'Allegheny_sample_xd_part1', 'XD_Identification.csv')
    # df_xd = pd.read_csv(filepath)

    # start_time = conf.start_time
    # end_time = conf.end_time

    # # merge tt with id data
    # df_merge = df.merge(df_xd, how='inner', left_on='xd_id', right_on='xd')

    # read data
    df = pd.read_csv(inrix_travel_time_inpath)
    df_xd = pd.read_csv(inrix_roadID_inpath)
    # merge tt with id data
    df_merge = df.merge(df_xd, how='inner', left_on='xd_id', right_on='xd')

    # add datetime columns
    df_merge['measurement_tstamp'] = pd.to_datetime(df_merge['measurement_tstamp']) 
    add_time_cols(df_merge, 'measurement_tstamp')
    tt_ratios = []
    for frc in df_merge.frc.unique():
        df_intraday_frc = intraday_variation(df_merge, frc, start_time, end_time)  # hour | minute | mean_travel_time (for selected frc)
        for hr in df_intraday_frc.hour.unique().astype('int'):
            for min in range(0,60,5):  # inrix data is every 5 min
                # get the base case travel time associated with the 7am; assumption is that 7am is free flow speed
                condition = (df_intraday_frc['hour'] == start_time) & (df_intraday_frc['minute'] == 0)
                base_time = df_intraday_frc[condition]['travel_time_seconds'].values[0]
                # get the travel time at the selected timestamp
                condition = (df_intraday_frc['hour'] == hr) & (df_intraday_frc['minute'] == min)
                tt_timestamp = df_intraday_frc[condition]['travel_time_seconds'].values[0]
                # check the ratio between tt_timestamp and the base_time. if less than 1, adjust it to 1 since we impose 7am as free flow
                tt_ratio = round(max(tt_timestamp/base_time, 1), 3)
                tt_ratios.append((frc, hr, min, tt_ratio))

    # create a lookup table for travel time and reliability ratios
    df_tt_ratio = pd.DataFrame(tt_ratios, columns =['frc', 'hr', 'min', 'tt_ratio']).sort_values(by=['frc','hr','min'])
    rel_ratio = [day2day_variation(df_merge, frc, hr, min) for frc in df_merge.frc.unique() for hr in [start_time, end_time-1] for min in range(0,60,5)]
    df_rel_ratio = pd.DataFrame(rel_ratio, columns=['frc', 'hour', 'minute', 'rel_ratio']).sort_values(by=['frc','hour','minute'])

    df_tt_ratio.to_csv(tt_ratio_outpath, index=False)
    df_rel_ratio.to_csv(rel_ratio_outpath, index=False)