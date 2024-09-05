
# description of file:

# libraries 

import networkx as nx
import re
import pandas as pd
import numpy as np

from nomad import conf
from nomad import utils
from nomad import costs

def assign_edge_travel_time(df_G, time_start, time_end, interval_spacing):
# Note: 'avg_tt_sec' is the amount of time spent traversing a link assuming free flow speed. The start of the interval at 7am is assumed to be free flow
# frcs: {0: active modes, 1: boarding, 2,3,4: vehicle modes}

    #df_G = costs.edges.nx_to_df(G_sn)
    df_tt_ratio = pd.read_csv(conf.travel_time_ratio_path)

    # special case: PUBLIC TRANSIT BOARDING. 
    df_boarding = df_G[df_G.mode_type == 'board'][['source','target','mode_type','length_m']]
    df_boarding[['stop_id','route_id','direction_id','stop_seq']] = df_boarding.copy()['target'].str.split('_',expand=True)
    df_boarding[['rt','stop_id']] = df_boarding.copy()['stop_id'].str.split('rt',expand=True)
    df_boarding['direction_id'] = df_boarding['direction_id'].astype('str')
    df_boarding = df_boarding[['source','target','mode_type','route_id','direction_id','stop_id']]

    col_dtypes = {'route_id':str, 'direction_id':str, 'stop_id':str, 'traveler_arrival_time':np.int64, 'headway':np.int64}
    processed_headway = pd.read_csv(conf.PT_headway_path_dynamic, dtype=col_dtypes)
    df_boarding_headway = df_boarding.merge(processed_headway, how='inner', on=['route_id','direction_id','stop_id'])

    arr_times = df_boarding_headway.traveler_arrival_time.unique().tolist()
    interval_mapping = dict(zip(arr_times,range(len(arr_times))))
    df_boarding_headway['i'] = df_boarding_headway['traveler_arrival_time'].map(interval_mapping)
    df_boarding_headway = df_boarding_headway.pivot(index=['source','target','mode_type'],columns='i', values='headway').add_prefix('i').reset_index()

    # just placeholders
    df_boarding_headway.insert(3, 'avg_tt_sec', df_boarding_headway['i0'])
    df_boarding_headway.insert(3, 'length_m', 0)
    df_boarding_headway.insert(3, 'frc', np.nan)

    # PUBLIC TRANSIT: TRAVERSAL and ALIGHTING
    df_pt_trav = df_G[df_G.mode_type == 'pt'][['source','target','mode_type','length_m','avg_tt_sec']].reset_index(drop=True)
    # assume frc = 2 for traversal edges
    #df_pt_trav['avg_tt_sec'] = tt_mult_by_frc[2] * df_pt_trav['avg_tt_sec']
    df_pt_trav['frc'] = 2 # assumption

    df_alight = df_G[df_G.mode_type == 'alight'][['source','target','mode_type','length_m']].reset_index(drop=True)
    df_alight['avg_tt_sec'] = conf.ALIGHTING_TIME  # sec, can be changed if desired
    df_alight['mode_type'] = 'alight'
    df_alight['frc'] = np.nan # placeholder

    # TNC, CARSHARE (inclusive of traversal and waiting edges)
    df_tz = df_G[df_G.mode_type.isin(['z','t','park'])][['source','target','mode_type','length_m','speed_lim','frc']]
    df_tz = df_tz.sort_values(by='frc').reset_index(drop=True)
    df_tz['frc'] = df_tz['frc'].astype('int')
    df_tz['avg_tt_sec'] = df_tz['length_m'] / (df_tz['speed_lim'] * conf.MILE_TO_METERS / 3600)

    # tnc waiting mode
    df_twait = df_G[df_G.mode_type.isin(['t_wait'])].reset_index(drop=True)[['source','target','length_m']]
    df_twait['avg_tt_sec'] = conf.TNC_WAIT_TIME * 60  # wait time in sec
    df_twait['mode_type'] = 't_wait'
    df_twait['frc'] = np.nan # placeholder

    # ACTIVE MODES: bikeshare, walk, and scooter: we will do these modes together since the process is the same
    # inherent assumption is that they are not affected by traffic conditions 
    df_active = df_G[df_G.mode_type.isin(['bs','sc','w'])][['source','target','mode_type','etype','length_m']].reset_index(drop=True)  # maybe also keep frc
    # adjust euclidean walking distance by a circuity factor (see: circuity factor, levinson)
    circuity_factor = conf.CIRCUITY_FACTOR
    df_active.loc[df_active['mode_type'] == 'w', 'length_m'] = circuity_factor * df_active.loc[df_active['mode_type'] == 'w', 'length_m']
    speeds = {'bs':conf.BIKE_SPEED, 'sc':conf.SCOOT_SPEED, 'w':conf.WALK_SPEED}
    df_active['speed'] = df_active['mode_type'].map(speeds)
    df_active['avg_tt_sec'] = df_active['length_m'] / df_active['speed']
    # add an inconvenience cost (in units of travel time) associated with transferring
    inc = conf.INCONVENIENCE_COST # minutes
    df_active.loc[df_active.etype=='transfer', 'avg_tt_sec'] =  df_active.loc[df_active.etype=='transfer', 'avg_tt_sec'] + (inc*60)
    df_active['frc'] = np.nan # placeholder

    # Combine dfs of the different modes, keyed by (source, target), to facilitate calculation of time-dependent travel times using the tt ratio by frc approach
    cols_keep = ['source','target','mode_type','length_m','frc','avg_tt_sec']   
    df_cost = pd.concat([df_pt_trav[cols_keep], df_alight[cols_keep], df_tz[cols_keep], df_twait[cols_keep], df_active[cols_keep]], axis=0).sort_values(by=['source','target'])

    df_tt_ratio_ext = costs.edges.dynamic.extend_inrix_data(df_tt_ratio, 'tt_ratio', time_start, time_end, interval_spacing).sort_values(by=['frc','sec_after_midnight']).reset_index(drop=True)  # extend the data 
    
    # Get time-dependent travel time. Take 7am free flow travel time (called 'avg_tt_sec') and multiply by the travel time ratio relative to 7am
    # e.g. if 7am travel time is 10 sec and 7:05am ratio is 1.1, then 7:05am travel time is 10 & 1.1 = 11 sec
    df_dict = {}
    veh_modes = ['pt','t','z']  
    for v in veh_modes:
        for frc in [2,3,4]:
            # Get travel time ratio by frc, departure time
            tt_ratio_arr = df_tt_ratio_ext[df_tt_ratio_ext['frc'] == frc]['tt_ratio'].values.T
            # Get subset of edges for this vehicle mode and frc pair
            tt_subset_by_frc = df_cost[(df_cost['mode_type'] == v) & (df_cost['frc'] == frc)]['avg_tt_sec'].values.reshape(-1,1)
            # Get identifying information of the edge
            df_edge_info = df_cost[(df_cost['mode_type'] == v) & (df_cost['frc'] == frc)].reset_index(drop=True)
            # Use matrix multiplication to get dynamic travel time (i.e. travel time for different edge entry times)
            tt_full_arr = tt_subset_by_frc * tt_ratio_arr
            df_intervals = pd.DataFrame(tt_full_arr, columns=[f'i{i}' for i in range(tt_full_arr.shape[1])])
            # Impute length of pt traversal edges for (needed for downstream discomfort calculations). We estimate as avg_speed [25 mph] * 
            # Store the data
            df_dict[(v, frc)] = pd.concat([df_edge_info, df_intervals], axis=1)  

    df_tt_vehicle = pd.concat(list(df_dict.values()), ignore_index=True).reset_index(drop=True)

    active_modes = ['bs','sc','w']
    tt_full_arr = df_cost[df_cost.mode_type.isin(active_modes + ['t_wait', 'alight'])]['avg_tt_sec'].values.reshape(-1,1) * np.ones((1,tt_full_arr.shape[1])) 
    df_intervals = pd.DataFrame(tt_full_arr, columns=[f'i{i}' for i in range(tt_full_arr.shape[1])])
    df_edge_info = df_cost[df_cost.mode_type.isin(active_modes + ['t_wait', 'alight'])].reset_index(drop=True)
    df_tt_active = pd.concat([df_edge_info, df_intervals], axis=1)

    # now concat all dfs
    cols_keep = ['source','target','mode_type','length_m','frc'] + [f'i{i}' for i in range(tt_full_arr.shape[1])]

    df_tt_dynamic = pd.concat([df_boarding_headway[cols_keep], df_tt_active[cols_keep], df_tt_vehicle[cols_keep]], ignore_index=True)
    df_tt_dynamic.sort_values(by=['source','target','mode_type'], inplace=True)
    
    return df_tt_dynamic