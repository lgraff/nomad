import pandas as pd
import numpy as np

def assign_edge_travel_time(df_G, hr, minute, config):
    '''Assign travel time cost to each edge at a given hr:min timestamp. Return df, keyed by edge, with travel time as an attribute.'''
    
    data_path = config['paths']['data']
    
    # Read in inrix travel time ratio data 
    df_tt_ratio = pd.read_csv(data_path['travel_time_ratio'])
    
    # Extract results for for the given hour/min time specifically
    df_tt_ratio_given_time = df_tt_ratio[((df_tt_ratio['hr'] == hr ) & (df_tt_ratio['min'] == minute))]
    tt_mult_by_frc = dict(zip(df_tt_ratio_given_time.frc, df_tt_ratio_given_time.tt_ratio))

    # PUBLIC TRANSIT: BOARDING, TRAVERSAL, and ALIGHTING
    col_dtypes = {'route_id':str, 'direction_id':str, 'stop_id':str, 'headway_mean':np.float64}
    df_pt_headway = pd.read_csv(data_path['PT_headway_static'], dtype = col_dtypes)     # use df_pt_headway as lookup table
    df_boarding = df_G[df_G.mode_type == 'board'][['source','target','mode_type']]
    df_boarding[['stop_id','route_id','direction_id','stop_seq']] = df_boarding.copy()['target'].str.split('_',expand=True)
    df_boarding[['rt','stop_id']] = df_boarding.copy()['stop_id'].str.split('rt',expand=True)
    df_boarding['direction_id'] = df_boarding['direction_id'].astype('str')
    df_boarding = df_boarding[['source','target','mode_type','route_id','direction_id','stop_id']]
    df_boarding_headway = df_boarding.merge(df_pt_headway, how='inner', on=['route_id','direction_id','stop_id'])[['source','target','headway_mean']]
    df_boarding_headway['mode_type'] = 'board'
    df_boarding_headway['avg_tt_sec'] = df_boarding_headway['headway_mean'] / 2 # calculation for avg waiting time, see literature for precedent

    df_pt_trav = df_G[df_G.mode_type == 'pt'][['source','target','mode_type','avg_tt_sec']].reset_index(drop=True)
    # assume frc = 2 for traversal edges
    df_pt_trav['avg_tt_sec'] = tt_mult_by_frc[2] * df_pt_trav['avg_tt_sec']

    df_alight = df_G[df_G.mode_type == 'alight'][['source','target','mode_type']].reset_index(drop=True)
    df_alight['avg_tt_sec'] = config['speed']['ALIGHTING_TIME']  # sec, can be changed if desired
    df_alight['mode_type'] = 'alight'
    # concatenate all pt dfs together
    cols_keep = ['source','target','mode_type','avg_tt_sec']
    df_pt_all = pd.concat([df_boarding_headway[cols_keep], df_pt_trav[cols_keep], df_alight[cols_keep]], axis=0)
    df_pt_all.loc[:, 'length_m'] = 0 # assign a length of 0m to board/traversal/alight

    # TNC, CARSHARE (inclusive of traversal and waiting edges)
    df_tz = df_G[df_G.mode_type.isin(['z','t','park'])][['source','target','mode_type','length_m','speed_lim','frc']]
    df_tz = df_tz.sort_values(by='frc').reset_index(drop=True)
    df_tz['frc'] = df_tz['frc'].astype('int')
    df_tz['avg_tt_sec'] = df_tz['length_m'] / (df_tz['speed_lim'] * config['conversion_factors']['MILE_TO_METERS'] / 3600)

    # tnc waiting mode
    df_twait = df_G[df_G.mode_type.isin(['t_wait'])].reset_index(drop=True)[['source','target']]
    df_twait['avg_tt_sec'] = config['speed']['TNC_WAIT_TIME'] * 60  # wait time in sec
    df_twait['mode_type'] = 't_wait'
    df_twait['length_m'] = 0

    # OTHER MODES: bikeshare, scooter, walk, microtransit: We will do these modes together since the process is the same. Inherent assumption is that they are not affected by traffic conditions 
    df_other = df_G[df_G.mode_type.isin(['bs','sc','w','mt'])][['source','target','mode_type','etype','length_m']].reset_index(drop=True)  # maybe also keep frc
    # Convert Euclidean distance to network distance by adjusting by a circuity factor (see: circuity factor, levinson)
    circuity_factor = config['CIRCUITY_FACTOR']
    mask = df_other['mode_type'].isin(['w', 'mt'])
    df_other.loc[mask, 'length_m'] *= circuity_factor
    speeds = {'bs':config['speed']['BIKE'], 'sc':config['speed']['SCOOT'], 'w':config['speed']['WALK'], 'mt':config['speed']['MICROTRANSIT']}
    df_other['speed'] = df_other['mode_type'].map(speeds)
    df_other['avg_tt_sec'] = df_other['length_m'] / df_other['speed']

    # Add an inconvenience cost (in units of travel time) associated with transferring by walking
    df_other.loc[df_other.etype=='transfer', 'avg_tt_sec'] += (config['time_factors']['INCONVENIENCE_COST'] * 60)
    # Add a microtransit wait time for microtransit transfers (headway / 2)
    microtransit_wait = (config['microtransit_headway'] * 60) / 2   # seconds
    df_other.loc[df_other.mode_type=='mt', 'avg_tt_sec'] += microtransit_wait

    # Combine dfs of the different modes
    cols_keep = cols_keep + ['length_m']
    df_cost = pd.concat([df_pt_all[cols_keep], df_tz[cols_keep], df_twait[cols_keep], df_other[cols_keep]], axis=0)
    return df_cost