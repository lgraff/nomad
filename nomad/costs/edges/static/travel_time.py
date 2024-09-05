import pandas as pd
import pickle
import re
import networkx as nx
import numpy as np

from nomad import conf
from nomad import utils
from nomad import costs

# def nx_to_df(G_sn):
#     '''Convert supernetwork object to pandas df, keyed by edge. Do minimal processing.'''
#     # Read in the supernetwork as an object
#     # with open(conf.G_sn_path, 'rb') as inp:
#     #     G_sn = pickle.load(inp)
    
    
#     # Convert the supernetwork object to a pandas df, keyed by the edge. Columns are edge attributes.
#     df_G = nx.to_pandas_edgelist(G_sn.graph)
#     # Add the node type of each edge's source and target
#     df_G['source_node_type'] = df_G['source'].map(lambda x: re.sub('[^a-zA-Z]+', '', x[:3]))  # only the first 3 characters is a hack to avoid rtL, rtA, etc.
#     df_G['target_node_type'] = df_G['target'].map(lambda x: re.sub('[^a-zA-Z]+', '', x[:3]))
#     df_G['edge_type'] = df_G.apply(utils.rename_mode_type, axis=1) # get the type of the edge (e.g., bs, park, od_cnx) based on source and target nodes

#     # Impute frc and speed_limit data for connection and walking edges. This is necessary to establish predicted crash risk.
#     df_G.loc[df_G['edge_type'].isin(['bs_cnx', 'z_cnx', 'park']), 'frc'] = 4   # assume frc=4 for connection edges
#     df_G.loc[df_G['edge_type'].isin(['bs_cnx', 'z_cnx', 'park']), 'speed_lim'] = 5 # assume speedlim=5 mph for connection eges
#     df_G.loc[df_G['mode_type'] == 'w', 'frc'] = 3   # assume frc=3 for walk edges
#     df_G.loc[df_G['mode_type'] == 'w', 'speed_lim'] = 35 # assume speedlim=35 mph for walking edges
#     df_G['const'] = 1
#     return df_G

def assign_edge_travel_time(df_G, hr, minute):
    '''Assign travel time cost to each edge at a given hr:min timestamp. Return df, keyed by edge, with travel time as an attribute.'''
    # Read in inrix travel time ratio data 
    df_tt_ratio = pd.read_csv(conf.travel_time_ratio_path)
    # Extract results for for the given hour/min time specifically
    df_tt_ratio_given_time = df_tt_ratio[((df_tt_ratio['hr'] == hr ) & (df_tt_ratio['min'] == minute))]
    tt_mult_by_frc = dict(zip(df_tt_ratio_given_time.frc, df_tt_ratio_given_time.tt_ratio))

    # PUBLIC TRANSIT: BOARDING, TRAVERSAL, and ALIGHTING
    col_dtypes = {'route_id':str, 'direction_id':str, 'stop_id':str, 'headway_mean':np.float64}
    df_pt_headway = pd.read_csv(conf.PT_headway_path_static, dtype = col_dtypes)     # use df_pt_headway as lookup table
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
    df_alight['avg_tt_sec'] = conf.ALIGHTING_TIME  # sec, can be changed if desired
    df_alight['mode_type'] = 'alight'
    # concatenate all pt dfs together
    cols_keep = ['source','target','mode_type','avg_tt_sec'] #,'reliability']
    df_pt_all = pd.concat([df_boarding_headway[cols_keep], df_pt_trav[cols_keep], df_alight[cols_keep]], axis=0)
    df_pt_all.loc[:, 'length_m'] = 0 # assign a length of 0m to board/traversal/alight

    # TNC, CARSHARE (inclusive of traversal and waiting edges)
    df_tz = df_G[df_G.mode_type.isin(['z','t','park'])][['source','target','mode_type','length_m','speed_lim','frc']]
    df_tz = df_tz.sort_values(by='frc').reset_index(drop=True)
    df_tz['frc'] = df_tz['frc'].astype('int')
    df_tz['avg_tt_sec'] = df_tz['length_m'] / (df_tz['speed_lim'] * conf.MILE_TO_METERS / 3600)

    # tnc waiting mode
    df_twait = df_G[df_G.mode_type.isin(['t_wait'])].reset_index(drop=True)[['source','target']]
    df_twait['avg_tt_sec'] = conf.TNC_WAIT_TIME * 60  # wait time in sec
    df_twait['mode_type'] = 't_wait'
    df_twait['length_m'] = 0

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

    # combine dfs of the different modes
    cols_keep = cols_keep + ['length_m']
    df_cost = pd.concat([df_pt_all[cols_keep], df_tz[cols_keep], df_twait[cols_keep], df_active[cols_keep]], axis=0)
    return df_cost