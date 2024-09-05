import pandas as pd
import pickle
import re
import networkx as nx
import numpy as np

from nomad import conf

def assign_edge_reliability(df_G, df_tt, hr, minute):
    '''Assign travel time cost to each edge at a given hr:min timestamp. Return df, keyed by edge, with travel time & reliability as attributes.
       Since we do not have exact data for the 95% travel time, we approximate it as a scalar multiplied by the average travel time.
       Hence we also need as an argument the df, keyed by edge, that has average travel time'''
    df_rel_ratio = pd.read_csv(conf.reliability_ratio_path)
    # Extract results for for the given hour/min time specifically
    df_rel_ratio_given_time = df_rel_ratio[((df_rel_ratio['hour'] == hr) & (df_rel_ratio['minute'] == minute))]
    rel_mult_by_frc = dict(zip(df_rel_ratio_given_time.frc, df_rel_ratio_given_time.rel_ratio))

    # PUBLIC TRANSIT: BOARDING, TRAVERSAL, and ALIGHTING
    df_boarding = df_tt[df_tt.mode_type == 'board'].reset_index(drop=True)
    waiting_rel_mult = conf.BOARDING_RELIABILITY 
    df_boarding['reliability'] = waiting_rel_mult * df_boarding['avg_tt_sec']

    df_pt_trav = df_tt[df_tt.mode_type == 'pt'].reset_index(drop=True)
    # assume frc = 2 for traversal edges
    df_pt_trav['reliability'] = rel_mult_by_frc[2] * df_pt_trav['avg_tt_sec']

    df_alight = df_tt[df_tt.mode_type == 'alight'].reset_index(drop=True)
    df_alight['reliability'] = 1 * df_alight['avg_tt_sec'] # assume 95 percentile alighting time is same as avg

    # TNC, CARSHARE (inclusive of traversal and waiting edges)
    df_tz = df_tt[df_tt.mode_type.isin(['z','t','park'])] #.sort_values(by='frc').reset_index(drop=True)
    df_tz = pd.merge(df_tz, df_G[['source','target', 'frc']], how='left', on=['source','target'])
    df_tz['frc'] = df_tz['frc'].astype('int')
    df_tz['reliability'] = df_tz['reliability'] = df_tz['frc'] * df_tz['avg_tt_sec']
    # tnc waiting mode
    df_twait = df_tt[df_tt.mode_type.isin(['t_wait'])].reset_index(drop=True)
    df_twait['reliability'] = conf.TNC_WAIT_RELIABILITY * df_twait['avg_tt_sec']

    # ACTIVE MODES: bikeshare, walk, and scooter: we will do these modes together since the process is the same
    # inherent assumption is that they are not affected by traffic conditions 
    df_active = df_tt[df_tt.mode_type.isin(['bs','sc','w'])].reset_index(drop=True)  # maybe also keep frc
    df_active['reliability'] = df_active['avg_tt_sec'].copy()  # make the assumption that 95% travel time = mean travel time for active modes
    #TODO: special case of scooter transfer. use: 95_length_m

    # combine dfs of the different modes
    df_cost = pd.concat([df_boarding, df_pt_trav, df_alight, df_tz.drop(columns='frc'), df_twait, df_active], axis=0)
    return df_cost