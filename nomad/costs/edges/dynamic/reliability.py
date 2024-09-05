
import pandas as pd
import numpy as np

from nomad import conf
from nomad import costs

def assign_edge_reliability(df_tt_dynamic, time_start, time_end, interval_spacing):
    df_rel_ratio = pd.read_csv(conf.reliability_ratio_path)
    df_rel_ratio_ext = costs.edges.dynamic.extend_inrix_data(df_rel_ratio, 'rel_ratio', time_start, time_end, interval_spacing).sort_values(by=['frc','sec_after_midnight']).reset_index(drop=True) 

    # Calculate by mode
    # Start with those who have a constant reliability factor (not dependent on time of day)
    interval_columns = [col for col in df_tt_dynamic.columns if col.startswith('i')]
    rel_factors = {'alight':1, 't_wait': conf.TNC_WAIT_RELIABILITY, 'bs':1, 'sc':1, 'w':1}
    df_dict = {}
    for mode, rel_factor in rel_factors.items():
        df_mode = df_tt_dynamic[df_tt_dynamic.mode_type == mode].reset_index(drop=True)
        rel_arr = rel_factor * df_mode[interval_columns].values
        df_intervals = pd.DataFrame(rel_arr, columns=[f'i{i}' for i in range(rel_arr.shape[1])])
        df_edge_info = df_mode[['source', 'target', 'mode_type']]
        df_rel_intervals = pd.concat([df_edge_info, df_intervals], axis=1)
        df_dict[mode] = df_rel_intervals

    # For boarding edges: calculate reliability as the waiting time (as a function of dept time) + average headway / 2
    df_static = pd.read_csv(conf.PT_headway_path_static)
    df_static['route_id_abbr'] = 'rt' + df_static['stop_id'] + '_' + df_static['route_id'] + '_' + df_static['direction_id'].astype(str)
    df_board = df_tt_dynamic[df_tt_dynamic['mode_type'] == 'board'].copy()
    df_board['route_id_abbr'] = df_board['target'].str.rsplit('_',n=1).str[0]
    df_board = pd.merge(df_board, df_static, on='route_id_abbr', how='left')
    rel_full_arr = df_board[interval_columns].values + (df_board['headway_mean'].values.reshape((-1,1)) / 2)
    df_intervals = pd.DataFrame(rel_full_arr, columns=[f'i{i}' for i in range(rel_full_arr.shape[1])])
    # Store the data
    df_edge_info = df_board[['source', 'target', 'mode_type']]
    df_dict['board'] = pd.concat([df_edge_info, df_intervals], axis=1)  

    # For vehicle modes: Get time-dependent travel time. Take average travel time and multiply by the reliaiblity ratio for the given departure time
    veh_modes = ['pt','t','z']  
    for v in veh_modes:
        for frc in [2,3,4]:
            # Get travel time ratio by frc, departure time
            rel_ratio_arr = df_rel_ratio_ext[df_rel_ratio_ext['frc'] == frc]['rel_ratio'].values.T
            # Get subset of edges for this vehicle mode and frc pair
            tt_subset_by_frc = df_tt_dynamic[(df_tt_dynamic['mode_type'] == v) & (df_tt_dynamic['frc'] == frc)][interval_columns].values
            # Get identifying information of the edge
            df_edge_info = df_tt_dynamic[(df_tt_dynamic['mode_type'] == v) & (df_tt_dynamic['frc'] == frc)][['source','target','mode_type']].reset_index(drop=True)
            # Use matrix multiplication to get dynamic travel time (i.e. travel time for different edge entry times)
            rel_full_arr = tt_subset_by_frc * rel_ratio_arr
            df_intervals = pd.DataFrame(rel_full_arr, columns=[f'i{i}' for i in range(rel_full_arr.shape[1])])
            # Store the data
            df_dict[(v, frc)] = pd.concat([df_edge_info, df_intervals], axis=1)  

    df_rel_dynamic = pd.concat(list(df_dict.values()), ignore_index=True).reset_index(drop=True)
    df_rel_dynamic.sort_values(by=['source','target','mode_type'], inplace=True)
    
    return df_rel_dynamic