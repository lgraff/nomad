#from .dynamic import assign_edge_travel_time

import re
import networkx as nx
from nomad import utils

def nx_to_df(G_sn):
    '''Convert supernetwork object to pandas df, keyed by edge. Do minimal processing.'''

    # Convert the supernetwork object to a pandas df, keyed by the edge. Columns are edge attributes.
    df_G = nx.to_pandas_edgelist(G_sn.graph)
    # Add the node type of each edge's source and target
    df_G['source_node_type'] = df_G['source'].map(lambda x: re.sub('[^a-zA-Z]+', '', x[:3]))  # only the first 3 characters is a hack to avoid rtL, rtA, etc.
    df_G['target_node_type'] = df_G['target'].map(lambda x: re.sub('[^a-zA-Z]+', '', x[:3]))
    df_G['edge_type'] = df_G.apply(utils.rename_mode_type, axis=1) # get the type of the edge (e.g., bs, park, od_cnx) based on source and target nodes

    # Impute frc and speed_limit data for connection and walking edges. This is necessary to establish predicted crash risk.
    df_G.loc[df_G['edge_type'].isin(['bs_cnx', 'z_cnx', 'park']), 'frc'] = 4   # assume frc=4 for connection edges
    df_G.loc[df_G['edge_type'].isin(['bs_cnx', 'z_cnx', 'park']), 'speed_lim'] = 5 # assume speedlim=5 mph for connection eges
    df_G.loc[df_G['mode_type'] == 'w', 'frc'] = 3   # assume frc=3 for walk edges
    df_G.loc[df_G['mode_type'] == 'w', 'speed_lim'] = 30 # assume speedlim=30 mph for walking edges
    df_G['const'] = 1

    # Impute length=0 for board, alight, and t_wait edges. This serves as a placeholder for downstream calculations
    df_G.loc[df_G['mode_type'].isin(['board','alight','t_wait']), 'length_m'] = 0
    
    # Estimate a length for pt traversal edges. This is necessary for risk and discomfort calculations 
    high_speed_idx = (df_G['mode_type'] == 'pt') & ((df_G['length_m'] / df_G['avg_tt_sec'] > 13.4))
    df_G.loc[high_speed_idx, 'length_m'] = df_G.loc[high_speed_idx]['avg_tt_sec'] * 13.4    # 30 mph = 13.4 m/s

    return df_G