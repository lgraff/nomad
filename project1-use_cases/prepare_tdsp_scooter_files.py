""" Prepare time-dependent shortest path (TDSP) files for different scooter prices."""

import os
import sys
from pathlib import Path

import numpy as np
import pickle

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from nomad import costs
from nomad import shortest_path as sp
from nomad.costs.nodes import dynamic
from conf import config

# Parameters
GRAPH_SN_PATH = Path().resolve() / 'graphs' / 'graph_sn.pkl'
TDSP_FOLDER = Path().resolve() / 'experiment_output' / 'tdsp_files_sn'
vot = 10
BETAS = {
    'tt': vot/3600,
    'rel': 10/3600,
    'x': 1,
    'risk': 20,
    'disc': 0}


def main():
    # Read in the supernetwork as an object
    with open(GRAPH_SN_PATH, 'rb') as inp:
        G_sn = pickle.load(inp)
    
    df_G = costs.edges.nx_to_df(G_sn).sort_values(by=['source', 'target', 'mode_type']).reset_index(drop=True)

    # Create graph file and get node/link IDs for subsequent use
    df_G = sp.prepare_graph_file(TDSP_FOLDER, G_sn)
    nid_map = sp.get_nid_map(df_G)
    inv_nid_map = dict(zip(nid_map.values(), nid_map.keys()))
    linkID_map = sp.get_link_id_map(df_G)
    inv_linkID_map = dict(zip(linkID_map.values(), linkID_map.keys()))
    linkID_arr = df_G['linkID'].to_numpy().reshape((-1, 1))

    # Parameters
    NUM_INTERVALS = config['time_factors']['NUM_INTERVALS']
    interval_columns = [f'i{i}' for i in range(NUM_INTERVALS)]

    # Get all time-dep edge costs
    df_tt_dynamic, df_rel_dynamic, df_price_dynamic, df_risk_dynamic, df_disc_dynamic = costs.edges.dynamic.assign_edge_costs(G_sn)

    # Get the indices of scooter traversal links
    sc_trav_links = [link_id for named_link, link_id in inv_linkID_map.items() if named_link[0].startswith('sc') and named_link[1].startswith('sc')]

    # Get the generalized travel cost array, not including the price component
    gtc_arr_no_price = BETAS['rel'] * df_rel_dynamic[interval_columns].values.astype(np.float16) + BETAS['tt'] * df_tt_dynamic[interval_columns].values.astype(np.float16) + BETAS['risk'] * df_risk_dynamic[interval_columns].values.astype(np.float16) + BETAS['disc'] * df_disc_dynamic[interval_columns].values.astype(np.float16)

    # Prepare gtc files for different scooter prices
    for sc_ppmin in [0.09, 0.14, 0.19, 0.24, 0.29, 0.34]:
        price_reduction_pct = (0.39 - sc_ppmin) / 0.39  # percent reduction in scoot link cost
        price_arr = df_price_dynamic[interval_columns].values.astype(np.float16).copy() 
        price_arr[sc_trav_links,:] *= (1-price_reduction_pct) 
        gtc_arr = gtc_arr_no_price + price_arr
        linkID_arr = df_G['linkID'].to_numpy().reshape((-1,1))
        filename = 'td_link_cost_sc' + str(sc_ppmin)
        sp.prepare_gtc_file(config, TDSP_FOLDER, filename, linkID_arr, gtc_arr)
        print(sc_ppmin, 'complete')

    # Get time-dep node costs
    df_node_cost_dynamic = costs.nodes.dynamic.get_node_cost_df(G_sn)

    # Create node files
    create_node_files(df_node_cost_dynamic, nid_map, inv_nid_map, inv_linkID_map, TDSP_FOLDER)

    # Create tt files
    sp.prepare_tt_file(config, TDSP_FOLDER, linkID_arr, df_tt_dynamic)


def create_node_files(df_node_cost_dynamic, nid_map, inv_nid_map, inv_linkID_map, TDSP_FOLDER):
    df_node_cost_dynamic[['node_id_from', 'node_id_via', 'node_id_to']] = df_node_cost_dynamic[['node_from', 'node_via', 'node_to']].applymap(lambda x: inv_nid_map[x])
    df_node_cost_dynamic['link_in'] = tuple(zip(df_node_cost_dynamic['node_id_from'], df_node_cost_dynamic['node_id_via']))
    df_node_cost_dynamic['link_out'] = tuple(zip(df_node_cost_dynamic['node_id_via'], df_node_cost_dynamic['node_id_to']))
    df_node_cost_dynamic[['linkID_in', 'linkID_out']] = df_node_cost_dynamic[['link_in', 'link_out']].applymap(lambda x: inv_linkID_map[(nid_map[x[0]], nid_map[x[1]])])
    sp.prepare_node_files(config, TDSP_FOLDER, df_node_cost_dynamic)


if __name__ == "__main__":
    main()
