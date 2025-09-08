
""" This script performs sensitivity analysis on different parameter (VOT and scooter price) values."""

import os
import sys
from pathlib import Path
import csv
import json
import functools
import multiprocessing as mp
import pickle

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from nomad import costs
from nomad import shortest_path as sp
import macposts
from nomad.costs.nodes import dynamic

from conf import config

# ---- User choice of sensitivity analysis type: change this only ----
ANALYSIS_TYPE = "vot"        # options: "vot" or "scooter_price"

# Params
GRAPH_SN_PATH = Path().resolve() / 'graphs' / 'graph_sn.pkl'
TDSP_FOLDER = Path().resolve() / 'experiment_output' / 'tdsp_files_sn'
MAX_INTERVAL = config['time_factors']['NUM_INTERVALS']  

# For the experiments: choose orgs and dsts
org_geo_list = ['1306002', '5648002', '1209002', '5623001', '5623003', '4838002']  # org geos of interest
dst_geo = '9822001' #'0402002'  # dst geo of interest

if ANALYSIS_TYPE == "vot":
    VOTS = list(range(0, 22, 2))  # Value of Time adjustments
    PARAM_LIST = [f"vot{v}" for v in VOTS]  # labels like 'vot0', 'vot2', ...
    OUTPUT_CSV = TDSP_FOLDER / "tdsp_results_vot.csv"
elif ANALYSIS_TYPE == "scooter_price":
    SCOOTER_PRICES = [0.09, 0.14, 0.19, 0.24, 0.29, 0.34]
    PARAM_LIST = [f"sc{p:.2f}" for p in SCOOTER_PRICES]  # labels like 'sc0.09'
    OUTPUT_CSV = TDSP_FOLDER / "tdsp_results_scooter.csv"
else:
    raise ValueError(f"Unknown ANALYSIS_TYPE: {ANALYSIS_TYPE}")


# **Note**: depending on the machine you are using (to avoid memory issues), you *may* need to run the sensitivity analysis a few times as necessary: 
# i.e., once with vot in range(0,12,2) and once with vot in range (12,22,2). Append the data using csv reader each time .

def main():
    # Run sensitivity analysis for value-of-time (VOT) and scooter prices.
    with open(GRAPH_SN_PATH, 'rb') as inp:
        G_sn = pickle.load(inp)

    nid_map = get_nid_map(G_sn)
    inv_nid_map = dict(zip(nid_map.values(), nid_map.keys()))

    # Get params for MAC POSTS tdsp function
    filename = TDSP_FOLDER / 'td_link_cost_vot0'
    with open(filename, 'r') as file:
        next(file)
        num_rows_link_file = sum(1 for _ in file)
    filename = TDSP_FOLDER / 'td_node_cost'
    with open(filename, 'r') as file:
        next(file)
        num_rows_node_file = sum(1 for _ in file)
    sp.write_config(TDSP_FOLDER, 'graph', num_rows_link_file, num_rows_node_file)     

    # Get org and dst IDs
    orgID_list = [inv_nid_map['org'+org_geo] for org_geo in org_geo_list]
    dstID = inv_nid_map['dst'+dst_geo]

    # Define a partial TDSP function with fixed arguments
    tdsp_sensitivity_partial = functools.partial(tdsp_sensitivity, TDSP_FOLDER, orgID_list, dstID, num_rows_link_file, num_rows_node_file)

    # Run the TDSP sensitivity analysis in parallel
    with mp.Pool(processes=mp.cpu_count()-16) as pool:
        tdsp_data = pool.map(tdsp_sensitivity_partial, PARAM_LIST)

    # Write results to csv
    header = ['org', 'dst', 'param', 'timestamp', 'node_seq', 'link_seq', 'gtc_tot', 'tt_tot']
    filepath = OUTPUT_CSV
    file_exists = os.path.isfile(filepath)
    
    tdsp_data_lists = [item for sublist in tdsp_data for item in sublist]
    with open(filepath, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        if not file_exists:
            csvwriter.writerow(header)  # Write header only if the file does not exist
        csvwriter.writerows(tdsp_data_lists)

def tdsp_sensitivity(TDSP_FOLDER, orgID_list, dstID, num_rows_link_file, num_rows_node_file, param):
    '''Run time-dependent shortest path (tdsp) sensitivity analysis, changing the value of "param" for each function call.'''
    print(param, 'start \n')
    tdsp_data = []  # list to store all tdsp data

    link_cost_file = f"td_link_cost_{param}"

    # Invoke TDSP api from mac-posts
    tdsp_api = macposts.tdsp_api()
    tdsp_api.initialize(str(TDSP_FOLDER), MAX_INTERVAL, num_rows_link_file, num_rows_node_file)
    tdsp_api.read_td_cost_txt(str(TDSP_FOLDER), 'td_link_tt', 'td_node_tt', link_cost_file, 'td_node_cost')

    print('TDSP api has successfully read the files')

    tdsp_api.build_tdsp_tree(dstID)

    for orgID in orgID_list:
        for timestamp in range(0, MAX_INTERVAL, 6):
            tdsp_arr = tdsp_api.extract_tdsp(orgID, timestamp)
            node_seq = json.dumps(list(tdsp_arr[:,0]))  
            link_seq = json.dumps(list(tdsp_arr[:-1,1]))
            gtc_total = tdsp_arr[0,2]
            tt_total = tdsp_arr[0,3]
            tdsp_data.append([orgID, dstID, param, timestamp, node_seq, link_seq, gtc_total, tt_total])

    print(param, "is complete")

    return tdsp_data

def get_nid_map(G_sn):
    df_edge_info = costs.edges.nx_to_df(G_sn)
    node_set = sorted(list(set(df_edge_info['source']).union(set(df_edge_info['target']))))
    nid_map = dict(zip(range(len(node_set)), node_set))
    return nid_map

if __name__ == "__main__":
    main()