
# description of file:
import os
import sys
from pathlib import Path
import csv
import json
import functools
import multiprocessing as mp
import pickle

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '.')))
from nomad import conf
from nomad import costs
from nomad import shortest_path as sp
import MNMAPI

from nomad.costs.nodes import dynamic

def tdsp_sensitivity(tdsp_folder, orgID_list, dstID, num_rows_link_file, num_rows_node_file, param):
    '''Run time-dependent shortest path (tdsp) sensitivity analysis, changing the value of "param" for each function call.'''
    print(param, 'start \n')
    tdsp_data = []  # list to store all tdsp data

    # Parameters
    max_interval = conf.NUM_INTERVALS      
    # Invoke TDSP api from mac-posts
    tdsp_api = MNMAPI.tdsp_api()
    tdsp_api.initialize(str(tdsp_folder), max_interval, num_rows_link_file, num_rows_node_file)
    tdsp_api.read_td_cost_txt(str(tdsp_folder), 'td_link_tt', 'td_node_tt', 'td_link_cost_' + str(param), 'td_node_cost')

    print('TDSP api has successfully read the files')

    tdsp_api.build_tdsp_tree(dstID)

    for orgID in orgID_list:
        for timestamp in range(0, max_interval, 6):
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

#shm_gtc_arr, gtc_shape, gtc_dtype, shm_tt_arr, tt_shape, tt_dtype

# del linkID_arr
# gc.collect()

if __name__ == "__main__":
    # Run sensitivity analysis for value-of-time (VOT) and scooter prices. Change the value of "param_list" below depending on the analysis.

    # Read in the supernetwork as an object
    graph_sn_path = Path().resolve() / 'project1-use_cases' / 'graph_sn.pkl'
    with open(graph_sn_path, 'rb') as inp:
        G_sn = pickle.load(inp)

    nid_map = get_nid_map(G_sn)
    inv_nid_map = dict(zip(nid_map.values(), nid_map.keys()))

    # Get params for MAC POSTS tdsp function
    tdsp_folder = Path().resolve() / 'project1-use_cases' / 'tdsp_files_sn'

    filename = tdsp_folder / 'td_link_cost_0'
    with open(filename, 'r') as file:
        next(file)
        num_rows_link_file = sum(1 for _ in file)
    filename = tdsp_folder / 'td_node_cost'
    with open(filename, 'r') as file:
        next(file)
        num_rows_node_file = sum(1 for _ in file)

    # For the experiments: choose orgs and dsts
    org_geo_list = ['1306002', '5648002', '1209002', '5623001', '5623003', '4838002']  # org geos of interest
    orgID_list = [inv_nid_map['org'+org_geo] for org_geo in org_geo_list]
    dst_geo = '9822001' #'0402002'  # dst geo of interest
    dstID = inv_nid_map['dst'+dst_geo]

    tdsp_sensitivity_partial = functools.partial(tdsp_sensitivity, tdsp_folder, orgID_list, dstID, num_rows_link_file, num_rows_node_file)

    # to avoid memory issues (which causes program to crash), run this a few times as necessary: once with vot in range(0,12,2) and once with vot in range (12,22,2)
    # append the data using csv reader each time
    
    param_list = ['sc0.24', 'sc0.29', 'sc0.34'] #['sc0.09', 'sc0.14', 'sc0.19'] #, 'sc0.24', 'sc0.29', 'sc0.34'] 
    #param_list = list(range(16,22,2))

    with mp.Pool(processes=mp.cpu_count()-2) as pool:
        tdsp_data = pool.map(tdsp_sensitivity_partial, param_list)

    # Write results to csv
    header = ['org', 'dst', 'param', 'timestamp', 'node_seq', 'link_seq', 'gtc_tot', 'tt_tot']
    filename = "tdsp_results_sc_prices.csv"
    
    tdsp_data_lists = [item for sublist in tdsp_data for item in sublist]
    with open(filename, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        #csvwriter.writerow(header) # header
        csvwriter.writerows(tdsp_data_lists)