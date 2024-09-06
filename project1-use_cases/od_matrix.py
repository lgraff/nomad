
from pathlib import Path
import pickle
import sys, os
import multiprocessing as mp
import functools
import csv

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '.')))
from nomad import shortest_path as sp
from nomad import costs

def get_nid_map(G):
    df_edge_info = costs.edges.nx_to_df(G)
    node_set = sorted(list(set(df_edge_info['source']).union(set(df_edge_info['target']))))
    nid_map = dict(zip(range(len(node_set)), node_set))
    return nid_map

def get_dstIDs(G, inv_nid_map):
    all_dsts = [n for n in G.graph.nodes if n.startswith('dst')]
    dstID_list = [inv_nid_map[dst_name] for dst_name in all_dsts]
    return dstID_list

def get_orgIDs(G, inv_nid_map):
    all_orgs = [n for n in G.graph.nodes if n.startswith('org')]
    orgID_list = [inv_nid_map[org_name] for org_name in all_orgs]
    return orgID_list

def initializer(tdsp_api, vot):
    """Initialize global variables."""
    global tdsp_api_global
    global vot_global
    tdsp_api_global = tdsp_api
    vot_global = vot

def process_dsts(orgID_list, dstID_list, timestamp_window):
    tdsp_data = []  # list to store all tdsp data
    for dstID in dstID_list:
        tdsp_api_global.build_tdsp_tree(dstID)

        for orgID in orgID_list:
            for timestamp in range(timestamp_window[0], timestamp_window[1], 6):  # 7:30-8:00am  180-361
                tdsp_arr = tdsp_api_global.extract_tdsp(orgID, timestamp)
                node_seq = list(tdsp_arr[:,0])
                link_seq = list(tdsp_arr[:-1,1])
                gtc_total = tdsp_arr[0,2]
                tt_total = tdsp_arr[0,3]
                tdsp_data.append([orgID, dstID, vot_global, timestamp, node_seq, link_seq, gtc_total, tt_total])

    print(dstID_list, 'complete')
    return tdsp_data

def write_data(data, header, filename):
    '''Write data to .csv file.'''
    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(header) # header
        csvwriter.writerows(data)

def calc_td_od_matrix(G, tdsp_folder, vot, BETAS, td_link_cost_filename, od_matrix_filename):
    tdsp_api = sp.prepare_tdsp_api(G, BETAS, tdsp_folder, td_link_cost_filename)
    nid_map = get_nid_map(G)
    inv_nid_map = dict(zip(nid_map.values(), nid_map.keys()))
    orgID_list = get_orgIDs(G, inv_nid_map)
    dstID_list = get_dstIDs(G, inv_nid_map)

    chunk_size = 10
    dst_chunks = [dstID_list[i:i+chunk_size] for i in range(0, len(dstID_list), chunk_size)]  
    
    process_dsts_partial = functools.partial(process_dsts, orgID_list, timestamp_window=(180,361))

    with mp.Pool(initializer=initializer, initargs=(tdsp_api, vot), processes=mp.cpu_count()-1) as pool:
        tdsp_data = pool.map(process_dsts_partial, dst_chunks)

    header = ['org', 'dst', 'vot', 'timestamp', 'node_seq', 'link_seq', 'gtc_tot', 'tt_tot']
    tdsp_data_lists = [item for sublist in tdsp_data for item in sublist]
    write_data(tdsp_data_lists, header, od_matrix_filename)

if __name__ == "__main__":
    # Parameters
    vot = 10
    # beta weighting factors
    BETAS = {
        'tt': vot/3600,
        'rel': 10/3600,
        'x': 1,
        'risk': 20,
        'disc': 0}
    
    mode_list = ['pt', 'pt_bs']
    for m in mode_list:
        try:
            # Read in the supernetwork as an object
            graph_path = Path().resolve() / 'project1-use_cases' / f'graph_{m}.pkl'
            with open(graph_path, 'rb') as inp:
                G = pickle.load(inp)

            tdsp_folder = Path().resolve() / 'project1-use_cases' / f'tdsp_files_{m}'  # Folder that stores TDSP files for each of the graphs
            calc_td_od_matrix(G, tdsp_folder, vot, BETAS, 'td_link_cost_' + str(vot), 'od_matrix_' + m + '.csv')  # Calculate the time-dep OD matrix
        
        except:
            print('Error processing mode ', m)