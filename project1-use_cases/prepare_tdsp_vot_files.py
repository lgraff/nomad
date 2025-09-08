""" Prepare time-dependent shortest path (TDSP) files for different values of time (VOT)."""

import os
import sys
from pathlib import Path
import gc
import functools
import multiprocessing.shared_memory as shm
import multiprocessing as mp

import numpy as np
import pickle

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from nomad import costs
from nomad import shortest_path as sp
from nomad.costs.nodes import dynamic
from conf import config

# Params
GRAPH_SN_PATH = Path().resolve() / 'graphs' / 'graph_sn.pkl'
TDSP_FOLDER = Path().resolve() / 'experiment_output' / 'tdsp_files_sn'
BETAS = {
        'rel': 10 / 3600,
        'x': 1,
        'risk': 20,
        'disc': 0
        }


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

    # Get time-dep node costs
    df_node_cost_dynamic = costs.nodes.dynamic.get_node_cost_df(G_sn)

    # Create node files
    create_node_files(df_node_cost_dynamic, nid_map, inv_nid_map, inv_linkID_map, TDSP_FOLDER)

    # Create tt files
    sp.prepare_tt_file(config, TDSP_FOLDER, linkID_arr, df_tt_dynamic)

    # Prepare link cost files for different VOTs
    prepare_link_cost_files(BETAS, df_tt_dynamic, df_rel_dynamic, df_price_dynamic, df_risk_dynamic, df_disc_dynamic, interval_columns, linkID_arr, TDSP_FOLDER)

def create_node_files(df_node_cost_dynamic, nid_map, inv_nid_map, inv_linkID_map, TDSP_FOLDER):
    df_node_cost_dynamic[['node_id_from', 'node_id_via', 'node_id_to']] = df_node_cost_dynamic[['node_from', 'node_via', 'node_to']].applymap(lambda x: inv_nid_map[x])
    df_node_cost_dynamic['link_in'] = tuple(zip(df_node_cost_dynamic['node_id_from'], df_node_cost_dynamic['node_id_via']))
    df_node_cost_dynamic['link_out'] = tuple(zip(df_node_cost_dynamic['node_id_via'], df_node_cost_dynamic['node_id_to']))
    df_node_cost_dynamic[['linkID_in', 'linkID_out']] = df_node_cost_dynamic[['link_in', 'link_out']].applymap(lambda x: inv_linkID_map[(nid_map[x[0]], nid_map[x[1]])])
    sp.prepare_node_files(config, TDSP_FOLDER, df_node_cost_dynamic)

def prepare_link_cost_files(BETAS, df_tt_dynamic, df_rel_dynamic, df_price_dynamic, df_risk_dynamic, df_disc_dynamic, interval_columns, linkID_arr, TDSP_FOLDER):

    gtc_arr_no_time = (BETAS['rel'] * df_rel_dynamic[interval_columns].values.astype(np.float16) +
                       BETAS['x'] * df_price_dynamic[interval_columns].values.astype(np.float16) +
                       BETAS['risk'] * df_risk_dynamic[interval_columns].values.astype(np.float16) +
                       BETAS['disc'] * df_disc_dynamic[interval_columns].values.astype(np.float16))

    del df_disc_dynamic, df_rel_dynamic, df_risk_dynamic, df_price_dynamic
    gc.collect()

    shm_gtc_arr, gtc_shape, gtc_dtype = init_shared_memory(gtc_arr_no_time)
    shm_tt_arr, tt_shape, tt_dtype = init_shared_memory(df_tt_dynamic[interval_columns].values.astype(np.float16))

    prepare_gtc_file_sensitivity_partial = functools.partial(prepare_gtc_file_sensitivity, TDSP_FOLDER, linkID_arr, shm_gtc_arr, gtc_shape, gtc_dtype, shm_tt_arr, tt_shape, tt_dtype)

    vot_list = list(range(0, 22, 2))
    with mp.Pool(processes=mp.cpu_count() - 16) as pool:
        pool.map(prepare_gtc_file_sensitivity_partial, vot_list)

def init_shared_memory(array):
    """ Initialize shared memory for a numpy array."""
    shm_array = shm.SharedMemory(create=True, size=array.nbytes)
    shared_array = np.ndarray(array.shape, dtype=array.dtype, buffer=shm_array.buf)
    np.copyto(shared_array, array)
    return shm_array, array.shape, array.dtype

def prepare_gtc_file_sensitivity(TDSP_FOLDER, linkID_arr, shm_gtc_arr, gtc_shape, gtc_dtype, shm_tt_arr, tt_shape, tt_dtype, vot):
    """ Prepare gtc file for a specific value of time (vot)."""
    BETA_tt = vot / 3600
    # Access existing shared memory
    existing_shm_gtc = shm.SharedMemory(name=shm_gtc_arr.name)  
    existing_shm_tt = shm.SharedMemory(name=shm_tt_arr.name)

    # Create gtc (without time) and tt arrays from shared memory
    gtc_arr_no_time = np.ndarray(gtc_shape, gtc_dtype, buffer=existing_shm_gtc.buf)  
    tt_arr = np.ndarray(tt_shape, tt_dtype, buffer=existing_shm_tt.buf)

    # Compute final gtc array
    gtc_arr = gtc_arr_no_time + (BETA_tt * tt_arr)
    filename = 'td_link_cost_vot' + str(vot)
    sp.prepare_gtc_file(config, TDSP_FOLDER, filename, linkID_arr, gtc_arr)

    # Close the shared memory
    existing_shm_gtc.close()
    existing_shm_tt.close()

    print(vot, 'complete')

if __name__ == "__main__":
    main()
