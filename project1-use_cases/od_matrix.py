"""Compute time-dependent OD matrices for multiple supernetworks using a single VOT."""

import os
import sys
from pathlib import Path
import csv
import json
import multiprocessing as mp

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from nomad import shortest_path as sp
from nomad import utils
import macposts

def get_dstIDs(G, inv_nid_map):
    all_dsts = [n for n in G.graph.nodes if str(n).startswith('dst')]
    dstID_list = [inv_nid_map[dst_name] for dst_name in all_dsts]
    return dstID_list

def get_orgIDs(G, inv_nid_map):
    all_orgs = [n for n in G.graph.nodes if str(n).startswith('org')]
    orgID_list = [inv_nid_map[org_name] for org_name in all_orgs]
    return orgID_list


# ---- Destination chunking utility ----
def chunk_list(lst, chunk_size):
    """Yield successive chunk_size-sized chunks from lst."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

# ---- Worker function for TDSP OD Matrix Calculation  ----
def tdsp_dst_chunk_worker(args):

    G, tdsp_folder, BETAS, timestamp_window, org_geo_list, dst_geo_chunk = args

    print(f"Processing chunk with {len(dst_geo_chunk)} destinations...")

    vot = int(BETAS['tt'] * 3600)
    MAX_INTERVAL = G.config['time_factors']['NUM_INTERVALS']
    LINK_COST_FILE = f"td_link_cost_vot{vot}"  # e.g., vot10 → td_link_cost_vot10

    # Build mappings
    _, inv_nid_map, _, _ = utils.build_mappings(G)

    # Define origin and destination IDs
    orgID_list = [inv_nid_map[org_geo] for org_geo in org_geo_list]
    dstID_list = [inv_nid_map[dst_geo] for dst_geo in dst_geo_chunk]

    # Get row counts for TDSP config
    with open(tdsp_folder / LINK_COST_FILE, 'r') as file:
        next(file)
        num_rows_link_file = sum(1 for _ in file)
    with open(tdsp_folder / 'td_node_cost', 'r') as file:
        next(file)
        num_rows_node_file = sum(1 for _ in file)
    sp.write_config(tdsp_folder, 'graph', num_rows_link_file, num_rows_node_file)

    # Initialize TDSP API
    tdsp_api = macposts.tdsp_api()
    tdsp_api.initialize(str(tdsp_folder), MAX_INTERVAL, num_rows_link_file, num_rows_node_file)
    tdsp_api.read_td_cost_txt(str(tdsp_folder), 'td_link_tt', 'td_node_tt', LINK_COST_FILE, 'td_node_cost')

    # Compute OD matrix
    results = []
    for dstID in dstID_list:
        tdsp_api.build_tdsp_tree(dstID)
        for orgID in orgID_list:
            for timestamp in range(*timestamp_window):
                tdsp_arr = tdsp_api.extract_tdsp(orgID, timestamp)
                node_seq = json.dumps(list(tdsp_arr[:, 0]))
                link_seq = json.dumps(list(tdsp_arr[:-1, 1]))
                gtc_total = tdsp_arr[0, 2]
                tt_total = tdsp_arr[0, 3]
                results.append([orgID, dstID, vot, timestamp, node_seq, link_seq, gtc_total, tt_total])

    return results

# ---- Main Execution ----
if __name__ == "__main__":
    # Parameters
    vot = 10
    BETAS = {
        'tt': vot / 3600,
        'rel': 10 / 3600,
        'x': 1,
        'risk': 20,
        'disc': 0
    }

    mode_list = ['pt', 'pt_bs']
    timestamp_window = (180, 361)  # 7:30–8:00am

    chunk_size = 10
    n_proc = mp.cpu_count() - 2 if mp.cpu_count() > 2 else 1

    for m in mode_list:
        print(f"Processing mode: {m}")
        G = utils.load_graph(Path().resolve() / 'graphs' / f'graph_{m}.pkl')
        
        # Prepare TDSP files once
        tdsp_folder = Path().resolve() / 'experiment_output' / f'tdsp_files_{m}'
        sp.prepare_tdsp_files(G, BETAS, tdsp_folder, f"td_link_cost_vot{vot}")

        output_csv = tdsp_folder / f"od_matrix_{m}.csv"

        org_geo_list = [n for n in G.graph.nodes if str(n).startswith('org')]
        dst_geo_list = [n for n in G.graph.nodes if str(n).startswith('dst')]
        dst_chunks = list(chunk_list(dst_geo_list, chunk_size))

        args_list = [(G, tdsp_folder, BETAS, timestamp_window, org_geo_list, dst_chunk) for dst_chunk in dst_chunks]

        with mp.Pool(processes=n_proc) as pool:
            all_results = pool.map(tdsp_dst_chunk_worker, args_list)

        tdsp_data_lists = [item for sublist in all_results for item in sublist]

        # Write to CSV
        header = ['org', 'dst', 'param', 'timestamp', 'node_seq', 'link_seq', 'gtc_tot', 'tt_tot']
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerows(tdsp_data_lists)

        print(f"Finished writing OD matrix for mode: {m}\n")
