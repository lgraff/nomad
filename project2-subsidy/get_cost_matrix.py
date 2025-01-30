import concurrent.futures
import functools
import itertools
from pathlib import Path
import pickle
import time
import csv
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nomad import shortest_path as sp

from conf import config

BETAS = {
    'tt': 10/3600,
    'rel': 10/3600,
    'x': 1,
    'risk': 20,
    'disc': 0
        }
hour = 8, 
minute = 0


if __name__ == "__main__":

    graphs_folder = Path(__file__).parent.absolute().resolve() / 'graphs'
    # Read in the supernetwork as an object
    with open(graphs_folder / 'graph_sn.pkl', 'rb') as inp:
        G_sn = pickle.load(inp)

    # Get edge and node costs
    df_edge_cost_sn, df_node_cost_sn = sp.static.get_graph_costs(G_sn, BETAS, hour, minute, config)

    # Get org and dst list
    org_list = sorted(set([n for n in G_sn.graph.nodes if n.startswith('org')]))
    dst_list = sorted(set([n for n in G_sn.graph.nodes if n.startswith('dst')]))

    # Define list of unique modes
    unique_modes = ['pt','tnc','sc','bs']

    # Generate all possible modal combinations e.g. pt+sc+walk, bs+walk, etc. We will ultimately run sp using every combination
    all_mode_combinations = [combo for i in range(1, len(unique_modes) + 1) for combo in itertools.combinations(unique_modes, i)]

    # Process each mode subset in parallel
    data = []
    max_processes = 8
    start = time.time()

    # Use partial to pass additional arguments to process_mode_subset
    od_cost_matrix_partial = functools.partial(sp.static.od_cost_matrix, df_edge_cost_sn=df_edge_cost_sn, df_node_cost_sn=df_node_cost_sn, org_list=org_list, dst_list=dst_list)

    print('Processing ODs by mode')
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_processes - 1) as executor:
        results = list(executor.map(od_cost_matrix_partial, all_mode_combinations))
    
    end = time.time()
    print(f"Execution time: {end - start:.2f} seconds")

    data = [item for sublist in results for item in sublist]     # Flatten the results list of lists

    # Write to csv
    filename = 'modal_travel_costs_2.csv'
    header = ['org', 'dst', 'mode_subset', 'travel_time', 'expense', 'transit_included', 'gtc']
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for line in data:
            writer.writerow(line)
