import concurrent.futures
import functools
import itertools
from itertools import chain, combinations
from pathlib import Path
import time
import csv
import multiprocessing

from nomad import costs
from nomad import shortest_path as sp

def powerset(iterable):
    "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1,len(s)+1))  # modified powerset because we skip the empty set

def get_graph_costs(G_sn, hour, minute):
    '''Get static graph costs at hour:minute departure time.'''
    df_edge_cost_sn = costs.get_edge_cost_df(G_sn, hour=hour, minute=minute)
    df_node_cost_sn = costs.get_node_cost_df(G_sn)
    df_node_cost_sn['cost'] = df_node_cost_sn['cost'].astype('float16') # save some memory (is this necessary?)
    return df_edge_cost_sn, df_node_cost_sn

def process_od(G_idx, node_cost_idx, weight_name, idx2name, df_edge_cost, od_pair):
    source = od_pair[0]
    target = od_pair[1]
    
    shortest_path, total_gtc = sp.run_shortest_path(G_idx, node_cost_idx, weight_name, source, target)
    sp_edge_list = sp.get_named_sp_edges(shortest_path, idx2name)  
    total_travel_time = sp.get_sp_travel_time(df_edge_cost, sp_edge_list)  
    total_expense, total_expense_less_pt = sp.get_sp_expense(df_edge_cost, shortest_path, idx2name, sp_edge_list)  
    result = [idx2name[source], idx2name[target], total_gtc, total_travel_time, total_expense, total_expense_less_pt]
    return(result)

def process_od_batch(G_idx, node_cost_idx, weight_name, idx2name, df_edge_cost, od_pairs):
    results = []
    for od_pair in od_pairs:
        result = process_od(G_idx, node_cost_idx, weight_name, idx2name, df_edge_cost, od_pair)
        results.append(result)
    return results

def modal_cost_matrix(G_sn, hour, minute, mode_subset):  
    '''Find the shortest path, along with its attribute, between each all O-D pairs in the graph for the mode subset provided at the hour:minute departure time.'''
    # Get edge and node cost dfs
    df_edge_cost_sn, df_node_cost_sn = get_graph_costs(G_sn, hour, minute)

    # Get edge and node costs subsets pertaining only to the given mode subset. Get a graph whose nodes are index (integer) values -- necessary for SP calculation
    df_edge_cost_subset, df_node_cost_subset = sp.get_cost_subsets(mode_subset, df_edge_cost_sn, df_node_cost_sn)
    name2idx = sp.get_node_idx_map(df_edge_cost_subset)
    G_idx = sp.get_G_idx(df_edge_cost_subset, name2idx)
    node_cost_idx = sp.get_node_cost_idx(df_node_cost_subset, name2idx)
    idx2name = dict(zip(name2idx.values(), name2idx.keys()))

    # Identify all OD pairs
    org_list = list(set([n for n in df_edge_cost_sn['source'].tolist() if n.startswith('org')]))
    dst_list = list(set([n for n in df_edge_cost_sn['target'].tolist() if n.startswith('dst')]))
    od_pairs_named = list(itertools.product(org_list, dst_list))
    od_pairs_idx = [(name2idx[org], name2idx[dst]) for org, dst in od_pairs_named]

    # Prepare for batch processing
    batch_size = 128
    batches = [od_pairs_idx[i: i+batch_size] for i in range(0, len(od_pairs_idx), batch_size)]

    process_od_batch_partial = functools.partial(process_od_batch, G_idx, node_cost_idx, 'GTC', idx2name, df_edge_cost_subset)

    # Process each batch in parallel
    data = []
    start = time.time()

    with multiprocessing.Pool() as pool:
        results = pool.map(process_od_batch_partial, batches)
    
    # end = time.time()
    # print(end-start)

    data = [item for sublist in results for item in sublist]     # Flatten the results list of lists
    for row in data:
        row.insert(0, ', '.join(mode_subset))

    print(mode_subset, "complete")
    return data

def multimodal_cost_matrix(G_sn, hour, minute, unique_mode_list, cost_matrix_path):
    '''For all combinations of unique_mode_list, find the shortest path between all OD pairs. Write the multimodal cost matrix to a file.'''
    # Partial modal_cost_matrix function
    # modal_cost_matrix_partial = functools.partial(modal_cost_matrix, G_sn, hour, minute)
    
    # Generate all possible modal combinations (31 total) e.g. pt+sc+walk, bs+walk, etc. We will ultimately run sp using every combination
    all_mode_combinations = list(powerset(unique_mode_list))
    
    all_data = []
    for mode_subset in all_mode_combinations:
        data = modal_cost_matrix(G_sn, hour, minute, mode_subset)
        all_data.append(data)

    # Write to csv
    filename = cost_matrix_path           #'modal_travel_costs.csv'
    header = ['mode_subset', 'org', 'dst', 'total_gtc', 'travel_time', 'expense', 'expense_less_pt']
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for line in all_data:
            writer.writerow(line)