import functools
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nomad import costs
from nomad import shortest_path as sp
from nomad import utils

def get_graph_costs(G_sn, BETAS, hour, minute, config):
    '''Get static graph costs for a specific set of betas at a given hour:minute.'''
    df_edge_cost_sn = costs.edges.static.get_edge_cost_df(G_sn, BETAS, hour, minute, config)
    df_node_cost_sn = costs.nodes.static.get_node_cost_df(G_sn, config) # costs.nodes.static.get_node_cost_df(G_sn)
    df_node_cost_sn['cost'] = df_node_cost_sn['cost'].astype('float16') # save some memory
    return (df_edge_cost_sn, df_node_cost_sn)

def calculate_shortest_path(org, dst, df_edge_cost_sn, df_node_cost_sn):
    '''Calculate the shortest path for a single OD pair and mode subset.'''
    #df_edge_cost_subset, df_node_cost_subset = utils.get_cost_subsets(mode_subset, df_edge_cost_sn, df_node_cost_sn)
    G_idx = utils.get_G_idx(df_edge_cost_sn)
    name2idx = utils.get_node_idx_map(df_edge_cost_sn)
    idx2name = dict(zip(name2idx.values(), name2idx.keys()))
    node_cost_idx = utils.get_node_cost_idx(df_node_cost_sn, name2idx)

    run_shortest_path_partial = functools.partial(sp.static.run_shortest_path, G_idx, node_cost_idx, 'GTC')
    get_named_sp_edges_partial = functools.partial(sp.static.get_named_sp_edges, idx2name=idx2name)

    source = name2idx[org]
    target = name2idx[dst]

    shortest_path, total_gtc = run_shortest_path_partial(source, target)
    sp_edge_list = get_named_sp_edges_partial(shortest_path)
    total_travel_time = sp.static.get_sp_travel_time(df_edge_cost_sn, sp_edge_list)
    total_expense = sp.static.get_sp_expense(df_edge_cost_sn, sp_edge_list)
    transit_included = sp.static.transit_included(idx2name, shortest_path)

    return (org, dst, total_travel_time, total_expense, transit_included, total_gtc)


def od_cost_matrix(mode_subset, df_edge_cost_sn, df_node_cost_sn, org_list, dst_list):  
    '''Find the shortest path, along with its attribute, between each all O-D pairs in the graph for the mode subset provided.'''

    # Get edge and node costs subsets pertaining only to the given mode subset. Get a graph whose nodes are index (integer) values -- necessary for SP calculation
    df_edge_cost_subset, df_node_cost_subset = utils.get_cost_subsets(mode_subset, df_edge_cost_sn, df_node_cost_sn)
    G_idx = utils.get_G_idx(df_edge_cost_subset)
    name2idx = utils.get_node_idx_map(df_edge_cost_subset)
    idx2name = dict(zip(name2idx.values(), name2idx.keys()))
    node_cost_idx = utils.get_node_cost_idx(df_node_cost_subset, name2idx)

    # Define partial functions for readability
    run_shortest_path_partial = functools.partial(sp.static.run_shortest_path, G_idx, node_cost_idx, 'GTC')
    get_named_sp_edges_partial = functools.partial(sp.static.get_named_sp_edges, idx2name=idx2name)
    
    results = []

    # For each O-D pair, compute the shortest path and associated attributes
    for org in org_list:
        source = name2idx[org]
        for dst in dst_list:
            target = name2idx[dst]

            shortest_path, total_gtc = run_shortest_path_partial(source, target)
            sp_edge_list = get_named_sp_edges_partial(shortest_path)
            total_travel_time = sp.static.get_sp_travel_time(df_edge_cost_subset, sp_edge_list)  
            total_expense = sp.static.get_sp_expense(df_edge_cost_subset, sp_edge_list)  
            transit_included = sp.static.transit_included(idx2name, shortest_path)

            results.append((org, dst, mode_subset, total_travel_time, total_expense, transit_included, total_gtc))  # store the data
    
    print(mode_subset, "complete")
    return results
