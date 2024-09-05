import numpy as np
from nomad import conf
from nomad import shortest_path as sp

'''*Note*: Shortest path = lowest generalized travel cost.'''

def get_named_sp_edges(shortest_path, idx2name):
    '''Return shortest path edge list.'''
    named_sp = [idx2name[n] for n in shortest_path]
    #print(named_sp)
    sp_edge_list = [(named_sp[i], named_sp[i+1]) for i in range(len(shortest_path)-1)]
    return sp_edge_list

def get_sp_travel_time(df_edge_cost_subset, sp_edge_list):
    '''Return shortest path travel time.'''
    # if org[3:] == dst[3:]:
    #     total_travel_time = 0
    # else: 
    total_travel_time = np.nan if len(sp_edge_list) == 0 else df_edge_cost_subset[df_edge_cost_subset['edge'].isin(sp_edge_list)]['avg_tt_sec'].sum()
    return total_travel_time

def get_sp_expense(df_edge_cost_subset, shortest_path, idx2name, sp_edge_list):
    '''Return shortest path monetary expense.'''
    if len(sp_edge_list) == 0:
        total_expense = np.nan
    else:
        total_expense = df_edge_cost_subset[df_edge_cost_subset['edge'].isin(sp_edge_list)]['price'].sum()

    named_sp = [idx2name[n] for n in shortest_path]

    transit_nodes = [n for n in named_sp if n.startswith('rt')]
    if len(transit_nodes) > 0:
        total_expense_less_pt = total_expense - conf.PRICE_PARAMS['board']['fixed']
    else: 
        total_expense_less_pt = total_expense

    return total_expense, total_expense_less_pt

#TODO: include node cost for feeless PT transfers

def process_od(G_idx, node_cost_idx, weight_name, idx2name, df_edge_cost, source, target):
    '''Run shortest path from source to target using G_idx as a input.
       Return source, target, total_gtc, total_travel_time, total_expense, total_expense_less_pt as a list.'''
    shortest_path, total_gtc = sp.run_shortest_path(G_idx, node_cost_idx, weight_name, source, target)
    sp_edge_list = sp.get_named_sp_edges(shortest_path, idx2name)  
    total_travel_time = sp.get_sp_travel_time(df_edge_cost, sp_edge_list)  
    total_expense, total_expense_less_pt = sp.get_sp_expense(df_edge_cost, shortest_path, idx2name, sp_edge_list)  
    return([source, target, total_gtc, total_travel_time, total_expense, total_expense_less_pt])