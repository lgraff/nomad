
from nomad import costs

def get_node_cost_df(G_sn, num_intervals):
    '''Add movement-based node costs to node trios.'''
    df_node_costs_sn = costs.nodes.dynamic.assign_node_costs(G_sn, num_intervals)
    return df_node_costs_sn