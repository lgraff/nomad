
from nomad import costs

def get_node_cost_df(G_sn):
    '''Add movement-based node costs to node trios.'''
    df_node_costs_sn = costs.nodes.static.assign_node_costs(G_sn)
    return df_node_costs_sn