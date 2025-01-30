
import pandas as pd
import numpy as np

from nomad import conf

def assign_node_costs(G_sn, config):
    '''
    --Create a node cost dict whose keys are of the form (node_from, node_via, node_to) and whose values are the node cost.
    --The node cost represents the cost of moving from one node to another node via an intermidate node.
    e.g. transferring transit routes at stop A would be associated with a negative node cost (the cost of a single ride)
    to implement fee-less transfers.
    --Returns a df with columns: from_node, via_node, to_node, node cost.
    '''

    node_cost_data = []
    for n in G_sn.graph.nodes:
        edges_in = list(G_sn.graph.in_edges(n, keys=True))  # Include edge keys
        edges_out = list(G_sn.graph.out_edges(n, keys=True))  # Include edge keys
        
        for ei in edges_in:
            for eo in edges_out:
                # Unpack edge details
                ei_u, ei_v, ei_key = ei
                eo_u, eo_v, eo_key = eo

                # Access edge attributes
                ei_attrs = G_sn.graph.get_edge_data(ei_u, ei_v, ei_key)
                eo_attrs = G_sn.graph.get_edge_data(eo_u, eo_v, eo_key)
                
                # Account for fee-less PT transfers
                if (
                    n.startswith('ps') and
                    ei_attrs['mode_type'] == 'alight' and
                    eo_v.startswith('ps')
                ):
                    node_cost_data.append((ei_u, n, eo_v, -config['price_params']['board']['fixed']))

                # Uncomment below to prevent two consecutive walking edges
                # if (
                #     ei_attrs['mode_type'] == 'w' and
                #     eo_attrs['mode_type'] == 'w'
                # ):
                #     node_cost_data.append((ei_u, n, eo_v, 10000))
                
    # Construct node cost DataFrame
    df_node_cost = pd.DataFrame(node_cost_data, columns=['node_from', 'node_via', 'node_to', 'cost'])
    return df_node_cost    