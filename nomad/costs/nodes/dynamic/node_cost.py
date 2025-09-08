
import pandas as pd
import numpy as np

def assign_node_costs(G_sn):
    '''
    --Create a node cost dict whose keys are of the form (node_from, node_via, node_to) and whose values are the node cost.
    --The node cost represents the cost of moving from one node to another node via an intermidate node.
    e.g. transferring transit routes at stop A would be associated with a negative node cost (the cost of a single ride)
    to implement fee-less transfers.
    --Returns a df with columns: from_node, via_node, to_node, node cost.
    '''
    config = G_sn.config
    
    node_cost_data = []
    for n in list(G_sn.graph.nodes):
        for ei in G_sn.graph.in_edges(n, keys=True):
            for eo in G_sn.graph.out_edges(n, keys=True):          
                # access edge data
                edge_in_data = G_sn.graph.edges[ei]
                edge_out_data = G_sn.graph.edges[eo]
                
                # account for fee-less PT transfers
                if (n.startswith('ps')) & (edge_in_data.get("mode_type") == "alight") & (eo[1].startswith('ps')) :
                    # node from, node via, node to
                    node_cost_data.append((ei[0], n, eo[1], - config['price_params']['board']['fixed']))
                
                # # cannot go org-ps-ps
                # if (n.startswith('ps')) & (ei[0] in ['org5623003', 'org5648002']) & (eo[1].startswith('ps')) :
                #     # node from, node via, node to
                #     node_cost_data.append((ei[0], n, eo[1], 100))
                # # # cannot go ps-ps-dst
                # if (n.startswith('ps')) & (ei[0].startswith('ps')) & (eo[1] in ['dst9822001']) :
                #     # node from, node via, node to
                #     node_cost_data.append((ei[0], n, eo[1], 100))

                # # prevent two consecutive walking edges
                # if (G_sn.graph.edges[ei]['mode_type'] == 'w') & (G_sn.graph.edges[eo]['mode_type'] == 'w'):
                #     node_cost_data.append((ei[0], n, eo[1], 10000))

    # construct node cost df
    df_node_cost = pd.DataFrame(node_cost_data, columns=['node_from','node_via','node_to', 'cost'])

    nodecost_arr = np.repeat(df_node_cost['cost'].values.astype(np.float16).reshape(-1,1), repeats=config['time_factors']['NUM_INTERVALS'], axis=1)
    df_intervals = pd.DataFrame(nodecost_arr, columns=[f'i{i}' for i in range(config['time_factors']['NUM_INTERVALS'])], dtype='float16')
    df_edge_info = df_node_cost[['node_from','node_via','node_to']]
    df_node_cost_dynamic = pd.concat([df_edge_info, df_intervals], axis=1)

    return df_node_cost_dynamic    