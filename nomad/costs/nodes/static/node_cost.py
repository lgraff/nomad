
import pandas as pd
import numpy as np

from nomad import conf

def assign_node_costs(G_sn):
    '''
    --Create a node cost dict whose keys are of the form (node_from, node_via, node_to) and whose values are the node cost.
    --The node cost represents the cost of moving from one node to another node via an intermidate node.
    e.g. transferring transit routes at stop A would be associated with a negative node cost (the cost of a single ride)
    to implement fee-less transfers.
    --Returns a df with columns: from_node, via_node, to_node, node cost.
    '''

    node_cost_data = []
    for n in list(G_sn.graph.nodes):
        edges_in = list(G_sn.graph.in_edges(n))
        edges_out = list(G_sn.graph.out_edges(n))
        for ei in edges_in:
            for eo in edges_out:            
                # account for fee-less PT transfers
                if (n.startswith('ps')) & (G_sn.graph.edges[ei]['mode_type'] == 'alight') & (eo[1].startswith('ps')) :
                    # node from, node via, node to
                    node_cost_data.append((ei[0], n, eo[1], - conf.PRICE_PARAMS['board']['fixed']))
                       #node_cost_data.append([inv_nid_map[n], link_id_map[ei_num], link_id_map[eo_num], 'pt_tx'])
                
                # # cannot go backwards i.e. bs1--bsd25--bs1
                # if (n.startswith('bs')) & (ei[0] == eo[1]):
                #     node_cost_data.append((ei[0], n, eo[1], float(math.inf)))
                #        #node_cost_data.append([inv_nid_map[n], link_id_map[ei_num], link_id_map[eo_num], 'backward'])

                # prevent two consecutive walking edges
                if (G_sn.graph.edges[ei]['mode_type'] == 'w') & (G_sn.graph.edges[eo]['mode_type'] == 'w'):
                    node_cost_data.append((ei[0], n, eo[1], 10000))

     # build node_cost dict
    #node_cost_dict = {(n_from, n_via, n_to): cost for n_from, n_via, n_to, cost in node_cost_data}

    # construct node cost df
    df_node_cost = pd.DataFrame(node_cost_data, columns=['node_from','node_via','node_to', 'cost'])

    return df_node_cost    