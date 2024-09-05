
# libraries
import networkx as nx

from nomad import utils

#%% SCOOTER graph:
    # Attributes: TT, reliability ,risk,
def build_graph(G_bike):
    G_sc = G_bike.copy()
    #TODO: filter out streets with speed limit > 25
    G_sc = utils.rename_nodes(G_sc, 'sc')
    nx.set_node_attributes(G_sc, 'sc', 'nwk_type')
    nx.set_edge_attributes(G_sc, 'sc', 'mode_type')
    nx.set_node_attributes(G_sc, 'sc', 'node_type')# all nodes have same node type (i.e. no special nodes)
    return G_sc