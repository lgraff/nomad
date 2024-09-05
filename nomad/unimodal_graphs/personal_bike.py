# libraries
import networkx as nx 
#import unimodal_graphs.utility_functions as ut

from nomad import utils

#%% PERSONAL BIKE graph:
# Attributes: TT, reliability, risk, price, discomfort

def build_graph(G_bike):
    G_pb = G_bike.copy()
    G_pb = utils.rename_nodes(G_pb, 'pb')
    nx.set_node_attributes(G_pb, 'pb', 'nwk_type')
    nx.set_node_attributes(G_pb, 'pb', 'node_type')
    nx.set_edge_attributes(G_pb, 'pb', 'mode_type')
    return G_pb