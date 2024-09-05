# libraries
import matplotlib.pyplot as plt
import networkx as nx 
#import unimodal_graphs.utility_functions as ut
from nomad import utils

# TNC graph: 
# Attributes: TT, reliability, risk, price, discomfort
def build_graph(G_drive):
    G_tnc = G_drive.copy()
    nx.set_node_attributes(G_tnc, 't', 'nwk_type')  
    nx.set_node_attributes(G_tnc, 't', 'node_type') # all nodes have same node type (i.e. no special nodes)
    nx.set_edge_attributes(G_tnc, 't', 'mode_type')
    G_tnc = utils.rename_nodes(G_tnc, 't')

    # # plot for visualization
    # node_color = ['black' if n.startswith('t') else 'blue' for n in G_tnc.nodes]
    # edge_color = ['grey'] * len(list(G_tnc.edges))
    # ax = ut.draw_graph(G_tnc, node_color, {'road intersection':'black', 'o/d':'blue'}, edge_color, 'solid')
    return G_tnc


