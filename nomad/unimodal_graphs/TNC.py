"""Module to build TNC graph from driving graph"""
import networkx as nx 
from nomad import utils

def build_graph(G_drive):
    """Build TNC graph from driving graph"""
    G_tnc = G_drive.copy()
    nx.set_node_attributes(G_tnc, 't', 'nwk_type')  
    nx.set_node_attributes(G_tnc, 't', 'node_type') # all nodes have same node type (i.e. no special nodes)
    nx.set_edge_attributes(G_tnc, 't', 'mode_type')
    G_tnc = utils.rename_nodes(G_tnc, 't')
    return G_tnc


