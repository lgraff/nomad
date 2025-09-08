""" Module for building a shared scooter graph from a biking graph."""
import networkx as nx

from nomad import utils

def build_graph(G_bike):
    """Builds shared scooter graph from biking graph by copying and renaming nodes and edges."""
    G_sc = G_bike.copy()
    G_sc = utils.rename_nodes(G_sc, 'sc')
    nx.set_node_attributes(G_sc, 'sc', 'nwk_type')
    nx.set_edge_attributes(G_sc, 'sc', 'mode_type')
    nx.set_node_attributes(G_sc, 'sc', 'node_type') # all nodes have same node type (i.e. no special nodes)

    return G_sc