""" Module to build personal bike graph """
import networkx as nx 

from nomad import utils

def build_graph(G_bike):
    """ Build personal bike graph from biking graph"""
    G_pb = G_bike.copy()
    G_pb = utils.rename_nodes(G_pb, 'pb')
    nx.set_node_attributes(G_pb, 'pb', 'nwk_type')
    nx.set_node_attributes(G_pb, 'pb', 'node_type')
    nx.set_edge_attributes(G_pb, 'pb', 'mode_type')
    return G_pb
