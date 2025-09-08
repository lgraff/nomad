""" Module to build personal vehicle graph """
import geopandas as gpd
import networkx as nx 

from nomad import utils

def build_graph(G_drive, parking_nodes_path):
    """ Build the personal vehicle graph by adding parking nodes and parking connection edges to the driving graph"""
    G_pv = G_drive.copy()  
    G_pv = utils.rename_nodes(G_pv, 'pv')
    nx.set_node_attributes(G_pv, 'pv', 'nwk_type')
    nx.set_node_attributes(G_pv, 'pv', 'node_type')
    nx.set_edge_attributes(G_pv, 'pv', 'mode_type')

    # Join parking nodes and connection edges to the personal vehicle network. Each parking node is connected to nearest driving intersection node
    gdf_parking_nodes = gpd.read_file(parking_nodes_path)
    gdf_parking_nodes['pos'] = tuple(zip(gdf_parking_nodes['longitude'], gdf_parking_nodes['latitude']))  # add position
    gdf_parking_nodes.insert(0, 'id', gdf_parking_nodes.index)  # add ID to each parking node
    gdf_drive_nodes = utils.create_gdf_nodes(G_drive)
    G_pv = utils.add_station_cnx_edges(G_pv, gdf_parking_nodes, gdf_drive_nodes,'k', 'pv', 'both')
    # Rename mode_type of parking edges
    for e in G_pv.edges:
        if e[1].startswith('k'):
            G_pv.edges[e]['mode_type'] = 'park'

    return G_pv