# libraries
import geopandas as gpd
import pandas as pd
import os
import matplotlib.pyplot as plt
import networkx as nx 
#import unimodal_graphs.utility_functions as utils

from nomad import utils

def build_graph(G_drive, parking_nodes_path):
    G_pv = G_drive.copy()  # the personal vehicle graph is a copy of the driving graph
    G_pv = utils.rename_nodes(G_pv, 'pv')
    nx.set_node_attributes(G_pv, 'pv', 'nwk_type')
    nx.set_node_attributes(G_pv, 'pv', 'node_type')
    nx.set_edge_attributes(G_pv, 'pv', 'mode_type')

    # join parking nodes and connection edges to the personal vehicle network
    gdf_parking_nodes = gpd.read_file(parking_nodes_path)
    gdf_parking_nodes['pos'] = tuple(zip(gdf_parking_nodes['longitude'], gdf_parking_nodes['latitude']))  # add position
    gdf_parking_nodes.insert(0, 'id', gdf_parking_nodes.index)  # add ID to each parking node
    # create df for driving nodes
    gdf_drive_nodes = utils.create_gdf_nodes(G_drive)
    # then connect each parking node to nearest driving intersection node
    G_pv = utils.add_station_cnx_edges(gdf_parking_nodes, gdf_drive_nodes, # ['ID','pos','zone','float_rate'],
                                   'k', 'pv', 'pv', G_pv, 'to_depot')
    # rename mode_type of parking edges
    for e in G_pv.edges:
        if e[1].startswith('k'):
            G_pv.edges[e]['mode_type'] = 'park'

    # #plot for visualization
    # node_color = ['black' if n.startswith('pv') else 'blue' for n in G_pv.nodes]
    # edge_color = ['grey' if e[0].startswith('pv') and e[1].startswith('pv') else 'magenta' for e in G_pv.edges]
    # ax = utils.draw_graph(G_pv, node_color, {'road intersection':'black', 'pnr':'blue'}, edge_color, 'solid')
    # ax.set_title('Personal Vehicle Network')

    return G_pv