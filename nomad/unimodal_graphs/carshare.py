""" Module to build carshare graph """
import geopandas as gpd
import pandas as pd
import networkx as nx 
from shapely import wkt

from nomad import utils

def build_graph(G_drive, study_area_path, carshare_station_path, parking_nodes_path):
    """ Build carshare graph from driving graph and carshare depot (station) locations and parking nodes"""

    # Read data which was obtained from Google MyMaps
    df_zip = pd.read_csv(carshare_station_path)
    gdf_zip = gpd.GeoDataFrame(data=df_zip, geometry=df_zip['WKT'].apply(wkt.loads), crs='EPSG:4326').reset_index()[['index','geometry']]
    gdf_zip['pos'] = tuple(zip(gdf_zip.geometry.x, gdf_zip.geometry.y)) # add position
    gdf_zip.rename(columns={'index':'id'}, inplace=True)
    study_area_gdf = gpd.read_file(study_area_path)
    gdf_zip_clip = gpd.clip(gdf_zip, study_area_gdf)

    # Copy the driving graph and add edge/node attributes
    G_cs = G_drive.copy()
    G_cs = utils.rename_nodes(G_cs, 'z')
    nx.set_node_attributes(G_cs, 'z', 'nwk_type')
    nx.set_node_attributes(G_cs, 'z', 'node_type')
    nx.set_edge_attributes(G_cs, 'z', 'mode_type')

    # Add parking nodes and parking cnx edge (connect each parking node to nearest driving intersection node)
    gdf_parking_nodes = gpd.read_file(parking_nodes_path)
    gdf_parking_nodes['pos'] = tuple(zip(gdf_parking_nodes['longitude'], gdf_parking_nodes['latitude']))  # add position
    gdf_parking_nodes.insert(0, 'id', gdf_parking_nodes.index)  # add ID to each parking node
    gdf_drive_nodes = utils.create_gdf_nodes(G_drive)
    G_cs = utils.add_station_cnx_edges(G_cs, gdf_parking_nodes, gdf_drive_nodes, 'kz', 'z', 'to_depot')

    # Connect each carshare station node to nearest driving intersection node with a connection edge
    G_cs = utils.add_station_cnx_edges(G_cs, gdf_zip_clip, gdf_drive_nodes, 'zd', 'z', 'from_depot')

    # Rename mode_type of parking edges
    for e in G_cs.edges:
        if e[1].startswith('k'):
            G_cs.edges[e]['mode_type'] = 'park'

    return(G_cs)