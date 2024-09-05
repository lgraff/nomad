
# TODO: *** once everything is checked, you can remove this file. it has been replaced by multimodal.utils.utils

# import libraries
from sklearn.neighbors import BallTree
import statsmodels as sm
import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import re
import pickle 
import ckanapi

# https://autogis-site.readthedocs.io/en/latest/notebooks/L3/06_nearest-neighbor-faster.html
def get_nearest(src_points, candidates, k_neighbors=1):
    """Find nearest neighbors for all source points from a set of candidate points"""

    # Create tree from the candidate points
    tree = BallTree(candidates, leaf_size=15, metric='haversine')

    # Find closest points and distances
    distances, indices = tree.query(src_points, k=k_neighbors)

    # Transpose to get distances and indices into arrays
    distances = distances.transpose()
    indices = indices.transpose()

    # Get closest indices and distances (i.e. array at index 0)
    # note: for the second closest points, you would take index 1, etc.
    closest = indices[0]
    closest_dist = distances[0]

    # Return indices and distances
    return (closest, closest_dist)

# cite: 
def nearest_neighbor(left_gdf, right_gdf, right_lat_col, right_lon_col, left_gdf_id, return_dist=False):
    """
    For each point in left_gdf, find closest point in right GeoDataFrame and return them.
    
    NOTICE: Assumes that the input Points are in WGS84 projection (lat/lon).
    """
    
    left_geom_col = left_gdf.geometry.name
    right_geom_col = right_gdf.geometry.name
    
    # Ensure that index in right gdf is formed of sequential numbers
    right = right_gdf.copy().reset_index(drop=True)
    
    # Parse coordinates from points and insert them into a numpy array as RADIANS
    # Notice: should be in Lat/Lon format 
    left_radians = np.array(left_gdf[left_geom_col].apply(lambda geom: (geom.y * np.pi / 180, geom.x * np.pi / 180)).to_list())
    right_radians = np.array(right[right_geom_col].apply(lambda geom: (geom.y * np.pi / 180, geom.x * np.pi / 180)).to_list())
    
    # Find the nearest points
    # -----------------------
    # closest ==> index in right_gdf that corresponds to the closest point
    # dist ==> distance between the nearest neighbors (in meters)
    
    closest, dist = get_nearest(src_points=left_radians, candidates=right_radians)
    
    
    # Return points from right GeoDataFrame that are closest to points in left GeoDataFrame
    closest_points = right.loc[closest]
    
    # Ensure that the index corresponds the one in left_gdf
    closest_points = closest_points.reset_index(drop=True)
    
    # Add distance if requested 
    if return_dist:
        # Convert to meters from radians
        earth_radius = 6371000  # meters
        closest_points['length_m'] = dist * earth_radius
    
    closest_points.rename(columns = {'id': 'nn_id'}, inplace=True)
    closest_points.insert(0, 'id', pd.Series(left_gdf.reset_index(drop=True)[left_gdf_id])) 
    #print(closest_points)
 
    return closest_points.drop(columns=[right_geom_col, right_lat_col, right_lon_col])

def rename_nodes(G, prefix):
    new_nodename = [prefix + re.sub('\D', '', str(i)) for i in G.nodes]
    namemap = dict(zip(G.nodes, new_nodename))
    G = nx.relabel_nodes(G, namemap, True)
    return G

def add_depots_cnx_edges(gdf_depot_nodes_orig, gdf_ref_nodes, depot_id_prefix, 
                         ref_id_prefix, cnx_edge_movement_type, G_ref, 
                         cnx_direction):
    # inputs: 
    # output: 

    # get point in reference network nearest to each depot node; record its id and the length of depot to id
    nn = nearest_neighbor(gdf_depot_nodes_orig, gdf_ref_nodes, 'lat', 'long', 'id', return_dist=True).reset_index(drop=True)
    #cols_keep = depot_cols_keep + ['nn_id', 'length_m']
    #gdf_depot_nodes = pd.concat([gdf_depot_nodes.reset_index(drop=True), nn], axis=1)[cols_keep]
    nn['id'] = nn.apply(lambda row: depot_id_prefix + str(int(row['id'])), axis=1)
    nn['nn_id'] = nn.apply(lambda row: ref_id_prefix + str(int(row['nn_id'])), axis=1)
    # add cnx edge travel time 
    cnx_edge_speed = {
        'pv': 5 / 3600 * 1609, # miles /hr / 3600 s/hr * 1609 meter/mile = m/s
        'bs': 15 / 3600 * 1000,    # 15 km/hr / (3600 s/hr ) * 1000 m/km = m/s
	    'zip': 5 / 3600 * 1609, # miles /hr / 3600 s/hr * 1609 meter/mile = m/s
        }
    movement_speed = cnx_edge_speed[cnx_edge_movement_type]
    #nn['pred_crash'] = 0.05  # the miniumum of pred_crash

    # these cnx edges go FROM ref network TO depot network
    #nn = pd.concat([nn, cnx_attr], axis=1)
    nn.set_index(['nn_id', 'id'], inplace=True)  # FROM ref network TO depot network
    cnx_edge_dict = nn.to_dict(orient='index')
    # also create edges FROM depot network TO ref network
    to_depot_edges = list(zip(*list(cnx_edge_dict.keys())))
    from_depot_edges = list(zip(to_depot_edges[1], to_depot_edges[0]))
    from_depot_edges_attr = dict(zip(from_depot_edges, cnx_edge_dict.values()))    
    
    # for node attributes
    gdf_depot_nodes = gdf_depot_nodes_orig.copy()
    gdf_depot_nodes['id'] = gdf_depot_nodes.apply(lambda row: depot_id_prefix + str(int(row['id'])), axis=1)
    gdf_depot_nodes.set_index(['id'], inplace=True)
    gdf_depot_nodes['node_type'] = depot_id_prefix 
    gdf_depot_nodes['nwk_type'] = ref_id_prefix
    node_dict = gdf_depot_nodes.to_dict(orient='index') #nn.drop(columns=['nn_id', 'length_m']).to_dict(orient='index')
    
    # add edges based on user-specified direction
    if cnx_direction == 'to_depot':
        # add connection edges to the graph. then add nodes and their attributes (depot_cols_keep)
        G_ref.add_edges_from(list(cnx_edge_dict.keys()))
        nx.set_edge_attributes(G_ref, cnx_edge_dict)
        nx.set_node_attributes(G_ref, node_dict)   
        # also add attributes for reliability, risk, price, and discomfort   
    elif cnx_direction == 'from_depot':
        G_ref.add_edges_from(list(from_depot_edges))
        nx.set_edge_attributes(G_ref, from_depot_edges_attr)
        nx.set_node_attributes(G_ref, node_dict)
    elif cnx_direction == 'both':
        G_ref.add_edges_from(list(cnx_edge_dict.keys()))  # one way
        nx.set_edge_attributes(G_ref, cnx_edge_dict)
        G_ref.add_edges_from(list(from_depot_edges))  # other way
        nx.set_edge_attributes(G_ref, from_depot_edges_attr)
        nx.set_node_attributes(G_ref, node_dict)

    all_cnx_edge_list = [e for e in G_ref.edges if (
        (e[0].startswith(depot_id_prefix) & e[1].startswith(ref_id_prefix)) | 
        (e[0].startswith(ref_id_prefix) & e[1].startswith(depot_id_prefix)))]
    
    for e in all_cnx_edge_list:
        G_ref.edges[e]['mode_type'] = ref_id_prefix   # surely a more efficient way to do this
        
    return G_ref

#%% create df_node files for both driving and biking
def create_gdf_nodes(G):    
    df = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index')
    df[['long', 'lat']] = pd.DataFrame(df['pos'].tolist(), index=df.index)
    df['id'] = df.index
    df['id'] = df['id'].astype('int')
    df.drop(columns='pos', inplace=True)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.long, df.lat))
    gdf.set_crs(crs='epsg:4326', inplace=True)
    return gdf