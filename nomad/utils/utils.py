from pathlib import Path
import itertools
from sklearn.neighbors import BallTree
import statsmodels as sm
import os
import pandas as pd
import geopandas as gpd
import numpy as np
import networkx as nx
import re
import numpy as np
import re
import geopandas as gpd
import os
import pickle 
import ckanapi

from nomad import costs


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

def create_gdf_nodes(G):    
    """Return GeoDataFrame of nodes from an input nx graph. Include id, lat, and long as attributes."""
    df = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index')
    df[['long', 'lat']] = pd.DataFrame(df['pos'].tolist(), index=df.index)
    df['id'] = df.index
    df['id'] = df['id'].astype('int')
    df.drop(columns='pos', inplace=True)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.long, df.lat))
    gdf.set_crs(crs='epsg:4326', inplace=True)
    return gdf

# GCD: https://medium.com/@petehouston/calculate-distance-of-two-locations-on-earth-using-python-1501b1944d97#:~:text=The%20Great%20Circle%20distance%20formula,that%20the%20Earth%20is%20spherical.
# code source: https://github.com/gboeing/osmnx/blob/main/osmnx/distance.py
def calc_great_circle_dist(row, matrix_ref, earth_radius=6371009):
    """
    Find the great circle distance between an input row (point) and a reference matrix (all other points).
    Args: 
        row: coords of single point.
        matrix_ref: coordinate matrix for all points.
    Returns:
        distance matrix from the row to matrix_ref.
    """
    y1 = np.deg2rad(row[1])  # y is latitude 
    y2 = np.deg2rad(matrix_ref[:,1])
    dy = y2 - y1

    x1 = np.deg2rad(row[0])
    x2 = np.deg2rad(matrix_ref[:,0])
    dx = x2 - x1

    h = np.sin(dy / 2) ** 2 + np.cos(y1) * np.cos(y2) * np.sin(dx / 2) ** 2
    h = np.minimum(1, h)  # protect against floating point errors
    arc = 2 * np.arcsin(np.sqrt(h))

    # return distance in units of earth_radius
    return arc * earth_radius

# returns walking catchment node for the node of interest
# inputs: nodeID of node interest, matrix of gc distances b/w all nodes, and max walking distance
# output: list of nodeIDs of all nodes within the wcz
def wcz(i, dist_matrix, W):
    """
    Build the walking catchment zone relative to a node of interest.
    Args:
        i: nodeID for node of interest.
        dist_matrix: distance between node i and all other nodes in the graph.
        W: maximum walking distnance
    Returns:
        catchment: nodeIDs for all nodes in the walking catchment zone relative to node of interest.
    """
    catchment = np.where(dist_matrix[i] <= W)[0].tolist()
    if i in catchment:
        catchment.remove(i)  # remove self
    return catchment

# inputs: node of interest, matrix of gcd distances b/w all nodes, travel mode of interest, all nodes in the original graph (id+name)
# output: nodeID of the node in the component network of the travel mode of interest that is nearest to the input node of interest
def nn(i, dist_matrix, travel_mode, node_id_map):
    """Find the nearest neighbor of a node of interest in the unimodal network corresponding to the node of interest.
    Args:
        i: nodeID for node of interest.
        dist_matrix: distance between node i and all other nodes in the graph.
        travel_mode: travel mode corresponding to the node of interest.
        node_id_map: dict of the form {nodeID, nodeName}, where nodeName includes the travel mode.
    Returns:
        tuple: (nodeID of nearest neighbor, nodeName of nearest neighbor, distance from node i to its nearest neighbor)
    """
    # Subset the node_id_map for the nodes in the component network of the travel mode of interest
    nid_map_travel_mode = [key for key,val in node_id_map.items() if val.startswith(travel_mode)]   # this is a list of IDs
    # Subset dist matrix for the nodes in the component network of the travel mode of interest
    dist_subset = dist_matrix[:, nid_map_travel_mode]
    # Find the node in the component network of interest that is nearest to the input node of interest
    nn_dist = np.amin(dist_subset[i])
    nn_idx = np.argmin(dist_subset[i])
    # Now map back to the original node ID
    original_nn_id = nid_map_travel_mode[nn_idx]
    original_nn_name = node_id_map[original_nn_id]
    return (original_nn_id, original_nn_name, nn_dist)

def add_station_cnx_edges(G, gdf_station_nodes_orig, gdf_ref_nodes, station_id_prefix, ref_id_prefix, cnx_direction):
    "Returns a newtorkx graph G with connection edges to/from a station (e.g. bikeshare, park&ride, carshare) from/to their nearest neighbor in their corresponding unimodal graph."

    # Get point in reference network nearest to each depot node; record its id and the length of depot to id
    nn = nearest_neighbor(gdf_station_nodes_orig, gdf_ref_nodes, 'lat', 'long', 'id', return_dist=True).reset_index(drop=True)
    nn['id'] = nn.apply(lambda row: station_id_prefix + str(int(row['id'])), axis=1)
    nn['nn_id'] = nn.apply(lambda row: ref_id_prefix + str(int(row['nn_id'])), axis=1)
    
    # add cnx edge travel time 
    # cnx_edge_speed = {
    #     'pv': 5 / 3600 * 1609, # miles /hr / 3600 s/hr * 1609 meter/mile = m/s
    #     'bs': 15 / 3600 * 1000,    # 15 km/hr / (3600 s/hr ) * 1000 m/km = m/s
	#     'zip': 5 / 3600 * 1609, # miles /hr / 3600 s/hr * 1609 meter/mile = m/s
    #     }
    # movement_speed = cnx_edge_speed[cnx_edge_movement_type]

    # Create cnx edges go FROM ref network TO depot network
    nn.set_index(['nn_id', 'id'], inplace=True)  # FROM ref network TO depot network
    cnx_edge_dict = nn.to_dict(orient='index')
    # Create edges FROM depot network TO ref network
    to_depot_edges = list(zip(*list(cnx_edge_dict.keys())))
    from_depot_edges = list(zip(to_depot_edges[1], to_depot_edges[0]))
    from_depot_edges_attr = dict(zip(from_depot_edges, cnx_edge_dict.values()))    
    
    # For node attributes
    gdf_depot_nodes = gdf_station_nodes_orig.copy()
    gdf_depot_nodes['id'] = gdf_depot_nodes.apply(lambda row: station_id_prefix + str(int(row['id'])), axis=1)
    gdf_depot_nodes.set_index(['id'], inplace=True)
    gdf_depot_nodes['node_type'] = station_id_prefix 
    gdf_depot_nodes['nwk_type'] = ref_id_prefix
    node_dict = gdf_depot_nodes.to_dict(orient='index') #nn.drop(columns=['nn_id', 'length_m']).to_dict(orient='index')
    
    # Add edges based on user-specified direction, along with nodes and their attributes
    if cnx_direction == 'to_depot':
        G.add_edges_from(list(cnx_edge_dict.keys()))
        nx.set_edge_attributes(G, cnx_edge_dict)
        nx.set_node_attributes(G, node_dict)    
    elif cnx_direction == 'from_depot':
        G.add_edges_from(list(from_depot_edges))
        nx.set_edge_attributes(G, from_depot_edges_attr)
        nx.set_node_attributes(G, node_dict)
    elif cnx_direction == 'both':
        G.add_edges_from(list(cnx_edge_dict.keys()))  # one way
        nx.set_edge_attributes(G, cnx_edge_dict)
        G.add_edges_from(list(from_depot_edges))  # other way
        nx.set_edge_attributes(G, from_depot_edges_attr)
        nx.set_node_attributes(G, node_dict)

    all_cnx_edge_list = [e for e in G.edges if (
        (e[0].startswith(station_id_prefix) & e[1].startswith(ref_id_prefix)) | 
        (e[0].startswith(ref_id_prefix) & e[1].startswith(station_id_prefix)))]
    
    for e in all_cnx_edge_list:
        G.edges[e]['mode_type'] = ref_id_prefix   # surely a more efficient way to do this
        
    return G

def rename_nodes(G, prefix):
    """Returns nx graph with relabeled nodes."""
    new_nodename = [prefix + re.sub('\D', '', str(i)) for i in G.nodes]
    namemap = dict(zip(G.nodes, new_nodename))
    G = nx.relabel_nodes(G, namemap, True)
    return G

def mode(node_name):
    """Extract and return the travel mode the corresponding to the node name."""
    mode_of_node = re.sub(r'[^a-zA-Z]', '', node_name)
    return mode_of_node

def rename_mode_type(row):
    '''Rename the mode type of bikeshare cnx, carshare (z) cnx, parking, transfer, and od cnx edges.'''
    if ((row['source_node_type'] == 'bs') & (row['target_node_type'] == 'bsd')):
        edge_type = 'bs_cnx'
    elif ((row['source_node_type'] == 'bsd') & (row['target_node_type'] == 'bs')):
        edge_type = 'bs_cnx'
    elif ((row['source_node_type'] == 'z') & (row['target_node_type'] == 'kz')):
        edge_type = 'park'
    elif ((row['source_node_type'] == 'kz') & (row['target_node_type'] == 'z')):
        edge_type = 'park'
    elif ((row['source_node_type'] == 'z') & (row['target_node_type'] == 'zd')):
        edge_type = 'z_cnx'
    elif ((row['source_node_type'] == 'zd') & (row['target_node_type'] == 'z')):
        edge_type = 'z_cnx'
    elif((row['source_node_type'] == 'org') | (row['target_node_type'] == 'dst')):
        edge_type = 'od_cnx'
    elif((row['etype'] == 'transfer') & (row['mode_type'] == 'w')):
        edge_type = 'transfer'
    else: 
        edge_type = row['mode_type']
    return edge_type


def generate_data(G_super, od_cnx=False):
    """
    Generate historical location data for conf.NUM_DAYS (for scooters in particular, could also be used for other modes) in the absence of real data.
    Calculate mean and 95th percentile distance from each fixed node and its nearest scooter based on this simulated data.
    Return dict: {fixed_node_ID: {'length_m': [mean_length_val_meters], '95_length_m': [95th_percentile_length_val_meters]}
    """
    # Define the bounds
    study_area_gdf = gpd.read_file(G_super.config['paths']['data']['study_area_out'])
    bbox = study_area_gdf.bounds.iloc[0]
    xlb, xub, ylb, yub = bbox['minx'], bbox['maxx'], bbox['miny'], bbox['maxy']
    
    if not od_cnx:
        nid_map_fixed = G_super.nid_map_fixed
    else: # only evaluate for org
        nid_map_fixed = {nid_num: nid_name for nid_num, nid_name in G_super.nid_map.items() if nid_name.startswith('org')} 
    
    # Initialize the scooter cost dictionary: key is the fixed node, the value is dict of costs (different cost for the different time intervals)
    all_costs = dict([(n, {}) for n in nid_map_fixed.values()])
    
    # For subsequent visualization purposes
    #fig, axs = plt.subplots(2, 5, sharex = True, sharey = True, figsize=(16,8))
    #plt.suptitle('Example: Scooter observations (red) for time interval 0 shown relative to fixed node bs1038 (blue)')
    # for subsequent plotting purposes 
    #node_coords = np.array([val for key,val in nx.get_node_attributes(G_u, 'pos').items() if key in list(node_id_map_fixed.values())])

    for i in range(1):  # just do this once and reuse the results for all time intervals 
        obs = {}  # obs is a dict, where the key is the day, the value is an array of coordinates representing different observations
        for j in range(G_super.config['scooter_simulation']['NUM_DAYS_OF_DATA']):  # each day
            # generate some random data: data is a coordinate matrix
            # the scooter observations should fit within the bounding box of the neighborhood mask polygon layer
            data = [(round(np.random.uniform(xlb, xub),8), 
                     round(np.random.uniform(ylb, yub),8)) for k in range(int(G_super.config['scooter_simulation']['NUM_OBS']))]  
            obs[j] = np.array(data)  

        # find edge cost
        node_cost_dict = {}
        for n in nid_map_fixed.values():  # for each fixed node (or, for the org/dst when generating for od_cnx)
            all_min_dist = np.empty((1, G_super.config['scooter_simulation']['NUM_DAYS_OF_DATA']))  # initialize the min distance matrix, one entry per day
                       
            for d in range(G_super.config['scooter_simulation']['NUM_DAYS_OF_DATA']):  # how many days of historical scooter data we have                
                all_dist = calc_great_circle_dist(np.array(G_super.graph.nodes[n]['pos']), obs[d])  # dist from the fixed node to all observed scooter locations 
                min_dist = np.min(all_dist)  # choose the scooter with min dist. assume a person always walks to nearest scooter
                all_min_dist[0,d] = min_dist # for the given day, the dist from the fixed node to the nearest scooter is min_dist
                
#                 if (i == 0 and n == 'bs1038'):   # testing
#                     print(all_dist)
                # **********************************
                # JUST FOR VISUALIZATION PURPOSES
#                 # for fixed node bs1038 and time interval 0, visualize the scooter location data for each day
#                 if (i == 0 and n == 'bs1037'):
#                     row = d // 5
#                     col = d if d <=4 else (d-5)
#                     for k in range(len(obs[d][:,0])):
#                         axs[row,col].plot([G.nodes[n]['pos'][0], obs[d][k,0]], [G.nodes[n]['pos'][1], obs[d][k,1]], 
#                                  c='grey', ls='--', marker = 'o', mfc='r', zorder=1)
#                     axs[row,col].scatter(x = G.nodes[n]['pos'][0], y = G.nodes[n]['pos'][1], c='b', s = 200, zorder=2)
#                     axs[row,col].set_title('Day ' + str(d))
#                     axs[row,col].text(-79.93, 40.412, 'closest scooter: ' + str(round(min_dist,3)) + ' miles', ha='center')
# #                 # **********************************
            
            mean_min_dist = np.mean(all_min_dist)  # mean distance from node n to any scooter in past "n_days" days
            p95 = np.percentile(all_min_dist, 95)  # 95th percentile distance from node n to any scooter in past "n_days" days
            node_cost_dict[n] = {'length_m': mean_min_dist, '95_length_m': p95, 'mode_type':'w', 'etype':'transfer'}

        for node, cost_dict in node_cost_dict.items():
            all_costs[node].update(cost_dict) 
    return all_costs


# Pre-processing steps before running djikstra() function. Graph must be in 'index' or numerical form.

def get_cost_subsets(mode_subset, df_edge_cost, df_node_cost):
    '''Create edge and node cost subsets, only inclusive of the edges and nodes corresponding to the selected mode combination.
       Return: a subset of the edge cost df; a subset of the node cost df.'''
    # 1) Edge cost subset
    # Define which edge types are associated with each mode type
    modes_to_edge_type = {'pt': ['board','pt','alight'], 'z': ['z','park'], 'tnc':['t','t_wait'], 'walk':['w'], 'sc': ['sc'], 'bs': ['bs'], 'mt':['mt']}
    edges_include = list(itertools.chain(*[modes_to_edge_type[m] for m in mode_subset])) + ['w']
    df_edge_cost_subset = df_edge_cost[df_edge_cost['mode_type'].isin(edges_include)].copy() # filter the df_edge_cost_subset by the included modes/edges
    df_edge_cost_subset['edge'] = tuple(zip(df_edge_cost_subset.source, df_edge_cost_subset.target))  # add the edge as a tuple
    # 2) Node cost subset
    nodes = set(df_edge_cost_subset.source.unique().tolist() + df_edge_cost_subset.target.unique().tolist())  # node set
    df_node_cost_subset = df_node_cost[((df_node_cost['node_from'].isin(nodes)) & (df_node_cost['node_via'].isin(nodes)) & (df_node_cost['node_to'].isin(nodes)))]

    return (df_edge_cost_subset, df_node_cost_subset) 

def get_node_idx_map(df_edge_cost):
    '''Convert node name to node index (numerical)
       Return a map from node name to node index.'''
    nodes = sorted(set(df_edge_cost.source.unique().tolist() + df_edge_cost.target.unique().tolist()))  # node set
    name2idx = dict(zip(nodes, range(len(nodes)))) # make map from node name to index
    return name2idx    

def get_G_idx(df_edge_cost):
    """Get the 'index' or numerical version of a graph i.e. all nodes are defined by a number, not a string name.
       Return: a new digraph, G_idx, where all nodes are numerical.
    """
    name2idx = get_node_idx_map(df_edge_cost)
    df_edge_cost.loc[:,'source_idx'] = df_edge_cost.loc[:,'source'].map(lambda x: name2idx[x])
    df_edge_cost.loc[:,'target_idx'] = df_edge_cost.loc[:,'target'].map(lambda x: name2idx[x])
    G_idx = nx.from_pandas_edgelist(df_edge_cost, source='source_idx', target='target_idx', edge_attr=['GTC', 'mode_type'], create_using=nx.DiGraph)
    G_idx.remove_nodes_from(list(nx.isolates(G_idx))) # remove isolated nodes (those without neighbors) - do we need this step?
    
    return G_idx

def get_node_cost_idx(df_node_cost, name2idx):
    '''Get node costs in index form.
       Return a dict of the form {(node_from, node_via, node_to): node_cost}.'''
    
    # Map node names to indices
    node_from_idx = list(map(lambda name: name2idx.get(name,-1), df_node_cost['node_from'].tolist()))
    node_via_idx = list(map(lambda name: name2idx.get(name,-1), df_node_cost['node_via'].tolist()))
    node_to_idx = list(map(lambda name: name2idx.get(name,-1), df_node_cost['node_to'].tolist()))
    
    # Create a dictionary from the keys and values
    node_cost_dict_keys = tuple(zip(node_from_idx, node_via_idx, node_to_idx))
    node_cost_dict_vals = df_node_cost['cost']
    
    # Return a dictionary mapping the keys to their corresponding costs
    return dict(zip(node_cost_dict_keys, node_cost_dict_vals))



def load_graph(graph_path):
    with open(graph_path, 'rb') as inp:
        return pickle.load(inp)

def nx_to_df(G):
    '''Convert supernetwork object to pandas df, keyed by edge. Do minimal processing.'''

    # Convert the supernetwork object to a pandas df, keyed by the edge. Columns are edge attributes.
    df_G = nx.to_pandas_edgelist(G.graph)
    # Add the node type of each edge's source and target
    df_G['source_node_type'] = df_G['source'].map(lambda x: re.sub('[^a-zA-Z]+', '', x[:3]))  # only the first 3 characters is a hack to avoid rtL, rtA, etc.
    df_G['target_node_type'] = df_G['target'].map(lambda x: re.sub('[^a-zA-Z]+', '', x[:3]))
    df_G['edge_type'] = df_G.apply(rename_mode_type, axis=1) # get the type of the edge (e.g., bs, park, od_cnx) based on source and target nodes

    # Impute frc and speed_limit data for connection, walking, and microtransit edges. This is necessary to establish predicted crash risk.
    df_G.loc[df_G['edge_type'].isin(['bs_cnx', 'z_cnx', 'park']), 'frc'] = 4        # assume frc=4 for connection edges
    df_G.loc[df_G['edge_type'].isin(['bs_cnx', 'z_cnx', 'park']), 'speed_lim'] = 5  # assume speedlim=5 mph for connection eges
    df_G.loc[df_G['mode_type'] == 'w', 'frc'] = 4                                   # assume frc=4 for walk edges --> local connecting roads
    df_G.loc[df_G['mode_type'] == 'w', 'speed_lim'] = 25                            # assume pedestrians walk on street edges with speedlim=25 mph
    df_G.loc[df_G['mode_type'] == 'mt', ['speed_lim', 'frc']] = [25, 3]   
    df_G['const'] = 1

    # Impute length=0 for board, alight, and t_wait edges. This serves as a placeholder for downstream calculations
    df_G.loc[df_G['mode_type'].isin(['board','alight','t_wait']), 'length_m'] = 0
    
    # Estimate a length for pt traversal edges. This is necessary for risk and discomfort calculations 
    high_speed_idx = (df_G['mode_type'] == 'pt') & ((df_G['length_m'] / df_G['avg_tt_sec'] > 13.4))
    df_G.loc[high_speed_idx, 'length_m'] = df_G.loc[high_speed_idx]['avg_tt_sec'] * 13.4    # 30 mph = 13.4 m/s

    return df_G

def get_cost_array_dict(G_sn):
    """
    Extract dynamic edge costs from a supernetwork and return them as numpy arrays.
    
    Parameters
    ----------
    G_sn : networkx.Graph
        Supernetwork graph.

    Returns
    -------
    cost_arrays : dict[str, np.ndarray]
        Dict of cost arrays for 'tt', 'rel', 'price', 'risk', 'disc'.
    df_edges : pd.DataFrame
        Edge info (source, target, mode_type).
    """
    df_tt, df_rel, df_price, df_risk, df_disc = costs.edges.dynamic.assign_edge_costs(G_sn)

    NUM_INTERVALS = G_sn.config['time_factors']['NUM_INTERVALS']
    interval_columns = [f'i{i}' for i in range(NUM_INTERVALS)]

    cost_array_dict = {
        'tt': df_tt[interval_columns].values.astype('float16'),
        'rel': df_rel[interval_columns].values.astype('float16'),
        'price': df_price[interval_columns].values.astype('float16'),
        'risk': df_risk[interval_columns].values.astype('float16'),
        'disc': df_disc[interval_columns].values.astype('float16'),
    }

    return cost_array_dict

def build_mappings(G_sn):
    """
    Build node and edge ID mappings for the supernetwork.
    
    Parameters
    ----------
    G : networkx graph

    Returns
    -------
    nid_map : dict[int, hashable]
        Node index -> node ID mapping.
    inv_nid_map : dict[hashable, int]
        Node ID -> node index mapping.
    linkID_map : dict[int, tuple]
        Edge index -> (source, target).
    inv_linkID_map : dict[tuple, int]
        (source, target) -> edge index.
    """
    df_edges = nx_to_df(G_sn)
    node_set = sorted(set(df_edges['source']).union(df_edges['target']))
    nid_map = dict(zip(range(len(node_set)), node_set))
    inv_nid_map = {v: k for k, v in nid_map.items()}

    linkID_map = dict(zip(df_edges.index,
                          zip(df_edges['source'], df_edges['target'])))
    inv_linkID_map = {v: k for k, v in linkID_map.items()}

    return nid_map, inv_nid_map, linkID_map, inv_linkID_map


# Utility functions to extract information from TDSP results
def get_named_path(node_seq, nid_map):
    """ Convert a sequence of node IDs to a sequence sequence of named nodes. """
    int_path = [int(n) for n in node_seq]
    path_named = [nid_map[n] for n in int_path]
    return path_named

def get_full_path_modes(named_path):
    """
    Extract and return a string of full mode names used in a path, always including 'walk'.
    
    Parameters:
        named_path (list): A list of node identifiers, where each node starts with a mode abbreviation.
    
    Returns:
        str: Slash-separated string of full mode names used in the path.
    """
    # Define mode abbreviation to full name mapping
    mode_map = {
        't': 'TNC',
        'r': 'public transit',
        'b': 'bikeshare',
        's': 'scooter'
    }

    # Extract mode abbreviations from the named path
    stripped_path = [node[0] for node in named_path]

    # Include modes used more than once
    modes_in_path = {
        mode_map[m] for m in mode_map if stripped_path.count(m) > 1
    }

    # Always include walking
    modes_in_path.add('walk')

    # Return as a slash-separated string
    return ' / '.join(sorted(modes_in_path))


def convert_timestamp_to_hhmm(timestamp, start_hour, start_minute):
    '''Convert timestamp to HH:MM representation.'''
    # Calc minutes since start time
    total_minutes = (timestamp // 6) + (start_hour * 60 + start_minute)
    hours = total_minutes // 60
    minutes = total_minutes % 60
    return f"{hours:01}:{minutes:02}"

def get_path_attributes(link_seq, timestamp, cost_arrays, td_link_tt):
    """ Get the path attributes (time, reliability, price, risk, discomfort) for a given path and departure time. """
    price_total, risk_total, rel_total, tt_total, discomfort_total = 0, 0, 0, 0, 0
    
    t = int(timestamp)

    for l in link_seq:  # the link seq
        # look up how many time intervals it takes to cross the link
        l = int(l) 
        if t >= td_link_tt.shape[1] - 1:   
            t = int(td_link_tt.shape[1] - 1 - 1)  # adjust the time if we reach the last interval
        intervals_cross = td_link_tt[l, t]  

        # # TODO: figure out how to add node costs more efficiently
        # if idx > 0:
        #     node_id = int(node_seq[idx])
        #     in_link_id = int(link_seq[idx-1])
        #     out_link_id = l
        #     #if any(np.equal(nodecost_ids,[node_id, in_link_id, out_link_id]).all(1)):
        #     if [node_id, in_link_id, out_link_id] in nodecost_ids:
        #         nodecost = -2.75 
        #     else:
        #         nodecost = 0
        # else:
        #     nodecost = 0

        price_link, risk_link, rel_link, tt_link, disc_link = cost_arrays['price'][l,t], cost_arrays['risk'][l,t], cost_arrays['rel'][l,t], cost_arrays['tt'][l,t], cost_arrays['disc'][l,t]  # (these arrays do not have a col for linkID)

        #print(nid_map[node_in], nid_map[node_out], round(risk_link/60,3))
        #cost_link = td_link_cost[l,t+1]  # cannot use td_link_cost b/c it only reflects most recently used betas     
        # update time and cost totals
        #print(nid_map[node_in], nid_map[node_out], 'price:', round(price_link,2), 'risk:', round(risk_link,2), 'rel:', round(rel_link/60,2), 'tt:',round(tt_link/60,2), 'discomf:',round(discomfort_link,2))
        price_total += price_link #+ nodecost
        risk_total += risk_link
        rel_total += rel_link
        tt_total += tt_link
        discomfort_total += disc_link
        #cost_total += cost_link
        t = int(t + intervals_cross)  

    return (tt_total, rel_total, price_total, risk_total, discomfort_total)