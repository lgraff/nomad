from pathlib import Path

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

from nomad import conf


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

#% create df_node files for both driving and biking
def create_gdf_nodes(G):    
    df = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index')
    df[['long', 'lat']] = pd.DataFrame(df['pos'].tolist(), index=df.index)
    df['id'] = df.index
    df['id'] = df['id'].astype('int')
    df.drop(columns='pos', inplace=True)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.long, df.lat))
    gdf.set_crs(crs='epsg:4326', inplace=True)
    return gdf

# find the great circle distance between an input row (point) and a reference matrix (all other points)
# GCD: https://medium.com/@petehouston/calculate-distance-of-two-locations-on-earth-using-python-1501b1944d97#:~:text=The%20Great%20Circle%20distance%20formula,that%20the%20Earth%20is%20spherical.
# inputs: row (coords of single point), matrix_ref (coordinate matrix for all points)
# code source: https://github.com/gboeing/osmnx/blob/main/osmnx/distance.py
def calc_great_circle_dist(row, matrix_ref, earth_radius=6371009):
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

# returns the travel mode the corresponds to the node
def mode(node_name):
    mode_of_node = re.sub(r'[^a-zA-Z]', '', node_name)
    return mode_of_node


# inputs: graph, num of days of historical data, num of time intervals, num of scooter obs per time-interval day lower bound
# and upper bound, lower and upper bound of potential (x,y) coordinate of scooter, node id map, some cost parameters
# output: dict of dicts
def gen_data(G_super, od_cnx=False):
    # define the bounds

    study_area_gdf = gpd.read_file(conf.study_area_outpath)
    bbox = study_area_gdf.bounds.iloc[0]
    xlb, xub, ylb, yub = bbox['minx'], bbox['maxx'], bbox['miny'], bbox['maxy']
    
    if not od_cnx:
        nid_map_fixed = G_super.nid_map_fixed
    else: # only evaluate for org
        nid_map_fixed = {nid_num: nid_name for nid_num, nid_name in G_super.nid_map.items() if nid_name.startswith('org')} 
    
    # initialize the scooter cost dictionary: key is the fixed node, the value is dict of costs (different cost for the different time intervals)
    all_costs = dict([(n, {}) for n in nid_map_fixed.values()])
    
    # For subsequent visualization purposes
    #fig, axs = plt.subplots(2, 5, sharex = True, sharey = True, figsize=(16,8))
    #plt.suptitle('Example: Scooter observations (red) for time interval 0 shown relative to fixed node bs1038 (blue)')
    # for subsequent plotting purposes 
    #node_coords = np.array([val for key,val in nx.get_node_attributes(G_u, 'pos').items() if key in list(node_id_map_fixed.values())])

    for i in range(1):  # i think we will just do this once and reuse the results  #(n_intervals):  # each time interval 
        obs = {}  # obs is a dict, where the key is the day, the value is an array of coordinates representing different observations
        for j in range(conf.NUM_DAYS_OF_DATA):  # each day
            # generate some random data: data is a coordinate matrix
            # the scooter observations should fit within the bounding box of the neighborhood mask polygon layer
            data = [(round(np.random.uniform(xlb, xub),8), 
                     round(np.random.uniform(ylb, yub),8)) for k in range(int(conf.NUM_OBS))]  
            obs[j] = np.array(data)  

        # find edge cost
        node_cost_dict = {}
        for n in nid_map_fixed.values():  # for each fixed node (or, for the org/dst when generating for od_cnx)
            all_min_dist = np.empty((1, conf.NUM_DAYS_OF_DATA))  # initialize the min distance matrix, one entry per day
                       
            for d in range(conf.NUM_DAYS_OF_DATA):  # how many days of historical scooter data we have 
                # TO DO: check if transfer is allowed between ut.mode(n) and 'sc'
                
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

# returns walking catchment node for the node of interest
# inputs: nodeID of node interest, matrix of gc distances b/w all nodes, and max walking distance
# output: list of nodeIDs of all nodes within the wcz
def wcz(i, dist_matrix, W):
    #print(dist_matrix.shape)
    catchment = np.where(dist_matrix[i] <= W)[0].tolist()
    if i in catchment:
        catchment.remove(i)  # remove self
    return catchment

# inputs: node of interest, matrix of gcd distances b/w all nodes, travel mode of interest, all nodes in the original graph (id+name)
# output: nodeID of the node in the component network of the travel mode of interest that is nearest to the input node of interest
def nn(i, dist_matrix, travel_mode, node_id_map):
    # subset the node_id_map for the nodes in the component network of the travel mode of interest
    nid_map_travel_mode = [key for key,val in node_id_map.items() if val.startswith(travel_mode)]   # this is a list of IDs
    # subset dist matrix for the nodes in the component network of the travel mode of interest
    dist_subset = dist_matrix[:, nid_map_travel_mode]
    # find the node in the component network of interest that is nearest to the input node of interest
    #print(dist_subset)
    nn_dist = np.amin(dist_subset[i])
    nn_idx = np.argmin(dist_subset[i])
    # now map back to the original node ID
    original_nn_id = nid_map_travel_mode[nn_idx]
    original_nn_name = node_id_map[original_nn_id]
    return (original_nn_id, original_nn_name, nn_dist)

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


#********************************************************

import concurrent.futures
import functools
import itertools
from pathlib import Path
import time
import csv

from nomad import costs
from nomad import conf
from nomad import shortest_path as sp

def get_graph_costs(hour, minute):
    df_edge_cost_sn = costs.get_edge_cost_df(conf.G_sn_path, hour=hour, minute=minute)
    df_node_cost_sn = costs.get_node_cost_df(conf.G_sn_path)
    df_node_cost_sn['cost'] = df_node_cost_sn['cost'].astype('float16') # save some memory (is this necessary?)
    return df_edge_cost_sn, df_node_cost_sn

def modal_cost_matrix(mode_subset, df_edge_cost_sn, df_node_cost_sn):  
    '''Find the shortest path, along with its attribute, between each all O-D pairs in the graph for the mode subset provided.'''

    # # Lazy loading and caching of dfs and org list, not sure if this is most memory-efficient
    # if not hasattr(process_mode_subset, "_cached_data"):
    #     print("Loading data...")
    #     process_mode_subset._cached_data = {
    #         "df_edge_cost_sn": costs.get_edge_cost_df(conf.G_sn_path, hour=8, minute=0),
    #         "df_node_cost_sn": costs.get_node_cost_df(conf.G_sn_path)
    #         #"org_list": pd.read_csv(Path().resolve() / 'multimodal' / 'subsidy' / 'selected_origins.csv')['org'].tolist()
    #     }

    # df_edge_cost_sn = process_mode_subset._cached_data["df_edge_cost_sn"]
    # df_node_cost_sn = process_mode_subset._cached_data["df_node_cost_sn"]
    # df_node_cost_sn['cost'] = df_node_cost_sn['cost'].astype('float16') # save some memory

    # #org_list = process_mode_subset._cached_data["org_list"]     #pd.read_csv(Path().resolve() / 'multimodal' / 'subsidy' / 'selected_origins.csv')['org'].tolist()
    
    # Identify orgs and dsts
    org_list = list(set([n for n in df_edge_cost_sn['source'].tolist() if n.startswith('org')]))
    dst_list = list(set([n for n in df_edge_cost_sn['target'].tolist() if n.startswith('dst')]))

    # Get edge and node costs subsets pertaining only to the given mode subset. Get a graph whose nodes are index (integer) values -- necessary for SP calculation
    df_edge_cost_subset, df_node_cost_subset = sp.get_cost_subsets(mode_subset, df_edge_cost_sn, df_node_cost_sn)
    name2idx = sp.get_node_idx_map(df_edge_cost_subset)
    G_idx = sp.get_G_idx(df_edge_cost_subset, name2idx)
    node_cost_idx = sp.get_node_cost_idx(df_node_cost_subset, name2idx)
    idx2name = dict(zip(name2idx.values(), name2idx.keys()))

    # Define partial functions for readability
    run_shortest_path_partial = functools.partial(sp.run_shortest_path, G_idx, node_cost_idx, 'GTC')
    get_named_sp_edges_partial = functools.partial(sp.get_named_sp_edges, idx2name=idx2name)
    
    results = []

    # For each O-D pair, compute the shortest path and associated attributes
    for org in org_list:
        source = name2idx[org]
        for dst in dst_list:
            target = name2idx[dst]

            shortest_path, total_gtc = run_shortest_path_partial(source, target)
            sp_edge_list = get_named_sp_edges_partial(shortest_path)
            total_travel_time = sp.get_sp_travel_time(df_edge_cost_subset, sp_edge_list)  
            total_expense, total_expense_less_pt = sp.get_sp_expense(df_edge_cost_subset, shortest_path, idx2name, sp_edge_list)  

            results.append((org, dst, mode_subset, total_travel_time, total_expense, total_expense_less_pt, total_gtc))  # store the data
    
    print(mode_subset, "complete")
    return results

def multimodal_cost_matrix(unique_mode_list, cost_matrix_path):
    '''For all combinations of unique_mode_list, find the shortest path between all OD pairs. Write the multimodal cost matrix to a file.'''
    # # Define list of unqiue modes
    # unique_mode_list = ['pt','tnc','sc','bs']
    
    # Define which edge types are associated with each mode type - actually moved this to conf.py
    
    # modes_to_edge_type = {'pt': ['board','pt','alight'], 'z': ['z','park'], 'tnc':['t','t_wait'], 'walk':['w'], 'sc': ['sc'], 'bs': ['bs']}
    # Generate all possible modal combinations (31 total) e.g. pt+sc+walk, bs+walk, etc. We will ultimately run sp using every combination
    all_mode_combinations = []
    for i in range(1,len(unique_mode_list)+1):
        mode_combinations = list(itertools.combinations(unique_mode_list, i))
        for mc in mode_combinations:
            all_mode_combinations.append(mc)

    # Process each mode subset in parallel
    data = []
    max_processes = 8
    start = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_processes - 1) as executor:
        results = list(executor.map(modal_cost_matrix, all_mode_combinations))
    
    end = time.time()
    print(end-start)

    data = [item for sublist in results for item in sublist]     # Flatten the results list of lists

    # Write to csv
    filename = cost_matrix_path           #'modal_travel_costs.csv'
    header = ['org', 'dst', 'mode_subset', 'travel_time', 'expense', 'expense_less_pt', 'gtc']
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for line in data:
            writer.writerow(line)