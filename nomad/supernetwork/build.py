import pickle

import networkx as nx
import geopandas as gpd

from nomad.supernetwork import Supernetwork
from nomad.unimodal_graphs import bikeshare, carshare, personal_bike, personal_vehicle, scooter, TNC, transit

def build_unimodals(config):
    '''Build all unimodal graphs.'''
    data_path = config['paths']['data']
    # Read the driving and biking graphs. In both cases, only keep the first connected component (i.e. most connected nodes)
    G_drive = nx.read_gpickle(data_path['G_drive'])
    first_component = [node_set for node_set in sorted(nx.connected_components(G_drive.to_undirected()), key=len, reverse=True)][0]
    G_drive = G_drive.subgraph(first_component)

    G_bike = nx.read_gpickle(data_path['G_bike'])
    first_component = [node_set for node_set in sorted(nx.connected_components(G_bike.to_undirected()), key=len, reverse=True)][0]
    G_bike = G_bike.subgraph(first_component)

    # Build the unimodal graphs
    G_bs = bikeshare.build_graph(G_bike, data_path['study_area_out'], data_path['bikeshare_station'], 'Latitude', 'Longitude', 'Id')
    G_cs = carshare.build_graph(G_drive, data_path['study_area_out'], data_path['carshare_station'], data_path['parking_out'])
    G_pb = personal_bike.build_graph(G_bike)
    G_pv = personal_vehicle.build_graph(G_drive, data_path['parking_out'])
    G_pt_full = transit.build_full_graph(data_path['GTFS'], data_path['PT_headway_static'], data_path['PT_traversal'], data_path['streets_processed'])
    G_pt = transit.bound_graph(G_pt_full, data_path['study_area_out'])
    G_sc = scooter.build_graph(G_bike)
    G_tnc = TNC.build_graph(G_drive)

    all_graphs_dict = {'t':G_tnc, 'pv':G_pv, 'bs':G_bs, 'pt':G_pt, 'sc':G_sc, 'cs':G_cs, 'pb':G_pb}
    return all_graphs_dict

def connect_unimodals(all_graphs_dict, modes_included, config):
    '''Construct a supernetwork object inclusive of the provided mode list.'''
    G_sn = Supernetwork.from_graphs_dict(all_graphs_dict, modes_included, config)
    
    print('number of edges:', len(G_sn.graph.edges))
    return G_sn

def get_node_idx2geo_dict(node_gdf, node_prefix, geoid_start):
    '''Get mapping from node idx in the node_gdf to its census GEOID.
       Requires that node_gdf have a column called GEOID.
       Example of geoid_start: 
        GEOID = '420034825001'. If geoid start = 5, then its new geoid is 4825001', which in this case is its tract + block group
    '''
    node_idxs = [node_prefix + str(i) for i in node_gdf.index]
    node_gdf['GEOID'] = node_gdf['GEOID'].astype(str) 
    node_geos = [node_prefix + str(i) for i in node_gdf['GEOID'].str[geoid_start:]]
    node_idx2geo_dict = dict(zip(node_idxs, node_geos)) 
    return node_idx2geo_dict

def add_od_cnx(G_sn, org_centroids_gdf, dst_centroids_gdf, config):
    '''Add orgs, dsts, and od connection edges to the graph of the supernetwork object.'''
    org_coords = org_centroids_gdf[['x','y']].to_numpy()  # convert org centroids to numpy array
    dst_coords = dst_centroids_gdf[['x','y']].to_numpy()  # convert dst centroids to numpy array

    G_sn.add_od_nodes(org_coords, dst_coords) 
    G_sn.add_org_cnx(org_coords, config) 
    print('origin cnx built')
    G_sn.add_dst_cnx(dst_coords) 
    print('destination cnx built')
    G_sn.add_direct_od_cnx(org_coords, dst_coords) # add org-dst direct walking edges if within some distance from each other
    G_sn.add_twait_nodes() # add t_wait nodes to the nidmap (i forget why we don't add them as we go; i think because we don't want them accounted for in the coordinate matrix ?)
    

def build_supernetwork(config, modes_included, org_gdf, dst_gdf, output_path):
    """
    Builds a multimodal supernetwork by combining unimodal graphs and adding transfer edges.

    This function constructs a full supernetwork using the specified modes from unimodal graphs.
    It also connects origin and destination centroids to the supernetwork and saves the resulting
    graph to the specified output path.

    Parameters
    ----------
    config : dict
        Configuration dictionary that contains parameters for building the supernetwork, including paths to data and settings for each mode.
    modes_included : list
        List of modes (e.g., 'car', 'bike', 'walk') to be included in the supernetwork.
    org_gdf : GeoDataFrame
        GeoDataFrame containing the origin centroids.
    dst_gdf : GeoDataFrame
        GeoDataFrame containing the destination centroids.
    output_path : str or Path
        Path to save the final supernetwork graph object.

    Returns
    -------
    None
        The function saves the built supernetwork graph to the specified output path.
    """

    # Build unimodal graphs
    all_graphs_dict = build_unimodals(config)

    # Connect unimodal graphs by transfer edges
    G_sn = connect_unimodals(all_graphs_dict, modes_included, config)  # Build full supernetwork inclusive of stated modes
    add_od_cnx(G_sn, org_gdf, dst_gdf, config)

    # Add microtransit edges
    if 'mt' in modes_included:
        zones_gdf = gpd.read_file(config['paths']['data']['microtransit_zones']).to_crs('EPSG:4326')
        G_sn.add_microtransit_edges(zones_gdf)

    # Map org/dst idx to its census GEOID so that we can identify the org/dst node by its GEOID
    org_idx2geo = get_node_idx2geo_dict(org_gdf, 'org', 5) 
    dst_idx2geo = get_node_idx2geo_dict(dst_gdf, 'dst', 5) 
    G_sn.graph = nx.relabel_nodes(G_sn.graph, (org_idx2geo | dst_idx2geo))

    # Save to output_path
    G_sn.save_graph(output_path)