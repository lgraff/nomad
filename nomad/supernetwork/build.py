import pickle

import networkx as nx
import geopandas as gpd

from nomad.supernetwork import Supernetwork
from nomad.unimodal_graphs import bikeshare, carshare, personal_bike, personal_vehicle, scooter, TNC, transit
from nomad import conf

def build_unimodals():
    '''Build all unimodal graphs.'''
    # Read the driving and biking graphs. In both cases, only keep the first connected component (i.e. most connected nodes)
    G_drive = nx.read_gpickle(conf.G_drive_path)
    first_component = [node_set for node_set in sorted(nx.connected_components(G_drive.to_undirected()), key=len, reverse=True)][0]
    G_drive = G_drive.subgraph(first_component)

    G_bike = nx.read_gpickle(conf.G_bike_path)
    first_component = [node_set for node_set in sorted(nx.connected_components(G_bike.to_undirected()), key=len, reverse=True)][0]
    G_bike = G_bike.subgraph(first_component)

    # Build the unimodal graphs
    G_bs = bikeshare.build_graph(G_bike, conf.study_area_outpath, conf.bikeshare_station_path, 'Latitude', 'Longitude', 'Id')
    G_cs = carshare.build_graph(G_drive, conf.study_area_outpath, conf.carshare_station_path, conf.parking_outpath)
    G_pb = personal_bike.build_graph(G_bike)
    G_pv = personal_vehicle.build_graph(G_drive, conf.parking_outpath)
    G_pt_full = transit.build_full_graph(conf.GTFS_path, conf.PT_headway_path_static, conf.PT_traversal_path, conf.streets_processed_path)
    G_pt = transit.bound_graph(G_pt_full, conf.study_area_outpath)
    G_sc = scooter.build_graph(G_bike)
    G_tnc = TNC.build_graph(G_drive)

    all_graphs_dict = {'t':G_tnc, 'pv':G_pv, 'bs':G_bs, 'pt':G_pt, 'sc':G_sc, 'cs':G_cs, 'pb':G_pb}
    return all_graphs_dict

def connect_unimodals(all_graphs_dict, modes_included):
    '''Construct a supernetwork object inclusive of the provided mode list.'''
    G_sn = Supernetwork.from_graphs_dict(all_graphs_dict, modes_included)
    
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

def add_od_cnx(G_sn, org_centroids_gdf, dst_centroids_gdf):
    '''Add orgs, dsts, and od connection edges to the graph of the supernetwork object.'''

    # with open(G_sn_path, 'rb') as inp:
    #     G_sn = pickle.load(inp)

    org_coords = org_centroids_gdf[['x','y']].to_numpy()  # convert org centroids to numpy array
    dst_coords = dst_centroids_gdf[['x','y']].to_numpy()  # convert dst centroids to numpy array
    org_idx2geo = get_node_idx2geo_dict(org_centroids_gdf, 'org', 5) # map org idx to its census GEOID. this way, we can identify the origin by its geoid instead of an unknown idx
    dst_idx2geo = get_node_idx2geo_dict(dst_centroids_gdf, 'dst', 5) # map dst idx to its census GEOID

    G_sn.add_od_nodes(org_coords, dst_coords) 
    G_sn.add_org_cnx(org_coords) 
    print('origin cnx built')
    G_sn.add_dst_cnx(dst_coords) 
    print('destination cnx built')
    G_sn.add_direct_od_cnx(org_coords, dst_coords) # add org-dst direct walking edges if within some distance from each other
    G_sn.add_twait_nodes() # add t_wait nodes to the nidmap (i forget why we don't add them as we go; i think because we don't want them accounted for in the coordinate matrix ?)
    
    #TODO: add a relabel nodes method to supernetwork object
    G_sn.graph = nx.relabel_nodes(G_sn.graph, (org_idx2geo | dst_idx2geo))

def build_supernewtork(modes_included, org_gdf, dst_gdf, output_path):
    # Build unimodal graphs
    all_graphs_dict = build_unimodals()

    # Connect unimodal graphs by transfer edges
    G_sn = connect_unimodals(all_graphs_dict, modes_included)  # Build full supernetwork inclusive of many modes
    add_od_cnx(G_sn, org_gdf, dst_gdf)

    # Save to output_path
    G_sn.save_graph(output_path)


    # graph_sn_path = Path(__file__).parent.absolute().resolve() / 'graph_sn.pkl'
    # graph_pt_path = Path(__file__).parent.absolute().resolve() / 'graph_pt.pkl'
    
    # # Build unimodal graphs
    # all_graphs_dict = build_unimodals()

    # # Connect unimodal graphs by transfer edges
    # G_sn = connect_unimodals(all_graphs_dict, modes_included, graph_sn_path)  # Build full supernetwork inclusive of many modes

    # # Get the origins and destinations (both are block group centroids)
    # org_centroids_gdf = gpd.read_file(conf.subsidy_eligible_pop_path)
    # org_centroids_eligible = org_centroids_gdf[org_centroids_gdf['total_eligible'] > 0].reset_index(drop=True)  # only the origins with eligible pop.

    # dst_centroids_gdf = gpd.read_file(conf.opp_jobs_path)
    # #dst_centroids_jobs_20 = dst_centroids_gdf[dst_centroids_gdf['opp_jobs_total'] > 20].reset_index(drop=True)  # only the destination centroids with > 25 jobs

    # # Add od cnx edges to G_pt and G_sn
    # #sn.add_od_cnx(graph_pt_path, org_centroids_eligible, dst_centroids_gdf)
    # add_od_cnx(G_sn, org_centroids_eligible, dst_centroids_gdf)