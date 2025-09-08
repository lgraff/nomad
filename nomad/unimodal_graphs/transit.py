
""" Module to build a public transit network graph from GTFS data."""
import pandas as pd
import geopandas as gpd
import os
import networkx as nx 
import re
from shapely import wkt

def build_full_graph(GTFS_filepath, headway_filepath, traversal_time_filepath, streets_processed_path):
    """ Build a full public transit network graph from GTFS data. 
    Includes physical stops, route nodes, route edges, and boarding/alighting edges.
    """
    # Get coordinates of all the bus stops directly from GTFS
    stops_df = pd.read_csv(os.path.join(GTFS_filepath, 'stops.txt'))
    
#**********************
    # Preprocessing: Join bus stops to streets (nearest). This is only necessary so we can get the predicted crash value for each route edge
    df_streets = pd.read_csv(streets_processed_path)
    df_streets['geometry'] = df_streets['geometry'].apply(wkt.loads)
    #df_streets[df_streets['pred_crash'].isna()] # quick check
    # convert to gdf for spatial join
    streets_gdf = gpd.GeoDataFrame(df_streets, geometry=df_streets.geometry, crs='EPSG:32128') 
    stops_gdf = gpd.GeoDataFrame(stops_df, geometry=gpd.points_from_xy(x=stops_df['stop_lon'], y=stops_df['stop_lat']), crs='EPSG:32128')
    # retain copy of streets geom for checking
    streets_gdf['saved_geom'] = streets_gdf.geometry
    # spatial join
    stops_streets = gpd.sjoin_nearest(stops_gdf, streets_gdf, how='left')
#**********************

    # Create physical stop nodes with position attribute
    stops_df['stop_id'] = 'ps' + stops_df['stop_id']  # prefix for physical stops
    stops_df['pos'] = tuple(
        zip(stops_df['stop_lon'], stops_df['stop_lat']))  
    stops_df.set_index('stop_id', inplace=True)
    cols_keep = ['stop_name', 'pos']
    stopnode_dict = stops_df[cols_keep].to_dict(orient='index')
    G_pt = nx.DiGraph()
    G_pt.add_nodes_from(list(stopnode_dict.keys()))
    nx.set_node_attributes(G_pt, stopnode_dict)  
    nx.set_node_attributes(G_pt, 'ps', 'node_type') 
    nx.set_node_attributes(G_pt, 'pt', 'nwk_type') 

    # Read route-level data (headways and traversal times). The headway and traversal dfs were obtained in /data/network/transit_headway.py and /data/network/tranist_traversal.py
    # headway: how long between trips for a given route-dir pair
    # traversal time: how long from one stop_id to the next stop_id in the sequence for a given route-dir pair
    df_traversal_time = pd.read_csv(traversal_time_filepath)
    df_headway = pd.read_csv(headway_filepath)

    df_traversal_time[['stop_id','route_id','direction_id']] = df_traversal_time[['stop_id','route_id','direction_id']].astype('str')
    df_headway[['stop_id','route_id','direction_id']] = df_headway[['stop_id','route_id','direction_id']].astype('str')
    
    # Define route_node_id as 'rt' + stop_id + route_id + dir_id
    df_headway['route_node_id'] = 'rt' + df_headway['stop_id'] + '_'+ df_headway['route_id']+ '_' + df_headway['direction_id']
    df_traversal_time['route_node_id'] = 'rt' + df_traversal_time['stop_id'] + '_' + df_traversal_time['route_id'] + '_' + df_traversal_time['direction_id'] + '_' + df_traversal_time['stop_sequence'].astype(str)

    # Associate a route node to a stopID
    stops_df.reset_index(inplace=True)
    stops_df['stop_id'] = stops_df['stop_id'].str.replace('ps','')
    route_nodes_df = df_traversal_time.merge(
        stops_df, how='left', on='stop_id')[
            ['route_id', 'direction_id', 'stop_id', 'route_node_id', 'stop_sequence', 'pos']]
    route_nodes_df.set_index('route_node_id', inplace=True)
    route_node_dict = route_nodes_df.to_dict(orient='index')
    
    # Build route edges
    df_ss = df_traversal_time.groupby(['route_id', 'direction_id']).agg(
         {'stop_id': list, 'stop_sequence': list}).reset_index()
    df_ss['id_seq'] = df_ss.apply(lambda x: list(zip(x.stop_id, x.stop_sequence)), axis=1)   # df_ss gives us the stops (as a list) associated with a route-dir pair
    route_dir_id_list = list(zip(df_ss.route_id, df_ss.direction_id))  # list of route-dir pairs

    for i, s in enumerate(df_ss.id_seq):   # s is a list of (stop_id, stop_seq) tuples
        #stop_ids = df_ss.stop_id  #list(zip(*s))[0]  # list of sequential stop IDs along the route
        route_nodes = ['rt'+stop_id + '_' + route_dir_id_list[i][0] + '_' +
                       str(route_dir_id_list[i][1]) + '_' + str(stop_seq) for stop_id, stop_seq in s]
        # Build route edge of the form: "rt" + stop_id + route_id + dir_id + stop_seq_num
        route_edges = list(
            zip(route_nodes[:len(route_nodes)], route_nodes[1:len(route_nodes)+1]))
        route_edges_attr = []
        # Assign edge attributes
        for e in route_edges:
            stop_id = e[0].split('rt')[1].split('_')[0]  # some str.split magic to get stop_id
            pred_crashes = stops_streets[stops_streets['stop_id'] == stop_id]['pred_crash'].values[0]
            trav_time_sec = df_traversal_time.loc[df_traversal_time['route_node_id'] == e[1]]['traversal_time_sec'].values[0]   # traversal time from GTFS data
            length_m = df_traversal_time.loc[df_traversal_time['route_node_id'] == e[1]]['length_m'].values[0]   # traversal time from GTFS data
            attr_dict = {'avg_tt_sec': trav_time_sec, 'pred_crash':pred_crashes, 'length_m':length_m}
            route_edges_attr.append((e[0], e[1], attr_dict))  # | is an operator for merging dicts

        # Add route edges to the PT graph, along with attriutes
        G_pt.add_edges_from(route_edges_attr)

    nx.set_edge_attributes(G_pt, "pt", 'mode_type')
    nx.set_node_attributes(G_pt, route_node_dict)
    nx.set_node_attributes(G_pt, {r: {'nwk_type':'pt', 'node_type':'rt'} for r in route_node_dict.keys()}) 

    # Add boarding and alighting edges
    ba_edges = []
    for n in list(G_pt.nodes):
        if G_pt.nodes[n]['node_type'] == 'rt':  #if n.startswith('rt'):   # is a route node
            # Find associated physical stop: # re.sub('\D', '', string) removes letters from string
            split_route_node = n.split('_')
            phys_stop = 'ps' + re.sub('\D', '', (split_route_node[0]))
            # BOARDING edges
            e_board = (phys_stop, n)
            ba_edges.append((e_board[0], e_board[1], {'mode_type':'board'}))
            # ALIGHTING edges
            e_alight = (n, phys_stop)
            ba_edges.append((e_alight[0], e_alight[1], {'mode_type':'alight'}))
            # offset the geometry of the route nodes, for visualization purposes
            G_pt.nodes[n]['pos'] = (G_pt.nodes[n]['pos'][0] + 0.001, G_pt.nodes[n]['pos'][1] + 0.001)

    G_pt.add_edges_from(ba_edges)  # add board/alight edges to the graph

    return G_pt


def bound_graph(G_pt_full, study_area_path):
    """Reduce the transit network to a buffered bounding box around the study area."""
    # --- Step 1: Convert graph nodes to GeoDataFrame ---
    df = pd.DataFrame.from_dict(dict(G_pt_full.nodes), orient="index").reset_index()
    df[['x','y']] = pd.DataFrame(df.pos.tolist())
    gdf_ptnodes = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x,df.y), crs='EPSG:4326')
    
    # --- Step 2: Load and buffer study area ---
    x = 0.25  # buffer distance in miles
    study_area_gdf = gpd.read_file(study_area_path)
    study_area_buffer = study_area_gdf.to_crs(crs='epsg:32128').buffer(x*1609).to_crs('EPSG:4326')  

    # --- Step 3: Clip PT nodes to buffer ---
    pt_graph_clip = gpd.clip(gdf_ptnodes, study_area_buffer)
    pt_graph_clip.set_index('index', inplace=True)

    # --- Step 4: Build new graph with clipped nodes and edges ---
    G_pt = nx.DiGraph()
    node_dict = pt_graph_clip.to_dict(orient='index')
    G_pt.add_nodes_from(node_dict.keys())
    nx.set_node_attributes(G_pt, node_dict)
    
    df_pt_edges = nx.to_pandas_edgelist(G_pt_full)
    df_edges_keep = df_pt_edges.loc[
        (df_pt_edges['source'].isin(pt_graph_clip.index.tolist())) & 
        (df_pt_edges['target'].isin(pt_graph_clip.index.tolist()))
    ]
    df_edges_keep.set_index(['source','target'], inplace=True)
    edge_dict = df_edges_keep.to_dict(orient='index')
    G_pt.add_edges_from(edge_dict.keys())
    nx.set_edge_attributes(G_pt, edge_dict)

    return G_pt 