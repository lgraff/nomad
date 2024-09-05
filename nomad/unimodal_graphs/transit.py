
# libraries
import pandas as pd
import geopandas as gpd
import os
import networkx as nx 
import re
import matplotlib.pyplot as plt
from shapely import wkt

from nomad import conf

def build_full_graph(GTFS_filepath, headway_filepath, traversal_time_filepath, streets_processed_path):
    # first we need the coordinates of all the bus stops, read directly from GTFS
    stops_df = pd.read_csv(os.path.join(GTFS_filepath, 'stops.txt'))
#**********************
    # Preprocessing: join bus stops to streets (nearest). 
    # read in stops and processed streets
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
    # add 'ps' in front of the stop_id to define it as a physical stop
    stops_df['stop_id'] = 'ps' + stops_df['stop_id']
    # add position
    stops_df['pos'] = tuple(
        zip(stops_df['stop_lon'], stops_df['stop_lat']))  
    stops_df.set_index('stop_id', inplace=True)
    cols_keep = ['stop_name', 'pos']
    # create dict of the form stop_id: {attr_name: attr_value}, then add to G_pt
    stopnode_dict = stops_df[cols_keep].to_dict(orient='index')
    G_pt = nx.DiGraph()
    G_pt.add_nodes_from(list(stopnode_dict.keys()))
    nx.set_node_attributes(G_pt, stopnode_dict)  
    nx.set_node_attributes(G_pt, 'ps', 'node_type') 
    nx.set_node_attributes(G_pt, 'pt', 'nwk_type') 

    #ax = ut.draw_graph(G_pt, 'blue', {'phsyical stop': 'blue'}, 'grey', 'solid')  # checks out

    # Convert to gdf, but don't clip to study area. Might be the case where the stops on a route go outside the study
    # area but then come back into the study area. If we remove the stops outside the study area, bus route will be
    # inconsistent with reality
    # stops_df['geometry'] = gpd.points_from_xy(
    #     stops_df.stop_lon, stops_df.stop_lat, crs='EPSG:4326')
    # stops_gdf = gpd.GeoDataFrame(stops_df, crs='EPSG:4326').rename(
    #     columns={'stop_lat': 'y', 'stop_lon': 'x'})


    # read headway and traversal time dfs
    # headway: how long between trips for a given route-dir pair
    # traversal time: how long from one stop_id to the next stop_id in the sequence for a given route-dir pair
    df_traversal_time = pd.read_csv(traversal_time_filepath)
    df_headway = pd.read_csv(headway_filepath)

    df_traversal_time[['stop_id','route_id','direction_id']] = df_traversal_time[['stop_id','route_id','direction_id']].astype('str')
    df_headway[['stop_id','route_id','direction_id']] = df_headway[['stop_id','route_id','direction_id']].astype('str')
    
    # define route_node_id as 'rt' + stop_id + route_id + dir_id
    df_headway['route_node_id'] = 'rt' + df_headway['stop_id'] + '_'+ df_headway['route_id']+ '_' + df_headway['direction_id']
    df_traversal_time['route_node_id'] = 'rt' + df_traversal_time['stop_id'] + '_' + df_traversal_time['route_id'] + '_' + df_traversal_time['direction_id'] + '_' + df_traversal_time['stop_sequence'].astype(str)

    # associate a route node to a stopID
    stops_df.reset_index(inplace=True)
    stops_df['stop_id'] = stops_df['stop_id'].str.replace('ps','')
    route_nodes_df = df_traversal_time.merge(
        stops_df, how='left', on='stop_id')[
            ['route_id', 'direction_id', 'stop_id', 'route_node_id', 'stop_sequence', 'pos']]
    route_nodes_df.set_index('route_node_id', inplace=True)
    route_node_dict = route_nodes_df.to_dict(orient='index')
    
    # build route edges
    # df_ss gives us the stops (as a list) associated with a route-dir pair
    df_ss = df_traversal_time.groupby(['route_id', 'direction_id']).agg(
         {'stop_id': list, 'stop_sequence': list}).reset_index()
    df_ss['id_seq'] = df_ss.apply(lambda x: list(zip(x.stop_id, x.stop_sequence)), axis=1)

    # Build route edges
    # list of route-dir pairs
    route_dir_id_list = list(zip(df_ss.route_id, df_ss.direction_id))

    for i, s in enumerate(df_ss.id_seq):   # s is a list of (stop_id, stop_seq) tuples
        #stop_ids = df_ss.stop_id  #list(zip(*s))[0]  # list of sequential stop IDs along the route
        route_nodes = ['rt'+stop_id + '_' + route_dir_id_list[i][0] + '_' +
                       str(route_dir_id_list[i][1]) + '_' + str(stop_seq) for stop_id, stop_seq in s]
        # build route edge of the form: "rt" + stop_id + route_id + dir_id + stop_seq_num
        route_edges = list(
            zip(route_nodes[:len(route_nodes)], route_nodes[1:len(route_nodes)+1]))
        route_edges_attr = []
        # add edge attributes
        for e in route_edges:
            stop_id = e[0].split('rt')[1].split('_')[0]  # some str.split magic to get stop_id
            pred_crashes = stops_streets[stops_streets['stop_id'] == stop_id]['pred_crash'].values[0]
            trav_time_sec = df_traversal_time.loc[df_traversal_time['route_node_id'] == e[1]]['traversal_time_sec'].values[0]   # traversal time from GTFS data
            length_m = df_traversal_time.loc[df_traversal_time['route_node_id'] == e[1]]['length_m'].values[0]   # traversal time from GTFS data
            attr_dict = {'avg_tt_sec': trav_time_sec, 'pred_crash':pred_crashes, 'length_m':length_m}
            route_edges_attr.append((e[0], e[1], attr_dict))  # | is an operator for merging dicts

        # add route edges to the PT graph, along with attriutes
        G_pt.add_edges_from(route_edges_attr)

    nx.set_edge_attributes(G_pt, "pt", 'mode_type')
    nx.set_node_attributes(G_pt, route_node_dict)
    nx.set_node_attributes(G_pt, {r: {'nwk_type':'pt', 'node_type':'rt'} for r in route_node_dict.keys()}) 

        # add boarding and alighting edges. avg_TT_min is the waiting time
        # add time to board and time to alight? just a few seconds, so will be negligible probably
        
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

# Reduce the size of the PT network through a bounding box approach:
# Find the bounding box of the pgh_study_area polygon. Extend this bounding box by x miles. Then clip the PT network by this extended bounding box

def bound_graph(G_pt_full, study_area_path):
    df = pd.DataFrame.from_dict(dict(G_pt_full.nodes), orient="index").reset_index()
    #gdf_pt = gpd.GeoDataFrame(data=df, geometry=df.pos)
    df[['x','y']] = pd.DataFrame(df.pos.tolist())
    gdf_ptnodes = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x,df.y), crs='EPSG:4326')
    #gdf_ptnodes.head(3)
    #bbox_study_area = conf.study_area_gdf['geometry'].bounds.T.to_dict()[0]  # bounding box of neighborhood polygon layer
    #bbox_df = study_area_gdf['geometry'].bounds
    x = 0.25 # miles (buffer the PT network even more than the street network b/c we can imagine the case of a bus route going outside the bounds and then returning inside)
    study_area_gdf = gpd.read_file(study_area_path)
    study_area_buffer = study_area_gdf.to_crs(crs='epsg:32128').buffer(x*1609).to_crs('EPSG:4326')  # 1609 meters/mile
    # check that this worked
    # fig,ax = plt.subplots(figsize=(4,4))
    # study_area_gdf.plot(ax=ax, color='blue')
    # study_area_buffer.plot(ax=ax, color='green', alpha=.4)

    # clip the list of all pt nodes to just those within the new bbox
    pt_graph_clip = gpd.clip(gdf_ptnodes, study_area_buffer)
    pt_graph_clip.set_index('index', inplace=True)
    # go back to the original PT graph, only keep nodes edges that are within the buffered bounding box
    # 1) Nodes
    G_pt = nx.DiGraph()
    node_dict = pt_graph_clip.to_dict(orient='index')
    G_pt.add_nodes_from(node_dict.keys())
    nx.set_node_attributes(G_pt, node_dict)
    # 2) Edges
    df_pt_edges = nx.to_pandas_edgelist(G_pt_full)
    df_edges_keep = df_pt_edges.loc[(df_pt_edges['source'].isin(pt_graph_clip.index.tolist())) & (df_pt_edges['target'].isin(pt_graph_clip.index.tolist()))]
    df_edges_keep.set_index(['source','target'], inplace=True)
    edge_dict = df_edges_keep.to_dict(orient='index')
    G_pt.add_edges_from(edge_dict.keys())
    nx.set_edge_attributes(G_pt, edge_dict)
    # plot for visualization
    # node_color = ['black' if n.startswith('ps') else 'blue' for n in G_pt.nodes]
    # edge_color = ['grey' if (e[0].startswith('ps') and e[1].startswith('rt')) | (e[0].startswith('rt') and e[1].startswith('ps')) else 'black' for e in G_pt.edges]
    # ax = ut.draw_graph(G_pt, node_color, {'physical stop':'black', 'route node':'blue'}, edge_color, 'solid')
    # ax.set_title('Public Transit Network')

    return(G_pt)