import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx 
#import unimodal_graphs.utility_functions as ut

from nomad import utils  # this calls the __init__.py files of utils sub-package

def build_graph(G_bike, study_area_path, station_filepath, lat_colname, long_colname, id_colname):
    # Inputs: bike network graph, filepath of bikeshare depot locations and attributes, lat/long/id/station name/availability
    # column names in the file. 'availability' refers to the number of bike racks at the station
    # Outputs: bikeshare graph inclusive of depot nodes and road intersection nodes. the depot nodes are connected to
    # the intersection nodes by 'connection edges' formed by matching the depot to its nearest neighbor in the road network
    G_bs = G_bike.copy()  
    nx.set_node_attributes(G_bs, 'bs', 'nwk_type')
    nx.set_node_attributes(G_bs, 'bs', 'node_type')
    nx.set_edge_attributes(G_bs, 'bs', 'mode_type')
    G_bs = utils.rename_nodes(G_bs, 'bs')

    # read in bikeshare depot locations and build connection edges
    df_bs = pd.read_csv(station_filepath)
    # generate point geometry from x,y coords, so that the GIS clip function can be used to only include depots within the study region
    df_bs['geometry'] = gpd.points_from_xy(df_bs[long_colname], df_bs[lat_colname], crs="EPSG:4326")
    gdf_bs = gpd.GeoDataFrame(df_bs)  # convert to geo df
    gdf_bs['pos'] = tuple(zip(gdf_bs[long_colname], gdf_bs[lat_colname]))  # add position
    
    # Clip the bs node network
    # study_area_gdf = gpd.read_file(os.path.join(os.path.join(os.getcwd(), 'Data', 'Output_Data'), 'study_area.csv'))
    # # Check 
    # fig,ax = plt.subplots()
    # study_area_gdf.plot(ax=ax)
    # gdf_bs.plot(ax=ax, color='black')
    study_area_gdf = gpd.read_file(study_area_path)
    gdf_bs_clip = gpd.clip(gdf_bs, study_area_gdf).reset_index().drop(columns=['index']).rename(
        columns={id_colname: 'id'})
    # join depot nodes and connection edges to the bikeshare (biking) network
    gdf_bike_nodes = utils.create_gdf_nodes(G_bike)
    G_bs = utils.add_station_cnx_edges(G_bs, gdf_bs_clip, gdf_bike_nodes, 'bsd', 'bs', 'both')   

    for n in G_bs.nodes:
        if n.startswith('bsd'):
            G_bs.nodes[n]['node_type'] = 'bsd'
            G_bs.nodes[n]['nwk_type'] = 'bs'
    
    return G_bs

    
