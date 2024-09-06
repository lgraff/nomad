# libraries
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx 
from shapely import wkt
#import unimodal_graphs.utility_functions as utils

from nomad import utils

def build_graph(G_drive, study_area_path, carshare_station_path, parking_nodes_path):
    '''summary here'''

    # read data which was obtained from Google MyMaps
    #filepath = os.path.join(cwd,'Data','Input_Data','Zipcar_Depot.csv')
    df_zip = pd.read_csv(carshare_station_path)
    gdf_zip = gpd.GeoDataFrame(data=df_zip, geometry=df_zip['WKT'].apply(wkt.loads), crs='EPSG:4326').reset_index()[['index','geometry']]
    gdf_zip['pos'] = tuple(zip(gdf_zip.geometry.x, gdf_zip.geometry.y)) # add position
    gdf_zip.rename(columns={'index':'id'}, inplace=True)
    study_area_gdf = gpd.read_file(study_area_path)
    gdf_zip_clip = gpd.clip(gdf_zip, study_area_gdf)

    # steps: copy the driving graph. add parking cnx edges. add zip depot cnx edges
    G_cs = G_drive.copy()
    G_cs = utils.rename_nodes(G_cs, 'z')
    nx.set_node_attributes(G_cs, 'z', 'nwk_type')
    nx.set_node_attributes(G_cs, 'z', 'node_type')
    nx.set_edge_attributes(G_cs, 'z', 'mode_type')

    # join parking nodes and connection edges to the carshare network
    gdf_parking_nodes = gpd.read_file(parking_nodes_path)
    gdf_parking_nodes['pos'] = tuple(zip(gdf_parking_nodes['longitude'], gdf_parking_nodes['latitude']))  # add position
    gdf_parking_nodes.insert(0, 'id', gdf_parking_nodes.index)  # add ID to each parking node
    # create df for driving nodes
    gdf_drive_nodes = utils.create_gdf_nodes(G_drive)
    # then connect each parking node to nearest driving intersection node
    G_cs = utils.add_station_cnx_edges(gdf_parking_nodes, gdf_drive_nodes, 'kz', 'z', 'zip', G_cs, 'to_depot')
    # also connect each station node to nearest driving intersection node
    G_cs = utils.add_station_cnx_edges(gdf_zip_clip, gdf_drive_nodes, 'zd', 'z', 'zip', G_cs, 'from_depot')

    # rename mode_type of parking edges
    for e in G_cs.edges:
        if e[1].startswith('k'):
            G_cs.edges[e]['mode_type'] = 'park'

    # # add cost of parking to zipcar: include parking rate + rate associated with zipcar rental
    # park_hours = conf.config_data['Supernetwork']['num_park_hours']
    # for e in G_cs.edges:
    #     if e[1].startswith('kz'):  # if an edge leading into a parking node
    #         parking_rate = G_cs.nodes[e[1]]['float_rate']  # rate per hour
    #         zip_rate = conf.config_data['Price_Params']['zip']['ppmin']*60 # zip rate per hour
    #         G_cs.edges[e]['0_price'] = park_hours * (parking_rate + zip_rate)
            
    # plot for visualization
    # node_color = ['blue' if n.startswith('zd') else 'red' if n.startswith('k') else 'black' for n in G_cs.nodes]
    # edge_color = ['grey' if e[0].startswith('z') and e[1].startswith('z') else 'magenta' for e in G_cs.edges]
    # ax = utils.draw_graph(G_cs, node_color, {'road intersection':'black', 'depot':'blue', 'park':'red'}, edge_color, 'solid')
    # ax.set_title('Personal Vehicle Network')

    return(G_cs)