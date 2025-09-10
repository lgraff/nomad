'''Execute to build supernetworks: 1) transit + walking, 2) transit + bikeshare + walking, 3) all modes.'''

from pathlib import Path
import os
import sys

import geopandas as gpd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from nomad import supernetwork as sn
from conf import config

def main():
    """ Main function to build and save supernetworks."""
    # Get the origins and destinations (both are block group centroids)
    data_path = config['paths']['data']
    org_centroids_gdf = gpd.read_file(data_path['block_group_centroids'])
    dst_centroids_gdf = gpd.read_file(data_path['block_group_centroids'])

    # Build and save the supernetworks for analysis
    graphs_folder = Path(__file__).parent.absolute().resolve() / 'graphs'  # where to save the graphs
    #sn.build_supernetwork(config, config['supernetwork']['modes_included'], org_centroids_gdf, dst_centroids_gdf, graphs_folder / 'graph_sn.pkl') # all modes
    sn.build_supernetwork(config, ['pt'], org_centroids_gdf, dst_centroids_gdf, graphs_folder / 'graph_pt.pkl') # public transit
    sn.build_supernetwork(config, ['pt', 'bs'], org_centroids_gdf, dst_centroids_gdf, graphs_folder / 'graph_pt_bs.pkl') # public transit and bikeshare

if __name__ == "__main__":
    main()