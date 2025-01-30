'''Execute once to build supernetwork.'''

from pathlib import Path
import os
import sys

import geopandas as gpd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from nomad import supernetwork as sn
from conf import config


def main():
    # Get the origins and destinations (both are block group centroids)
    data_path = config['paths']['data']
    study_area_gdf = gpd.read_file(data_path['study_area_out'])
    study_area_gdf.to_crs(epsg=4269, inplace=True)
    print(Path(__file__).parent.resolve())
    blockgroups_shapefile_path = Path(__file__).parent.resolve() / 'data' / 'demographics' / 'raw' / 'tl_2022_42_bg' / 'tl_2022_42_bg.shp'
    blockgroups = gpd.read_file(blockgroups_shapefile_path, mask=study_area_gdf)  # only include blocks within study area
    blockgroups.to_crs(epsg=4269, inplace=True)
    blockgroups = gpd.clip(blockgroups, study_area_gdf) # clip again
    blockgroups.to_crs(epsg=4326, inplace=True)
    # Get centroid coordinates
    blockgroups['x'] = blockgroups.to_crs(epsg=2272).centroid.to_crs(epsg=4326).x
    blockgroups['y'] = blockgroups.to_crs(epsg=2272).centroid.to_crs(epsg=4326).y

    org_centroids_gdf = blockgroups[['GEOID','x','y']]
    dst_centroids_gdf = blockgroups[['GEOID','x','y']]

    # Build the supernetworks for analysis
    graphs_folder = Path(__file__).parent.resolve() / 'graphs'
    sn.build_supernetwork(config, config['supernetwork']['modes_included'], org_centroids_gdf, dst_centroids_gdf, graphs_folder / 'graph_sn.pkl') # all modes

if __name__ == "__main__":
    main()
