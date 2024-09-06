'''Execute once to build supernetworks: 1) transit + walking, 2) transit + bikeshare + walking, 3) all modes.'''

from pathlib import Path
import os
import sys

import geopandas as gpd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from nomad import conf
from nomad import supernetwork as sn

def main():
    # Get the origins and destinations (both are block group centroids)
    org_centroids_gdf = gpd.read_file(conf.subsidy_eligible_pop_path)
    #org_centroids_eligible = org_centroids_gdf[org_centroids_gdf['total_eligible'] > 0].reset_index(drop=True)  # only the origins with eligible pop.
    dst_centroids_gdf = gpd.read_file(conf.opp_jobs_path)

    # Build the supernetworks for analysis
    graphs_folder = Path(__file__).parent.absolute().resolve() / 'graphs'
    sn.build_supernewtork(['t','bs','pt','sc'], org_centroids_gdf, dst_centroids_gdf, graphs_folder / 'graph_sn.pkl')
    sn.build_supernewtork(['bs','pt'], org_centroids_gdf, dst_centroids_gdf,  graphs_folder / 'graph_pt_bs.pkl')
    sn.build_supernewtork(['pt'], org_centroids_gdf, dst_centroids_gdf,  graphs_folder / 'graph_pt.pkl')

if __name__ == "__main__":
    main()