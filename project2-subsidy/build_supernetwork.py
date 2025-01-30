'''Execute once to build two supernetworks: 1) transit + walking, 2) transit + bikeshare + scooter + tnc + walking.'''

from pathlib import Path
import os
import sys

import geopandas as gpd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from nomad import supernetwork as sn
from conf import config

# def main():
#     # Build unimodal graphs
#     all_graphs_dict = sn.build_unimodals()

#     # Connect unimodal graphs by transfer edges
#     sn.connect_unimodals(all_graphs_dict, ['t', 'bs', 'pt', 'sc'], conf.G_sn_path)  # Build full supernetwork inclusive of many modes
#     sn.connect_unimodals(all_graphs_dict, ['pt'], conf.G_pt_path)  # Build transit supernetwork for subsequent analysis

#     # Get the origins and destinations (both are block group centroids)
#     org_centroids_gdf = gpd.read_file(conf.subsidy_eligible_pop_path)
#     org_centroids_eligible = org_centroids_gdf[org_centroids_gdf['total_eligible'] > 0].reset_index(drop=True)  # only the origins with eligible pop.

#     dst_centroids_gdf = gpd.read_file(conf.opp_jobs_path)
#     #dst_centroids_jobs_20 = dst_centroids_gdf[dst_centroids_gdf['opp_jobs_total'] > 20].reset_index(drop=True)  # only the destination centroids with > 25 jobs

#     # Add od cnx edges to G_pt and G_sn
#     sn.add_od_cnx(conf.G_pt_path, org_centroids_eligible, dst_centroids_gdf)
#     sn.add_od_cnx(conf.G_sn_path, org_centroids_eligible, dst_centroids_gdf)


def main():
    # Get the origins and destinations (both are block group centroids)
    data_path = config['paths']['data']
    org_centroids_gdf = gpd.read_file(data_path['poverty_pop'])
    org_centroids_eligible = org_centroids_gdf[org_centroids_gdf['total_eligible'] > 0].reset_index(drop=True)  # only the origins with eligible pop.
    dst_centroids_gdf = gpd.read_file(data_path['opp_jobs'])
    dst_centroids_gdf = dst_centroids_gdf[dst_centroids_gdf['opp_jobs_total'] > 0].reset_index(drop=True)       # only the destinations with opp job total > 0

    # Build the supernetworks for analysis
    graphs_folder = Path(__file__).parent.absolute().resolve() / 'graphs'
    sn.build_supernetwork(config, config['supernetwork']['modes_included'], org_centroids_eligible, dst_centroids_gdf, graphs_folder / 'graph_sn.pkl') # all modes

    #sn.build_supernetwork(config, ['pt'], org_centroids_eligible, dst_centroids_gdf,  graphs_folder / 'graph_pt.pkl')  # transit

if __name__ == "__main__":
    main()
