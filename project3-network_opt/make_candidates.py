from pathlib import Path
import geopandas as gpd
import pandas as pd
from shapely import wkt
import matplotlib.pyplot as plt
from conf import config

def make_bikeshare_candidates(bikeshare_mymaps_csv_path):
    bikeshare_mymaps_csv = pd.read_csv(bikeshare_mymaps_csv_path)
    bikeshare_gdf = gpd.GeoDataFrame(bikeshare_mymaps_csv, geometry=bikeshare_mymaps_csv['WKT'].apply(wkt.loads), crs='EPSG:4326').reset_index()[['index','geometry']]
    bikeshare_gdf.rename(columns={'index':'id'}, inplace=True)

    #bikeshare_gdf.plot(ax=ax, color='red')

    # Extract Longitude and Latitude from the geometry column
    bikeshare_gdf['Longitude'] = bikeshare_gdf.geometry.x
    bikeshare_gdf['Latitude'] = bikeshare_gdf.geometry.y

    # Create a new DataFrame with the required columns
    bikeshare_gdf = bikeshare_gdf[['geometry', 'id', 'Longitude', 'Latitude']]
    bikeshare_path = config['paths']['data']['bikeshare_station']
    bikeshare_gdf.to_csv(bikeshare_path, index=False)

def make_microtransit_candidates(zone_candidate_list):
    dissolved_zones = []
    neighborhoods_gdf = gpd.read_file(config['paths']['data']['neighborhoods_in'])

    # Assign zones based on the zone_candidate_list
    for zone in zone_candidate_list:
        # Select neighborhoods that belong to the current zone
        zone_geometry = neighborhoods_gdf[neighborhoods_gdf['hood'].isin(zone)].copy().dissolve()['geometry'][0]
        dissolved_zones.append(zone_geometry)

    # Create a GeoDataFrame from the list of geometries
    zone_gdf = gpd.GeoDataFrame(geometry=dissolved_zones)
    zone_gdf['id'] = range(len(zone_gdf))

    # Save the dissolved GeoDataFrame to a new file
    zone_path = config['paths']['data']['microtransit_zones']
    zone_gdf.to_file(zone_path, driver='GeoJSON')

    # # Plot the dissolved GeoDataFrame
    # fig, ax = plt.subplots(figsize=(6, 6))
    # zone_gdf.plot(ax=ax, column='id', cmap='tab20', legend=True, edgecolor='black')
    # plt.show()