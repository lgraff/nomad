
import geopandas as gpd

def get_od_centroids(config):    
    """ Get the centroids of the polygons that represent the origins and destinations."""
    data_path = config['paths']['data']

    study_area_gdf = gpd.read_file(data_path['study_area_out'])
    od_polygon_path = data_path['block_group_shapefile']  
    
    study_area_gdf.to_crs(epsg=4269, inplace=True)
    od_polygons = gpd.read_file(od_polygon_path, mask=study_area_gdf)  # only include blocks within study area
    od_polygons = gpd.clip(od_polygons, study_area_gdf).to_crs(epsg=4326)
    # Get centroid coordinates
    od_polygons['x'] = od_polygons.to_crs(epsg=2272).centroid.to_crs(epsg=4326).x
    od_polygons['y'] = od_polygons.to_crs(epsg=2272).centroid.to_crs(epsg=4326).y

    od_polygons.to_file(data_path['block_group_centroids'], index=False, driver='GeoJSON')

    return od_polygons