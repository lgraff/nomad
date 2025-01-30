import geopandas as gpd
import pandas as pd

from conf import config

def get_od_demand():
    # Get clipped block group shapefile
    data_path = config['paths']['data']
    study_area_gdf = gpd.read_file(data_path['study_area_out'])
    study_area_gdf.to_crs(epsg=4269, inplace=True)
    shapefile_path = data_path['block_group_shapefile']
    shape = gpd.read_file(shapefile_path, mask=study_area_gdf)  # only include blocks within study area
    shape.to_crs(epsg=4269, inplace=True)
    shape = gpd.clip(shape, study_area_gdf) # clip again
    shape.to_crs(epsg=4326, inplace=True)
    # Get centroid coordinates
    shape['x'] = shape.to_crs(epsg=2272).centroid.to_crs(epsg=4326).x
    shape['y'] = shape.to_crs(epsg=2272).centroid.to_crs(epsg=4326).y
    shape.plot()

    shape_blockgroups = shape['GEOID'].unique().tolist()

    # Get demand (OD jobs)
    jobs_path = data_path['lodes']
    od_demand = pd.read_csv(jobs_path)
    od_demand['w_geocode_blockgroup'] = od_demand['w_geocode'].astype(str).str[0:12]
    od_demand['h_geocode_blockgroup'] = od_demand['h_geocode'].astype(str).str[0:12]
    od_demand = od_demand[((od_demand['h_geocode_blockgroup'].isin(shape_blockgroups)) & (od_demand['w_geocode_blockgroup'].isin(shape_blockgroups)))]
    od_demand_agg = od_demand.groupby(['h_geocode_blockgroup', 'w_geocode_blockgroup'])['S000'].sum().reset_index()

    # Save to csv
    od_demand_path = data_path['od_demand']
    od_demand_agg.to_csv(od_demand_path, index=False)

def main():
    get_od_demand()

if __name__ == '__main__':
    main()