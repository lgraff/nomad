from pathlib import Path
import geopandas as gpd

from nomad.data import demographics

def process_data(config):
    '''Process demographic data.
       Save two files. 
       1) GeoDataFrame of block groups with an attribute that counts the number of people below the poverty line for past 12 months.
       2) GeoDataFrame of block groups with an attribute that counts the number of opportunity jobs (see paper for description).'''
    
    data_path = config['paths']['data']

    # Get origins: block group centroids, with population data attached. 
    df_census = demographics.get_poverty_data(config)
    study_area_gdf = gpd.read_file(data_path['study_area_out'])
    gdf_pop = demographics.join_df_to_shapefile(df_census, data_path['block_group_shapefile'], study_area_gdf)
    cols_keep = ['GEOID', 'geometry', 'x', 'y', 'total_eligible']
    gdf_pop[cols_keep].to_file(data_path['poverty_pop'], index=False, driver='GeoJSON')

    # Get destinations: block groups centroids, with job data attached.
    df_opp_jobs = demographics.get_opp_jobs(config)
    df_opp_jobs['GEOID20'] = df_opp_jobs['GEOID20'].astype(str)
    df_opp_jobs['GEOID'] = df_opp_jobs['GEOID20'].str[:12] # get GEOID of block group, which is first 12 digits of the GEOID that includes block number
    df_opp_jobs_bg = df_opp_jobs.groupby('GEOID')['opp_jobs_total'].sum().reset_index() # sum opp jobs by block group instead of block b/c the number of blocks is too high
    gdf_jobs = demographics.join_df_to_shapefile(df_opp_jobs_bg, data_path['block_group_shapefile'], study_area_gdf)
    cols_keep = ['GEOID', 'geometry', 'x', 'y', 'opp_jobs_total']
    gdf_jobs[cols_keep].to_file(data_path['opp_jobs'], index=False, driver='GeoJSON')

#TODO: Would be better to save as a single gdf and then add orgs and dsts from this single gdf?