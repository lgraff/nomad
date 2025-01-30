from pathlib import Path

from nomad.data import demographics
from nomad import conf

def process_data():
    '''Process demographic data.
       Save two files. 
       1) GeoDataFrame of block groups with an attribute that counts the number of people below the poverty line for past 12 months.
       2) GeoDataFrame of block groups with an attribute that counts the number of opportunity jobs (see paper for description).'''
    data_path = Path().resolve() / 'nomad' / 'data' / 'demographics' / 'raw'

    # Get origins: block group centroids, with population data attached. Save at conf.eligible_pop_path
    df_census = demographics.get_eligible_population(data_path / 'ACS_5Y_2022_B17017_transp.csv')
    bg_shapefile_path = data_path / 'tl_2022_42_bg' / 'tl_2022_42_bg.shp'
    gdf_pop = demographics.join_df_to_shapefile(df_census, bg_shapefile_path)
    cols_keep = ['GEOID', 'geometry', 'x', 'y', 'total_eligible']
    gdf_pop[cols_keep].to_file(conf.eligible_pop_path, index=False, driver='GeoJSON')

    # Get destinations: block groups centroids, with job data attached. ave at conf.opp_jobs_path
    lodes_path = data_path / 'pa_wac_S000_JT02_2021.csv'
    df_opp_jobs = demographics.get_opp_jobs(lodes_path, county_avg_wage=conf.COUNTY_AVG_WAGE)
    df_opp_jobs['GEOID20'] = df_opp_jobs['GEOID20'].astype(str)
    df_opp_jobs['GEOID'] = df_opp_jobs['GEOID20'].str[:12] # get GEOID of block group, which is first 12 digits of the GEOID that includes block number
    df_opp_jobs_bg = df_opp_jobs.groupby('GEOID')['opp_jobs_total'].sum().reset_index() # sum opp jobs by block group instead of block b/c the number of blocks is too high
    gdf_jobs = demographics.join_df_to_shapefile(df_opp_jobs_bg, bg_shapefile_path)
    cols_keep = ['GEOID', 'geometry', 'x', 'y', 'opp_jobs_total']
    gdf_jobs[cols_keep].to_file(conf.opp_jobs_path, index=False, driver='GeoJSON')

#TODO: Would be better to save as a single gdf and then add orgs and dsts from this single gdf.