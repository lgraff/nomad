from pathlib import Path

import pandas as pd
import geopandas as gpd

from nomad import conf

# extract fips code: https://www.census.gov/programs-surveys/geography/guidance/geo-identifiers.html
# this function establishes the fips code for a census dataset where each row is uniquely defined by a string that contains state, county, block group, and tract 
def generate_fips_code(df_census, fips_map):
    df_census[['Block_Group', 'Census_Tract', 'County', 'State']] = df_census['geography'].str.split(';', expand=True)
    df_census['State'] = df_census['State'].str.strip().map(fips_map)
    df_census['County'] = df_census['County'].str.strip().map(fips_map)
    df_census['Block_Group'] = df_census['Block_Group'].str.extract(r'([\d]+)')
    df_census['Census_Tract'] = df_census['Census_Tract'].str.extract(r'([\d.]+)')
    # https://www2.census.gov/geo/pdfs/reference/GARM/Ch


    df_census['split_census_tract'] = df_census['Census_Tract'].str.split('.')
    df_census['tract_post_decimal'] = df_census['split_census_tract'].apply(lambda x: x[1] if len(x) > 1 else '00')
    df_census['tract_pre_decimal'] = df_census['split_census_tract'].apply(lambda x: x[0]).str.zfill(4) # pad if less than 4 digits
    # df_census['tract_trailing_digits'].unique() # quick test
    df_census['FIPS'] = df_census['State'] + df_census['County'] + df_census['tract_pre_decimal'] + df_census['tract_post_decimal'] + df_census['Block_Group']

def get_eligible_population(census_data_filepath):
    '''Use census data to find the subsidy-eligble population. 
       Note: this code works for a specific census table, would need to be adapted if a different table is preferred.'''
    df = pd.read_csv(census_data_filepath)
    df.index = df['Label (Grouping)']
    df = df.shift(periods=-1)
    df = df.loc[df.index.str.startswith('Block Group')]
    df['geography'] = df.index
    #df['Label'] = df['Label (Grouping)'].str.strip()
    #labels_keep = ['Married-couple family:', ]
    pov_cols = [c for c in df.columns if c.startswith('Total:!!Income in the past 12 months below poverty level:')]
    df = df[pov_cols]
    prefix = 'Total:!!Income in the past 12 months below poverty level:!!'
    cols_keep = [prefix+'Family households:!!Married-couple family:', prefix+'Family households:!!Other family:', prefix+'Nonfamily households:']
    df = df[cols_keep]
    new_cols = dict(zip(cols_keep, ['married_couple', 'other_family', 'non_family']))
    df.rename(columns=new_cols, inplace=True)
    df[['married_couple', 'other_family', 'non_family']] = df.astype(dict(zip(['married_couple', 'other_family', 'non_family'], ['int']*3)))
    df['geography'] = df.index
    fips_map = {'Pennsylvania': '42', 'Allegheny County': '003'}
    generate_fips_code(df, fips_map)
    # get total number of people who qualify per fips code
    # married couple familes count as 2 people; 
    # other family counts as 1 person because if you go into the table, it states that no spouse is present
    # nonfamily household count as 1 person (either a male or female householder)
    # https://www.census.gov/programs-surveys/cps/technical-documentation/subject-definitions.html#:~:text=the%20related%20subfamily.-,Household%2C%20nonfamily,he%2Fshe%20is%20not%20related.
    df['total_eligible'] = 2*df['married_couple'] + 1*df['other_family'] + 1*df['non_family'] # this checks out with the 'Total' column in the table
    df['GEOID'] = df['FIPS'].copy()
    return df

# def join_census_to_shapefile(df_census, shapefile_path):
#     '''Join census data to shapefile so that census data is associated with a geometry. Note: census and shapefile GEOIDs must match.
#        Output a gdf file of centroids which contains population data as an attribute.'''
#     study_area_gdf = gpd.read_file(conf.study_area_outpath)
#     study_area_gdf.to_crs(epsg=4269)
#     shape = gpd.read_file(shapefile_path, mask=study_area_gdf)  # only include blocks within study area
#     shape.to_crs(epsg=4326, inplace=True)
#     centroids = shape.copy()
#     centroids['centroid'] = centroids.to_crs(epsg=2272).centroid.to_crs(epsg=4326)
#     centroids.set_geometry('centroid', inplace=True)
#     centroids = centroids.merge(df_census, on='GEOID', how='left')
#     centroids['x'] = centroids.geometry.x
#     centroids['y'] = centroids.geometry.y
#     return centroids

# USING OTHER CENSUS TABLES
# # output is a .csv file
# data_path = '/home/lgraff/Desktop/Data_Testing'
# df_pop = pd.read_csv(os.path.join(data_path, 'ACS_5Y_2022_B19058.csv'))
# # see census table B19058: public assistance income or food stamps/SNAP
# # https://data.census.gov/table/ACSDT5Y2022.B19058?t=Income%20and%20Poverty&g=050XX00US42003$1500000&tp=true


# df_pop.index = df_pop['Label (Grouping)']
# df_pop = df_pop.shift(periods=-1)
# df_pop = df_pop.loc[df_pop.index.str.startswith('Block Group')]
# df_pop['geography'] = df_pop.index

# generate_fips_code(df_pop, fips_map)

# # alternatively, we can use income threshold from census table B19001
# # https://data.census.gov/table/ACSDT5Y2022.B19001?q=median%20income&g=050XX00US42003$1500000
# df_income = pd.read_csv(os.path.join(data_path, 'ACS_5Y_2022_B19001.csv'))
# df_income.index = df_income['Label (Grouping)']
# df_income = df_income.shift(periods=-1)
# #df_income = df_income[~df_income['Label (Grouping)'].isna()]
# df_income = df_income.loc[df_income.index.str.startswith('Block Group')]
# df_income['geography'] = df_income.index
# generate_fips_code(df_income, fips_map)

# # also see: B17010 for poverty status by family type
