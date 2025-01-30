from pathlib import Path

import pandas as pd
import geopandas as gpd

from nomad import conf

def weighted_average(df, value, weight):
    '''Return the weighted average of the column df.value, weighted by the column df.weight.'''
    val = df[value]
    wt = df[weight]
    return (val * wt).sum() / wt.sum()

def get_opp_jobs(lodes_path, county_avg_wage):
    '''Find number of opportunity jobs by block.'''
    data_path = Path().resolve() / 'nomad' / 'data' / 'subsidy' / 'raw'
    #TODO: for more precision, if desired, use this instead: https://www.bls.gov/oes/current/naics2_11.htm#11-0000 
    # this will provide median/mean hourly wage by NAICS code. can adjust by CPS (see Wardrip)

    # Crosswalks for mapping
    naics_nem_crosswalk = pd.read_excel(data_path / 'naics-nem-crosswalk.xlsx', sheet_name='Table 1.9', skiprows=1)  # nationwide data: NAICs code to national employment matrix code
    # clean up the nem code for consistency (add dash after first two digits)
    naics_nem_crosswalk['2022 National Employment Matrix code'] = naics_nem_crosswalk['2022 National Employment Matrix code'].str[0:2] + '-' + naics_nem_crosswalk['2022 National Employment Matrix code'].str[2:]
    nem_onet_crosswalk =  pd.read_excel(data_path / 'nem-onet-to-soc-crosswalk.xlsx', skiprows=4)   # nationwide data

    # O*NET data for surveyed workers' responses to required level of edu, keyed by O*NET SOC Code
    df_onet = pd.read_excel(data_path / 'onet_education.xlsx')  # nationwide data
    data_path = Path().resolve() / 'nomad' / 'data' / 'subsidy' / 'raw'
    df_onet = df_onet[df_onet['Scale ID'] == 'RL']  # RL stands for "required level of education"
    df_onet['Category'] = df_onet['Category'].astype('int') 
    df_onet_categories = pd.read_excel(data_path / 'onet_edu_categories.xlsx')  # nationwide data
    edu_categories = df_onet_categories.loc[df_onet_categories['Scale ID'] == 'RL'][['Category', 'Category Description']]
    df_onet = df_onet.merge(edu_categories, on='Category', how='left')
    df_onet = df_onet.merge(nem_onet_crosswalk, how='left', on='O*NET-SOC Code') # add NEM code
    df_onet = df_onet[['O*NET-SOC Code', 'Title', 'Category', 'Data Value', 'NEM Code']]
    df_onet = df_onet.pivot(index=['O*NET-SOC Code', 'NEM Code'], columns='Category', values='Data Value').add_prefix('category_').reset_index()
    df_onet['pct_less_bachelors_onet'] = df_onet['category_1'] + df_onet['category_2'] + df_onet['category_3'] +  df_onet['category_4'] + df_onet['category_5']
    onet_cols = ['O*NET-SOC Code', 'NEM Code', 'pct_less_bachelors_onet']
    df_onet = df_onet[onet_cols].sort_values(by='O*NET-SOC Code', ascending=True).drop_duplicates(subset=['NEM Code'], keep='first')

    # BLS data for median hourly wage, keyed by NAICS code
    df_wage = pd.read_excel(data_path / 'naics-nem-crosswalk.xlsx', sheet_name='Table 1.7', skiprows=1) # nationwide data
    # data cleaning with the wage variable
    df_wage['Median annual wage, 2022(1)'].fillna('0', inplace=True) # fill na values
    df_wage['Median annual wage, 2022(1)'] = df_wage['Median annual wage, 2022(1)'].apply(lambda x: '0' if x == '—' else x)
    df_wage['Median annual wage, 2022(1)'] = df_wage['Median annual wage, 2022(1)'].apply(lambda x: ''.join(c for c in str(x) if c.isdigit()))
    df_wage['Median_annual_wage'] = df_wage['Median annual wage, 2022(1)'].astype('int')
    #df_wage = df_wage[df_wage['AREA_TITLE'] == 'Pittsburgh, PA']
    #Merge the wage and required education dfs together. result is one df, keyed by NEM code
    df_wage['NEM Code'] = df_wage['2022 National Employment Matrix code'].copy() # make the key name consistent
    df_wage = df_wage[~df_wage['NEM Code'].isna()] # only keep if there's a NEM Code
    wage_cols = ['NEM Code', '2022 National Employment Matrix title', 'Median_annual_wage', 'Typical education needed for entry']
    df_wage = df_wage[wage_cols]

    # Get 'opportunity' df, which has wage info and required level of edu by NEM code
    df_opp = df_wage.merge(df_onet, on='NEM Code', how='left') 
    # source for below: https://labormarketinfo.edd.ca.gov/LMID/BLS_Training_Levels.html
    less_bachelors_list = ['High school diploma or equivalent','Some college, no degree','No formal educational credential','Postsecondary nondegree award',"Associate's degree"]
    df_opp['less_bachelors_bls'] = df_opp['Typical education needed for entry'].apply(lambda x: 1 if x in less_bachelors_list else 0)
    df_opp['ind_less_bachelors'] = df_opp.apply(lambda row: 1 if ((row['pct_less_bachelors_onet'] >= 50) | (row['less_bachelors_bls'] == 1)) else 0, axis=1)
    # Label NEM code as 'opportunity' if median annual wage > county's per capita avg wage in the && required level of education < Bachelors
    df_opp['opp_occupation'] = df_opp.apply(lambda row: 1 if ((row['Median_annual_wage'] >= county_avg_wage) & (row['ind_less_bachelors'] == 1)) else 0, axis=1)
    df_opp['OCCSOC'] = df_opp['NEM Code'].str[:2] + df_opp['NEM Code'].str[3:]

    # here is why i think OCCSOC and NEM are equivalent: https://www.bls.gov/emp/data/occupational-data.htm
    # "The occupational structure of the Matrix is based on the structure used by the OEWS program, which is currently using the 2018 Standard Occupational Classification (SOC) system
    # furthermore, under directories and crosswalks heading, we see: National Employment Matrix/SOC

    # IPUMS data
    ipums_path = data_path / 'usa_00002.csv'  # nationwide data
    df_ipums = pd.read_csv(ipums_path)
    df_ipums['OCCSOC'] = df_ipums['OCCSOC'].str.strip()
    df_ipums = df_ipums[~(df_ipums['OCCSOC']=='0')]  # remove null data
    df_final = df_ipums.merge(df_opp, how='left', on='OCCSOC')
    df_final['NAICS_2digit'] = df_final['INDNAICS'].str[:2]  # naics industry code is first two digits
    # each row is one person, who represents N persons with the same characeteristics.
    # N is given by the "PERWT" columns. needs to be considered as we calculate opp share
    opp_share_by_naics = df_final.groupby(by='NAICS_2digit').apply(weighted_average, 'opp_occupation', 'PERWT').reset_index()
    opp_share_by_naics.columns = ['NAICS_2digit', 'opp_occupation']
    #opp_share_by_naics = df_final.groupby(by='NAICS_2digit').mean().reset_index()[['NAICS_2digit','opp_occupation']]
    naics_opp_share_dict = dict(zip(opp_share_by_naics['NAICS_2digit'], opp_share_by_naics['opp_occupation']))

    # use lodes/lehd to find number of jobs by census block, NAICS industry code pair
    #lodes_path = data_path / 'pa_wac_S000_JT02_2021.csv'
    df_lodes = pd.read_csv(lodes_path)
    df_lodes['GEOID20'] = df_lodes['w_geocode'].copy()

    #gdf = gpd.read_file(os.path.join(pa_blocks_folder, pa_blocks_shapefile))

    naics_sectors = [[11], [21], [22], [23], [31,32,33], [42], [44, 45], [48, 49], [51], [52],
                    [53], [54], [55], [56], [61], [62], [71], [72], [81], [92]]  # from lodes documentation

    lodes_cols = [col for col in df_lodes.columns.to_list() if col.startswith('CNS')]
    df_lodes = df_lodes[['GEOID20'] + lodes_cols]
    lodes_to_naics_sector_dict = dict(zip(lodes_cols, naics_sectors))

    # for each column (and associated naics code), what is the opportunity occupation share?
    # for columns that encompass more than 1 naics code, take the average 
    naics_opp_share_dict = dict(zip(opp_share_by_naics['NAICS_2digit'], opp_share_by_naics['opp_occupation']))
    lodes_opp_share_dict = {}
    for col, naics in lodes_to_naics_sector_dict.items():
        opp_share = 0
        for i,n in enumerate(naics):
            opp_share += naics_opp_share_dict[str(n)]
        opp_share = opp_share/(i+1)
        lodes_opp_share_dict[col] = opp_share

    # the final df which contains the number of opp jobs for each naics group by GEOID
    df_opp_jobs = df_lodes.copy()
    for col in lodes_cols:
        df_opp_jobs[col] = df_opp_jobs[col] * lodes_opp_share_dict[col]

    # add all the jobs; these are the destinations 
    df_opp_jobs['opp_jobs_total'] = df_opp_jobs.drop(columns='GEOID20').sum(axis=1)
    # these are the destinations (by block). need block-level shapefile for merging
    df_opp_jobs = df_opp_jobs[[ 'GEOID20', 'opp_jobs_total']]

    return df_opp_jobs

def join_df_to_shapefile(df, shapefile_path):
    '''Join df to polygon shapefile so that df's data is associated with a geometry. Note: df and shapefile GEOIDs must match.
       Output a shapefile that contains df's attributes, as well as the polygon's x & y centroid coordinates.'''
    study_area_gdf = gpd.read_file(conf.study_area_outpath)
    study_area_gdf.to_crs(epsg=4269, inplace=True)
    
    # # Shrink the study area by x miles
    # x = 0.25 # miles (buffer the PT network even more than the street network b/c we can imagine the case of a bus route going outside the bounds and then returning inside)
    # study_area_gdf = study_area_gdf.to_crs(crs='epsg:32128').buffer(x*1609).to_crs('EPSG:4326')  # 1609 meters/mile

    shape = gpd.read_file(shapefile_path, mask=study_area_gdf)  # only include blocks within study area
    shape.to_crs(epsg=4269, inplace=True)
    shape = gpd.clip(shape, study_area_gdf) # clip again
    shape.to_crs(epsg=4326, inplace=True)
    # Get centroid coordinates
    shape['x'] = shape.to_crs(epsg=2272).centroid.to_crs(epsg=4326).x
    shape['y'] = shape.to_crs(epsg=2272).centroid.to_crs(epsg=4326).y
    shape = shape.merge(df, on='GEOID', how='left')
    return shape