
#%%
# import libraries
import geopandas as gpd
import pandas as pd
import os
import matplotlib.pyplot as plt
import re

def create_parking_nodes(input_filepath, output_filepath, study_area_filepath):
    """
    Create representative parking nodes from input parking data and save them as a GeoJSON file.

    This function reads parking data from a CSV file, processes it to remove invalid or
    irrelevant entries, computes average rates, and generates representative parking nodes.
    The resulting GeoDataFrame of parking nodes is clipped to a specified study area and saved
    as a GeoJSON file.

    Parameters:
        input_filepath (str): The file path of the input parking data in CSV format.
        output_filepath (str): The file path to save the resulting parking nodes as a GeoJSON file.
    """
    cwd = os.getcwd()
    df_park = pd.read_csv(input_filepath)
    # Remove rows that do not have both a lat and long populated
    df_park = df_park.loc[~((df_park['latitude'].isnull()) | (df_park['longitude'].isnull()))]
    df_park.loc[df_park.rate == 'Multi_Rate']
    # Remove rows that do not have a rate populated or has a "Multi-Rate"
    df_park = df_park.loc[(~(df_park['rate'].isnull()) & ~(df_park['rate'] == 'Multi-Rate'))]

    # extract rate and remove leading $ sign 
    def to_float_rate(string_rate):
        float_rate = float(re.split(r'[(|/]', string_rate)[0].lstrip('$'))
        return(float_rate)

    df_park['float_rate'] = df_park.apply(lambda row: to_float_rate(row['rate']), axis=1)  # hourly

    # For simplicity, choose just one "representative" parking point for each zone
    df_park_avg = df_park.groupby('zone').agg({'latitude':'mean', 'longitude':'mean', 'float_rate':'mean'}).reset_index()
    gdf_park_avg = gpd.GeoDataFrame(data=df_park_avg, geometry=gpd.points_from_xy(x=df_park_avg.longitude, y=df_park_avg.latitude),crs='epsg:4326')
    gdf_park_avg.plot()
    study_area_gdf = gpd.read_file(study_area_filepath)
    gdf_park_avg_clip = gpd.clip(gdf_park_avg, study_area_gdf)

    gdf_park_avg_clip.to_file(output_filepath, driver='GeoJSON')

    fig, ax = plt.subplots()
    study_area_gdf.plot(ax=ax)
    gdf_park_avg.plot(ax=ax, color='red')

# call function
# cwd = os.getcwd()
# filepath = os.path.join(cwd, 'Data', 'Input_Data','ParkingMetersPaymentPoints.csv')
# create_parking_nodes(filepath)
# %%
