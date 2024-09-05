#%% libraries
from pathlib import Path
import os

from nomad.data.network import parking, study_area, transit_headway, transit_traversal, vehicle_crashes, inrix, streets
from nomad import conf

def process_data():
    '''function description here'''

    study_area.create_study_area(conf.neighborhoods_inpath, conf.neighborhoods, conf.study_area_outpath)
    print('study area created')

    # download crash data
    site = "https://data.wprdc.org"
    resource_ids = ["514ae074-f42e-4bfb-8869-8d8c461dd824","cb0a4d8b-2893-4d20-ad1c-47d5fdb7e8d5"]
    check_file = os.path.isfile(conf.crash_sample_path)
    # only download the data if doesn't already exist
    if not check_file: 
        vehicle_crashes.download_crash_data(site, resource_ids, conf.crash_sample_path)  
        print('crash data downloaded')
    else:
        print('crash data already downloaded')

    # create parking nodes as a GeoJSON file
    parking.create_parking_nodes(conf.parking_inpath, conf.parking_outpath, conf.study_area_outpath)
    print('parking nodes created')

    # create and save transit headway and traversal files
    TIME_START = conf.TIME_START
    TIME_END = conf.TIME_END
    INTERVAL_SPACING = conf.INTERVAL_SPACING
    transit_headway.create_PT_headway_static(conf.GTFS_path, conf.PT_headway_path_static, TIME_START, TIME_END)
    check_file = os.path.isfile(conf.PT_headway_path_dynamic)
    if not check_file:
        transit_headway.create_PT_headway_dynamic(conf.GTFS_path, conf.PT_headway_path_dynamic, TIME_START, TIME_END, INTERVAL_SPACING)
    transit_traversal.create_PT_traversal(conf.GTFS_path, TIME_START, TIME_END, conf.PT_traversal_path)
    print('PT headway and traversal files created')

    # construct travel time and reliability ratio dfs relative to 7am for each hour:min (in increments of 5 min)
    inrix.inrix_to_ratios(conf.inrix_travel_time_path, conf.inrix_roadID_path, int(TIME_START/3600), int(TIME_END/3600), conf.travel_time_ratio_path, conf.reliability_ratio_path)
    print('INRIX data processed')

    # convert streets shapefile into two graphs: driving and biking. save them both to disk.
    # process the raw streets shapefile by adding an indicator for type of bike infrastructure (if any) as well as predicted number of crashes
    streets.process_street_centerlines(conf.study_area_outpath, conf.streets_shapefile_path, conf.crash_sample_path, 
                                       conf.bike_map_folder, conf.streets_processed_path, conf.G_drive_path, conf.G_bike_path)
    print('street processed and drive/bike graphs created')