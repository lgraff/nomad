# Libraries
import os
from nomad.data.network import parking, study_area, transit_headway, transit_traversal, vehicle_crashes, historical_travel_time, streets

def process_data(config):
    """
    This function processes a variety of datasets including crash data, parking data, transit headway and traversal,
    travel time ratios, street networks, and more. The function reads input paths and parameters from the provided 
    `config` dictionary.
    """
    data_paths = config['paths']['data']

    # Create study area
    study_area.create_study_area(data_paths['neighborhoods_in'], config['geography']['neighborhoods'], data_paths['study_area_out'])
    print('Study area created')

    # Download crash data
    api_info = config['paths']['api']
    site = api_info['crash_data_api']
    resource_ids = api_info['crash_api_resource_ids']
    check_file = os.path.isfile(data_paths['crash_sample'])
    
    # Only download the data if it doesn't already exist
    if not check_file: 
        vehicle_crashes.download_crash_data(site, resource_ids, data_paths['crash_sample'])  
        print('Crash data downloaded')
    else:
        print('Crash data already downloaded')

    # Create parking nodes as a GeoJSON file
    parking.create_parking_nodes(data_paths['parking_in'], data_paths['parking_out'], data_paths['study_area_out'])
    print('Parking nodes created')

    # Create and save transit headway and traversal files
    TIME_START = config['time_factors']['TIME_START']
    TIME_END = config['time_factors']['TIME_END']
    INTERVAL_SPACING = config['time_factors']['INTERVAL_SPACING']
    
    transit_headway.create_PT_headway_static(data_paths['GTFS'], data_paths['PT_headway_static'], TIME_START, TIME_END)
    
    check_file = os.path.isfile(data_paths['PT_headway_dynamic'])
    if not check_file:
        transit_headway.create_PT_headway_dynamic(data_paths['GTFS'], data_paths['PT_headway_dynamic'], TIME_START, TIME_END, INTERVAL_SPACING)
    
    transit_traversal.create_PT_traversal(data_paths['GTFS'], TIME_START, TIME_END, data_paths['PT_traversal'])
    print('PT headway and traversal files created')

    # Construct travel time and reliability ratio dataframes
    historical_travel_time.historical_obs_to_ratios(config['time_factors']['HISTORICAL_TRAVEL_TIME_OBSERVATION_SPACING'], 
                                                    data_paths['historical_obs_travel_time'], data_paths['historical_obs_roadID'], 
                                                    int(TIME_START / 3600), int(TIME_END / 3600), 
                                                    data_paths['travel_time_ratio'], data_paths['reliability_ratio'])
    print('Historical travel time data processed') 

    # Convert streets shapefile into two graphs (driving and biking), save them to disk
    streets.process_street_centerlines(data_paths['study_area_out'], data_paths['streets_shapefile'], data_paths['crash_sample'], data_paths['crash_model'],
                                       data_paths['bike_map_folder'], data_paths['streets_processed'], data_paths['G_drive'], data_paths['G_bike'])
    print('Street processed and drive/bike graphs created')
