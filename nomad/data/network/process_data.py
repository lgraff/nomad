# Libraries
import os
from nomad.data.network import parking, study_area, transit_headway, transit_traversal, vehicle_crashes, inrix, streets

def process_data(config):
    """
    Process data for various transportation and urban infrastructure layers.
    
    This function processes a variety of datasets including crash data, parking data, transit headway and traversal,
    travel time ratios, street networks, and more. The function reads input paths and parameters from the provided 
    `config` dictionary, which allows easy configuration of file paths and processing options.

    Parameters:
    ----------
    config : dict
        A configuration dictionary containing the following keys, among others:
        
        - 'paths': A dictionary of paths containing:
            - 'neighborhoods_in': Path to the input shapefile of neighborhoods.
            - 'study_area_out': Path where the processed study area data will be saved.
            - 'crash_data_api': API URL for downloading crash data.
            - 'crash_api_resource_ids': List of resource IDs for the crash data API.
            - 'crash_sample': Path to save the downloaded crash data.
            - 'parking_in': Path to the input CSV file for parking meter data.
            - 'parking_out': Path where the processed parking nodes will be saved.
            - 'GTFS': Path to the GTFS transit data.
            - 'PT_headway_static': Path to save the static transit headway data.
            - 'PT_headway_dynamic': Path to save the dynamic transit headway data.
            - 'PT_traversal': Path to save the transit traversal data.
            - 'inrix_travel_time': Path to INRIX travel time data.
            - 'inrix_roadID': Path to INRIX road ID data.
            - 'travel_time_ratio': Path to save the travel time ratio data.
            - 'reliability_ratio': Path to save the reliability ratio data.
            - 'streets_shapefile': Path to the streets shapefile.
            - 'bike_map_folder': Path to the bike map folder.
            - 'streets_processed': Path to save the processed streets data.
            - 'G_drive': Path to save the driving graph.
            - 'G_bike': Path to save the biking graph.
        
        - Other parameters: 
            - 'neighborhoods': List of neighborhoods to keep during the study area creation.
            - 'TIME_START': Integer representing the start time for processing transit data (in seconds).
            - 'TIME_END': Integer representing the end time for processing transit data (in seconds).
            - 'INTERVAL_SPACING': Interval spacing (in minutes) for dynamic transit headway.
    
    Returns:
    -------
    None
        The function performs data processing tasks and saves the outputs to the specified locations.
    
    Notes:
    -----
    - Ensure that all necessary input files exist at the paths specified in the `config` dictionary.
    - The function checks if certain output files (like crash data and dynamic headways) already exist before processing,
      to avoid redundant downloads or recalculations.
    
    Example:
    -------
    config = {
        'paths': {
            'neighborhoods_in': 'path/to/neighborhoods.shp',
            'study_area_out': 'path/to/study_area.csv',
            'crash_data_api': 'https://data.wprdc.org',
            'crash_api_resource_ids': ['resource_id_1', 'resource_id_2'],
            'crash_sample': 'path/to/crash_data.csv',
            'parking_in': 'path/to/parking.csv',
            'parking_out': 'path/to/parking_processed.csv',
            'GTFS': 'path/to/gtfs',
            'PT_headway_static': 'path/to/pt_headway_static.csv',
            'PT_headway_dynamic': 'path/to/pt_headway_dynamic.csv',
            'PT_traversal': 'path/to/pt_traversal.csv',
            'inrix_travel_time': 'path/to/inrix_travel_time.csv',
            'inrix_roadID': 'path/to/inrix_roadID.csv',
            'travel_time_ratio': 'path/to/travel_time_ratio.csv',
            'reliability_ratio': 'path/to/reliability_ratio.csv',
            'streets_shapefile': 'path/to/streets.shp',
            'bike_map_folder': 'path/to/bike_map/',
            'streets_processed': 'path/to/processed_streets.csv',
            'G_drive': 'path/to/driving_graph.gpkg',
            'G_bike': 'path/to/biking_graph.gpkg',
        },
        'neighborhoods': ['Neighborhood A', 'Neighborhood B'],
        'TIME_START': 0,
        'TIME_END': 86400,  # 24 hours
        'INTERVAL_SPACING': 15,  # 15 minutes
    }
    
    process_data(config)
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
    inrix.inrix_to_ratios(data_paths['inrix_travel_time'], data_paths['inrix_roadID'], int(TIME_START / 3600), int(TIME_END / 3600), data_paths['travel_time_ratio'], data_paths['reliability_ratio'])
    print('INRIX data processed')

    # Convert streets shapefile into two graphs (driving and biking), save them to disk
    streets.process_street_centerlines(data_paths['study_area_out'], data_paths['streets_shapefile'], data_paths['crash_sample'], data_paths['crash_model'],
                                       data_paths['bike_map_folder'], data_paths['streets_processed'], data_paths['G_drive'], data_paths['G_bike'])
    print('Street processed and drive/bike graphs created')
