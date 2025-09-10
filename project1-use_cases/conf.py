"""
Configuration file for the project.

This file defines various configuration parameters and file paths used in the project.
It includes settings for data paths, study area geography, supernetwork parameters, 
scooter simulation settings, time factors, speed parameters, reliability parameters, 
price parameters, risk crash index parameters, discomfort parameters, and demographic data.

Attributes:
    base_path (Path): The absolute path to the base directory of the project.
    network_raw_data_folder (Path): Path to the folder containing raw network data.
    network_processed_data_folder (Path): Path to the folder containing processed network data.
    demographics_raw_data_folder (Path): Path to the folder containing raw demographic data.
    demographics_processed_data_folder (Path): Path to the folder containing processed demographic data.
    config (dict): A dictionary containing all configuration parameters, including:
        - geography: Study area neighborhoods.
        - supernetwork: Parameters for the supernetwork model.
        - paths: File paths for raw and processed data.
        - api: API details for accessing crash data.
        - scooter_simulation: Parameters for simulated scooter historical data.
        - time_factors: Time-related parameters for dynamic analysis.
        - speed: Speed parameters for different modes of transport.
        - reliability: Reliability parameters for transport modes.
        - price_params: Pricing parameters for various transport modes.
        - risk_crash_idx: Risk crash index parameters for different transport modes.
        - discomfort_params: Discomfort parameters for different transport modes.
        - demographics: Demographic data parameters.
"""
from pathlib import Path

# Define the base paths
base_path = Path(__file__).resolve().parent.absolute()
# Network
network_raw_data_folder = base_path / 'data' / 'network' / 'raw'
network_processed_data_folder = base_path / 'data' / 'network' / 'processed'
# Demographics
demographics_raw_data_folder = base_path / 'data' / 'demographics' / 'raw'
demographics_processed_data_folder = base_path / 'data' / 'demographics' / 'processed'

config = {
    # Neighborhoods in the study area
    'geography': {
        'neighborhoods': [
            'Central Oakland', 'North Oakland', 'Squirrel Hill South', 'Squirrel Hill North', 'Shadyside', 'Bloomfield', 'Friendship', 'Garfield', 'East Liberty', 'Larimer',
            'Homewood West', 'Homewood South', 'Homewood North', 'Point Breeze North',
            'Point Breeze', 'South Oakland', 'Greenfield', 'Hazelwood', 'Glen Hazel', 'Regent Square', 'Swisshelm Park'
        ]
    },

    # Supernetwork parameters
    'supernetwork': { 
        'modes_included': ['bs', 'sc', 't', 'pt'],
        'W_tx': 0.5,      # max allowable walking transfer distance (miles)
        'W_od_cnx': 0.6,  # max allowable walking distance for origin/destination connector edge (miles)
        'W_od': 0.82      # max allowable walking distance from origin to destination (miles)
    },

    # Base folder paths
    'paths': {
        'network_raw_data_folder': str(network_raw_data_folder),
        'network_processed_data_folder': str(network_processed_data_folder),
        'demographics_processed_data_folder': str(demographics_processed_data_folder),
        
        'data': {
            # Raw network data files
            'neighborhoods_in': str(network_raw_data_folder / 'Neighborhoods' / 'Neighborhoods_.shp'),
            'crash_sample': str(network_raw_data_folder / 'crashes_sample.csv'),
            'parking_in': str(network_raw_data_folder / 'ParkingMetersPaymentPoints.csv'),
            'GTFS': str(network_raw_data_folder / 'GTFS'),
            'inrix_travel_time': str(network_raw_data_folder / 'Allegheny_sample_xd_part1' / 'Allegheny_sample_xd_part1.csv'),
            'inrix_roadID': str(network_raw_data_folder / 'Allegheny_sample_xd_part1' / 'XD_Identification.csv'),
            'streets_shapefile': str(network_raw_data_folder / 'alleghenycounty_streetcenterlines202305' / 'AlleghenyCounty_StreetCenterlines202304.shp'),
            'bike_map_folder': str(network_raw_data_folder / 'bike-map-2019'),
            'bikeshare_station': str(network_raw_data_folder / 'pogoh-station-locations-2022.csv'),
            'carshare_station': str(network_raw_data_folder / 'Zipcar_Depot.csv'),

            # Processed network data files
            'study_area_out': str(network_processed_data_folder / 'study_area.csv'),
            'parking_out': str(network_processed_data_folder / 'parking_points.csv'),
            'PT_headway_static': str(network_processed_data_folder / 'headway_static.csv'),
            'PT_headway_dynamic': str(network_processed_data_folder / 'headway_dynamic.csv'),
            'PT_traversal': str(network_processed_data_folder / 'traversal.csv'),
            'crash_model': str(network_processed_data_folder / 'crash_model.pickle'),
            'travel_time_ratio': str(network_processed_data_folder / 'tt_ratio.csv'),
            'reliability_ratio': str(network_processed_data_folder / 'rel_ratio.csv'),
            'streets_processed': str(network_processed_data_folder / 'streets_processed.csv'),

            # Base graphs
            'G_drive': str(network_processed_data_folder / 'base_graphs' / 'G_drive.gpickle'),
            'G_bike': str(network_processed_data_folder / 'base_graphs' / 'G_bike.gpickle'),

            # Raw demographic files
            'block_group_shapefile': str(demographics_raw_data_folder / 'tl_2022_42_bg' / 'tl_2022_42_bg.shp'),
            'naics_nem_xwalk':  str(demographics_raw_data_folder / 'naics-nem-crosswalk.xlsx'),
            'nem_onet_xwalk': str(demographics_raw_data_folder / 'nem-onet-to-soc-crosswalk.xlsx'),
            'onet_edu': str(demographics_raw_data_folder / 'onet_education.xlsx'),
            'onet_categories': str(demographics_raw_data_folder /'onet_edu_categories.xlsx'),
            'ipums': str(demographics_raw_data_folder / 'usa_00002.csv'),    
            'lodes': str(demographics_raw_data_folder / 'pa_wac_S000_JT02_2021.csv'),
            
            # Processed demographic files
            'block_group_centroids': str(demographics_processed_data_folder / 'block_group_centroids.csv'),
            'opportunity_jobs': str(demographics_processed_data_folder / 'opportunity_jobs.csv'),
        },

        # API details
        'api': {
            'crash_data_api': "https://data.wprdc.org",
            'crash_api_resource_ids': ["514ae074-f42e-4bfb-8869-8d8c461dd824", "cb0a4d8b-2893-4d20-ad1c-47d5fdb7e8d5"],
        }
    },

    # Scooter simulation
    'scooter_simulation': {
        'NUM_OBS': 1500 * (2/3),  # per movepgh report
        'NUM_DAYS_OF_DATA': 30    # simulate as if we had 30 days of data
    },

    # Time factors
    'time_factors': {
        'TIME_START': 7 * 3600,    # (must be in the form of seconds_after_midnight)
        'TIME_END': 9 * 3600,      # (must be in the form of seconds_after_midnight)
        'INTERVAL_SPACING': 10,    # sec
        #'NUM_INTERVALS': int((TIME_END - TIME_START) / INTERVAL_SPACING),     
        'INRIX_SPACING': 300      # seconds (5 min*60 sec/min); how often are measurements taken with inrix data
        #'INCONVENIENCE_COST': 2    # minutes
    },

    # Speed parameters
    'speed': {
        'WALK_SPEED': 1.3,                 # m/s
        'SCOOT_SPEED': 2.78,               # m/s
        'BIKE_SPEED': 14.5 / 3600 * 1000,  # m/s
        'TNC_WAIT_TIME': 6,                # minutes
        'ALIGHTING_TIME': 5                # seconds
    },

    # Reliability parameters
    'reliability': {
        'TNC_WAIT': 2
    },

    # w = walk
    # sc = scooter
    # sc_tx = scooter transfer
    # bs = bikeshare
    # t = tnc
    # t_wait = tnc waiting edge
    # board = public transit boarding edge
    # alight = public transit alighting edge
    # rt = public transit route edge (in-vehicle)
    # pb = personal bike
    # z = carshare (zipcar)
    # pv = personal vehicle
    # park = parking edge

    # Price parameters  # ppmin (price per minute); ppmile (price per mile); fixed (fixed price per trip)
    'price_params': {
        'w': {'ppmin': 0, 'ppmile': 0, 'fixed': 0},  
        'sc': {'ppmin': 0.39, 'ppmile': 0, 'fixed': 0},  
        'sc_tx': {'ppmin': 0, 'ppmile': 0, 'fixed': 1},  # price to transfer to a scooter
        'bs': {'ppmin': 25/200, 'ppmile': 0, 'fixed': 0},
        't': {'ppmin': 0.19, 'ppmile': 1.12, 'fixed': 0},  
        't_wait': {'ppmin': 0, 'ppmile': 0, 'fixed': 3.03 + 2.64 + 1},  # fixed price is: base fare + "booking fee" + $1 minfare buffer
        'board': {'ppmin': 0, 'ppmile': 0, 'fixed': 2.75},
        'alight': {'ppmin': 0, 'ppmile': 0, 'fixed': 0},
        'pt': {'ppmin': 0, 'ppmile': 0, 'fixed': 0},
        'rt': {'ppmin': 0, 'ppmile': 0, 'fixed': 0},
        'pb': {'ppmin': 0, 'ppmile': 0, 'fixed': 0},
        'z': {'ppmin': 11/60, 'ppmile': 0, 'fixed': 0, 'fixed_per_month': 9, 'est_num_trips': 4},
        'pv': {'ppmin': 0, 'ppmile': 0.20, 'fixed': 0},
        'park': {'ppmin': 0, 'ppmile': 0, 'fixed': 2.50 * 8} 
    },

    # Risk crash index parameters
    'risk_crash_idx': {
        'w': 0.28,
        'sc': 1.81,
        'bs': 1.81,
        't': 1,
        't_wait': 0,
        'board': 0.19,
        'alight': 0.19,
        'pt': 0.19,
        'pb': 1.81,
        'z': 1,
        'pv': 1,
        'park': 1
    },

    # Discomfort parameters
    'discomfort_params': {
        'w': 2.86 / 1.34,
        'sc': 3.26 / 1.34,
        'bs': 3.26 / 1.34,
        't': 1.34 / 1.34,  # use vehicle as baseline
        't_wait': 0,
        'board': 2.22 / 1.34,  # could change if thinking about cold weather conditions
        'alight': 0,
        'pt': 2.22 / 1.34,
        'pb': 3.26 / 1.34,
        'z': 1,
        'pv': 1,
        'park': 1
    },

    'conversion_factors': {
        'MILE_TO_METERS': 1609.34
    },

    'INCONVENIENCE_COST': 2,   # minutes, associated with transferring
    'CIRCUITY_FACTOR': 1.2,     # to adjust euclidean walking distance to network distance

    'demographics': {
        'COUNTY_AVG_WAGE': 45939,  # in Allegheny County
        'AC_AVG_COMMUTE': 26.6 * 60, # seconds
    }

}

# compute number of time intervals dynamically
tf = config['time_factors']
tf['NUM_INTERVALS'] = int((tf['TIME_END'] - tf['TIME_START']) / tf['INTERVAL_SPACING'])

