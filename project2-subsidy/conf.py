from pathlib import Path

# Define the base paths
base_path = Path(__file__).resolve().parent.absolute()
# Network
raw_data_folder = base_path / 'data' / 'network' / 'raw'
processed_data_folder = base_path / 'data' / 'network' / 'processed'
# Demographics
demographics_raw_data_folder = base_path / 'data' / 'demographics' / 'raw'
demographics_processed_data_folder = base_path / 'data' / 'demographics' / 'processed'

config = {
    # Study area geography
    'geography': {
        'neighborhoods': [
            'Central Oakland', 'North Oakland', 'Squirrel Hill South', 'Squirrel Hill North', 'Shadyside',
            'Point Breeze', 'South Oakland', 'Greenfield', 'Hazelwood', 'Glen Hazel', 'Regent Square', 'Swisshelm Park'
        ]
    },

    # Supernetwork parameters
    'supernetwork': { 
        'modes_included': ['bs', 'sc', 't', 'pt'],
        'W_tx': 0.5,
        'W_od_cnx': 0.6,
        'W_od': 0.82
    },

    # Base folder paths
    'paths': {
        'raw_data_folder': str(raw_data_folder),
        'processed_data_folder': str(processed_data_folder),
        'demographics_processed_data_folder': str(demographics_processed_data_folder),
        
        'data': {
            # Raw network data files
            'neighborhoods_in': str(raw_data_folder / 'Neighborhoods' / 'Neighborhoods_.shp'),
            'crash_sample': str(raw_data_folder / 'crashes_sample.csv'),
            'parking_in': str(raw_data_folder / 'ParkingMetersPaymentPoints.csv'),
            'GTFS': str(raw_data_folder / 'GTFS'),
            'inrix_travel_time': str(raw_data_folder / 'Allegheny_sample_xd_part1' / 'Allegheny_sample_xd_part1.csv'),
            'inrix_roadID': str(raw_data_folder / 'Allegheny_sample_xd_part1' / 'XD_Identification.csv'),
            'streets_shapefile': str(raw_data_folder / 'alleghenycounty_streetcenterlines202305' / 'AlleghenyCounty_StreetCenterlines202304.shp'),
            'bike_map_folder': str(raw_data_folder / 'bike-map-2019'),
            'bikeshare_station': str(raw_data_folder / 'pogoh-station-locations-2022.csv'),
            'carshare_station': str(raw_data_folder / 'Zipcar_Depot.csv'),

            # Processed network data files
            'study_area_out': str(processed_data_folder / 'study_area.csv'),
            'parking_out': str(processed_data_folder / 'parking_points.csv'),
            'PT_headway_static': str(processed_data_folder / 'headway_static.csv'),
            'PT_headway_dynamic': str(processed_data_folder / 'headway_dynamic.csv'),
            'PT_traversal': str(processed_data_folder / 'traversal.csv'),
            'crash_model': str(processed_data_folder / 'crash_model.pickle'),
            'travel_time_ratio': str(processed_data_folder / 'tt_ratio.csv'),
            'reliability_ratio': str(processed_data_folder / 'rel_ratio.csv'),
            'streets_processed': str(processed_data_folder / 'streets_processed.csv'),

            # Base graphs
            'G_drive': str(processed_data_folder / 'base_graphs' / 'G_drive.gpickle'),
            'G_bike': str(processed_data_folder / 'base_graphs' / 'G_bike.gpickle'),

            # Raw demographic files
            'poverty': str(demographics_raw_data_folder / 'ACS_5Y_2022_B17017_transp.csv'),
            'block_group_shapefile': str(demographics_raw_data_folder / 'tl_2022_42_bg' / 'tl_2022_42_bg.shp'),
            'naics_nem_xwalk':  str(demographics_raw_data_folder / 'naics-nem-crosswalk.xlsx'),
            'nem_onet_xwalk': str(demographics_raw_data_folder / 'nem-onet-to-soc-crosswalk.xlsx'),
            'onet_edu': str(demographics_raw_data_folder / 'onet_education.xlsx'),
            'onet_categories': str(demographics_raw_data_folder /'onet_edu_categories.xlsx'),
            'ipums': str(demographics_raw_data_folder / 'usa_00002.csv'),    
            'lodes': str(demographics_raw_data_folder / 'pa_wac_S000_JT02_2021.csv'),
            # Processed demographic files
            'poverty_pop': str(demographics_processed_data_folder / 'poverty_pop.csv'),
            'opp_jobs': str(demographics_processed_data_folder / 'opportunity_jobs.csv')
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
        'NUM_DAYS_OF_DATA': 30   # simulate as if we had 30 days of data
    },

    # Time factors
    'time_factors': {
        'TIME_START': 7 * 3600,  # (must be in the form of seconds_after_midnight)
        'TIME_END': 9 * 3600,    # (must be in the form of seconds_after_midnight)
        'INTERVAL_SPACING': 10,   # sec
        'NUM_INTERVALS': None,    # to be calculated later
        'INRIX_SPACING': 300,      # seconds (5 min*60 sec/min); how often are measurements taken with inrix data
        'INCONVENIENCE_COST': 2    # minutes
    },

    # Speed parameters
    'speed': {
        'WALK_SPEED': 1.3,              # m/s
        'SCOOT_SPEED': 2.78,            # m/s
        'BIKE_SPEED': 14.5 / 3600 * 1000,  # m/s
        'TNC_WAIT_TIME': 6,             # minutes
        'ALIGHTING_TIME': 5              # seconds
    },

    # Reliability parameters
    'reliability': {
        'BOARDING_RELIABILITY': 1.5,
        'TNC_WAIT_RELIABILITY': 2
    },

    # Price parameters
    'price_params': {
        'w': {'ppmin': 0, 'ppmile': 0, 'fixed': 0},  
        'sc': {'ppmin': 0.39, 'ppmile': 0, 'fixed': 0},  
        'sc_tx': {'ppmin': 0, 'ppmile': 0, 'fixed': 1},
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

    'demographics': {
        'COUNTY_AVG_WAGE': 45939,  # in Allegheny County
        'AC_AVG_COMMUTE': 26.6 * 60, # seconds
        'TRAVEL_TIME_THRESHOLD': 30 * 60       # 30 min = 30*60 seconds
    }

}
