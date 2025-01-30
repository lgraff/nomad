# General Parameters File
from pathlib import Path

# # conversion factors
MILE_TO_METERS = 1609

# # supernetwork
# MODES_INCLUDED = ['bs', 'z', 'sc', 't', 'pt']
CIRCUITY_FACTOR = 1.2
W_tx = 0.5 # **Use of Smart Card Fare Data to Estimate Public Transport Origin–Destination Matrix** ==> 800m = 0.5 miles
W_od_cnx = 0.6 # "New evidence on walking distances to transit stops: identifying redundancies and gaps using variable service areas". From Fig 2, we estimate 95th percentile dist is 1000 m ~= 0.6 miles
W_od = 0.82 
# '''See: Purpose-Based Walking Trips by Duration, Distance, and Select Characteristics, 2017 National Household Travel Survey'''
# Calculation: (mean walking time + 2 SE) * (avg walking speed). 

# Other papers: 
# 1) Active-transport walking behavior: destinations, durations, distances, 
# 2) Exploring built environment correlates of walking distance of transit egress in the Twin Cities
# 3) Distances people walk for transport, which is a meta-analysis

# # Define which edge types are associated with each mode type
modes_to_edge_type = {'pt': ['board','pt','alight'], 'z': ['z','park'], 'tnc':['t','t_wait'], 'walk':['w'], 'sc': ['sc'], 'bs': ['bs'], 'mt':['mt']}

# scooter simulation
NUM_OBS = 1500 * (2/3)  # per movepgh report
NUM_DAYS_OF_DATA = 30   # simulate as if we had 30 days of data

# # time factors
# TIME_START = 7 * 3600 # (must be in the form of seconds_after_midnight)
# TIME_END = 9 * 3600 #   (must be in the form of seconds_after_midnight)
# INTERVAL_SPACING = 10 # sec
# NUM_INTERVALS = int((TIME_END - TIME_START) / INTERVAL_SPACING)
# INRIX_SPACING = 300 #  seconds (5 min*60 sec/min) ; how often are measurements taken with inrix data

INCONVENIENCE_COST = 2 # minutes

# speed
WALK_SPEED = 1.3 # m/s
SCOOT_SPEED = 2.78 # m/s
BIKE_SPEED = 14.5 / 3600 * 1000 # m/s
TNC_WAIT_TIME = 6 # minutes
ALIGHTING_TIME = 5 # seconds

# reliability
BOARDING_RELIABILITY = 1.5
TNC_WAIT_RELIABILITY = 2

# PRICE_PARAMS = {
#     'w': {'ppmin': 0, 'ppmile':0, 'fixed': 0},  
#     'sc': {'ppmin': 0.39, 'ppmile':0, 'fixed': 0},  
#     'sc_tx': {'ppmin': 0, 'ppmile':0, 'fixed': 1},
#     'bs': {'ppmin': 25/200, 'ppmile':0, 'fixed': 0},
#     't': {'ppmin': 0.19, 'ppmile': 1.12, 'fixed': 0}, 
#     't_wait': {'ppmin':0, 'ppmile':0, 'fixed': 3.03 + 2.64 + 1},  # fixed price is: base fare + "booking fee" + $1 minfare buffer
#     'board': {'ppmin':0, 'ppmile':0, 'fixed': 2.75},
#     'alight': {'ppmin':0, 'ppmile':0, 'fixed': 0},
#     'pt': {'ppmin':0, 'ppmile':0, 'fixed': 0},
#     'rt': {'ppmin':0, 'ppmile':0, 'fixed': 0},
#     'pb': {'ppmin': 0, 'ppmile':0, 'fixed': 0},
#     'z': {'ppmin': 11/60, 'ppmile':0, 'fixed': 0, 'fixed_per_month': 9, 'est_num_trips': 4},
#     'pv': {'ppmin':0, 'ppmile': 0.20, 'fixed': 0},
#     'park': {'ppmin':0, 'ppmile':0, 'fixed':2.50 * 8} 
# }

RISK_CRASH_IDX = {
    'w':0.28,
    'sc':1.81,
    'bs':1.81,
    't':1,
    't_wait':0,
    'board':0.19,
    'alight':0.19,
    'pt':0.19,
    'pb':1.81,
    'z':1,
    'pv':1,
    'park':1
}

DISCOMFORT_PARAMS = {
    'w': 2.86/1.34,
    'sc': 3.26/1.34,  # 
    'bs': 3.26/1.34,
    't': 1.34/1.34,  # use vehicle as baseline
    't_wait': 0,
    'board': 2.22/1.34,  # could change if thinking about cold weather conditions
    'alight': 0,
    'pt': 2.22/1.34,
    'pb': 3.26/1.34,
    'z': 1.34/1.34,
    'pv': 1.34/1.34,
    'park': 1.34/1.34
}

