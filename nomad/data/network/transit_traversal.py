#%% libraries
import partridge as ptg
import pandas as pd
import numpy as np
import os

def calc_great_circle_dist(lat1, lat2, lon1, lon2, earth_radius=6371009):
    y1 = np.deg2rad(lat1)  # y is latitude 
    y2 = np.deg2rad(lat2)
    dy = y2 - y1

    x1 = np.deg2rad(lon1)
    x2 = np.deg2rad(lon2)
    dx = x2 - x1

    h = np.sin(dy / 2) ** 2 + np.cos(y1) * np.cos(y2) * np.sin(dx / 2) ** 2
    h = np.minimum(1, h)  # protect against floating point errors
    arc = 2 * np.arcsin(np.sqrt(h))

    # return distance in units of earth_radius
    return arc * earth_radius

def create_PT_traversal(GTFS_filepath, time_start, time_end, traversal_outpath):

    # time_start = 7*3600 # seconds start after midnight
    # time_end = 9*3600 # seconds end after midnight
    # Partridge example: https://gist.github.com/invisiblefunnel/6015e65684325281e65fa9339a78229b

    view = {"trips.txt": {}}

    # Choose a date
    view["trips.txt"]["service_id"] = ptg.read_busiest_date(GTFS_filepath)[1]
    # Build a working feed (all route-dir pairs). Load the feed once here
    feed = ptg.load_feed(GTFS_filepath, view)
    trips_interval = feed.stop_times[
        (feed.stop_times.arrival_time >= time_start - 1*60*60) 
        & (feed.stop_times.arrival_time <= time_end + 1*60*60) 
        ].trip_id.unique().tolist()  # list of trip_ids in the time interval

    # Find the traversal time between stops for a single representative trip, based on scheduled departure time of the stop
    trips = feed.trips[['trip_id', 'route_id', 'direction_id']]
    # Filter trips df by trips_interval
    trips = trips[trips.trip_id.isin(trips_interval)]

    trips_stoptimes = pd.merge(trips, feed.stop_times, on='trip_id', how='inner')[
        ['trip_id', 'route_id','direction_id','stop_sequence','stop_id','departure_time']]
    trips_stoptimes = trips_stoptimes.sort_values(by='trip_id').groupby(
        by=['route_id', 'direction_id', 'stop_sequence']).first().reset_index()  # .first() indicates we selected first trip as representative

    # drop duplicates by route_id, direction_id, and stop_id triple i.e. focus on one trip
    trips_stoptimes = trips_stoptimes.sort_values(by=['route_id', 'direction_id', 'stop_id', 'departure_time'], ascending=True)
    trips_stoptimes.drop_duplicates(subset=['route_id','direction_id','stop_id'], inplace=True)

    # trips_stoptimes = trips_stoptimes.sort_values(by='trip_id').groupby(
    #     by=['route_id', 'direction_id', 'stop_sequence']).first().reset_index()
    trips_stoptimes.sort_values(by=['route_id', 'direction_id', 'stop_sequence'], ascending=True, inplace=True)
    trips_stoptimes = trips_stoptimes[trips_stoptimes['route_id'] != '28X'] # domain knowledge. 28X allows pickups/dropoffs only heading to/leaving airport

    traversal_time_sec = trips_stoptimes.groupby(['route_id', 'direction_id', 'trip_id'])['departure_time'].diff()  # seconds
    trips_stoptimes['traversal_time_sec'] = traversal_time_sec

    # Calculate the length of each segment
    trips_stoptimes = pd.merge(trips_stoptimes, feed.stops[['stop_id', 'stop_lat','stop_lon']], on='stop_id', how='inner').sort_values(by=['route_id','direction_id','stop_sequence'])
    trips_stoptimes['lat_prev'] = trips_stoptimes['stop_lat'].shift(1)
    trips_stoptimes['lon_prev'] = trips_stoptimes['stop_lon'].shift(1)
    trips_stoptimes['length_m'] = calc_great_circle_dist(trips_stoptimes['stop_lat'], trips_stoptimes['lat_prev'], trips_stoptimes['stop_lon'], trips_stoptimes['lon_prev'])

    trips_stoptimes.to_csv(traversal_outpath, index=False)   
