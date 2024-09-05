
import pandas as pd
import numpy as np
import partridge as ptg

def create_PT_headway_static(GTFS_filepath, headway_outpath, time_start, time_end):
    # Arguments: time_start and time end must be given in seconds start after midnight
    # Return average headway for dpearture times between time_start and time_end.

    # Partridge example: https://gist.github.com/invisiblefunnel/6015e65684325281e65fa9339a78229b

    view = {"trips.txt": {}}

    # Choose a date
    view["trips.txt"]["service_id"] = ptg.read_busiest_date(GTFS_filepath)[1]
    # Build a working feed (all route-dir pairs). Load the feed once here
    feed = ptg.load_feed(GTFS_filepath, view)

    df_trips_stops = feed.stop_times.merge(feed.trips, how='inner', on='trip_id')
    # Find trips overlapping the time window
    # buffer the end time by 1 hr so we can still calculate the headway for traveler arrivals at stops at the end of the study period
    df_trips_stops = df_trips_stops[df_trips_stops['departure_time'].between(time_start, time_end + 1*60*60)]

    # **Use the following to get average headway within the time interval, if analysis not dependent**
    df_routes_stops = df_trips_stops[['route_id','direction_id','stop_id','arrival_time','departure_time']].sort_values(by=['route_id','direction_id','stop_id','arrival_time','departure_time'])
    headway_col = df_routes_stops.groupby(['route_id','direction_id','stop_id'])[['departure_time']].diff()
    df_headway = pd.concat([df_routes_stops[['route_id','direction_id','stop_id']], headway_col], axis=1).groupby(['route_id','direction_id','stop_id']).mean().reset_index()
    df_headway.columns = ['route_id', 'direction_id', 'stop_id', 'headway_mean']
    # if headway_mean is na, that indicates that the stop is not served more than once during the full interval
    # consequently, we set the headway as the full length of the interval
    df_headway.loc[df_headway['headway_mean'].isna(), 'headway_mean'] = time_end - time_start

    df_headway.to_csv(headway_outpath, index=False)

def create_PT_headway_dynamic(GTFS_filepath, headway_outpath, time_start, time_end, interval_spacing):
    # Arguments: time_start and time end must be given in seconds start after midnight
    # Return headway for departure times every interval_spacing seconds

    # Partridge example: https://gist.github.com/invisiblefunnel/6015e65684325281e65fa9339a78229b

    view = {"trips.txt": {}}

    # Choose a date
    view["trips.txt"]["service_id"] = ptg.read_busiest_date(GTFS_filepath)[1]
    # Build a working feed (all route-dir pairs). Load the feed once here
    feed = ptg.load_feed(GTFS_filepath, view)

    all_routes = feed.trips.route_id.unique()    # all routes
    all_dirs = feed.trips.direction_id.unique()  # all directions

    df_trips_stops = feed.stop_times.merge(feed.trips, how='inner', on='trip_id')
    # Find trips overlapping the time window
    # buffer the end time by 1 hr so we can still calculate the headway for traveler arrivals at stops at the end of the study period
    df_trips_stops = df_trips_stops[df_trips_stops['departure_time'].between(time_start - 1*60*60, time_end + 1*60*60)]
    # get dept times every [interval_spacing] seconds
    all_arrival_times_at_stop = [sec for sec in range(time_start, time_end, interval_spacing)]
    # routes = ['61C', '64']  # for testing
    df_headway_rows = []
    for r in all_routes: 
        for d in all_dirs: 
            # Get the stop_ids associated with the route-dir pair
            route_dir_condition = (df_trips_stops['route_id'] == r) & (df_trips_stops['direction_id'] == d)
            stop_ids = df_trips_stops[route_dir_condition].stop_id.unique().tolist()

            # Get headway for every stop
            for stop in stop_ids:
                condition = (df_trips_stops['route_id'] == r) & (df_trips_stops['direction_id'] == d) & (df_trips_stops['stop_id'] == stop)
                stop_times_filtered = df_trips_stops[condition][['route_id', 'direction_id', 'trip_id', 'stop_id', 'arrival_time', 'departure_time']].sort_values(by='departure_time', ascending=True)
                # Account for different traveler arrival times at the stop
                for a in all_arrival_times_at_stop:
                    stop_dep_times = stop_times_filtered.departure_time.unique()  # departure times for the stop 
                    try:
                        next_departure = np.sort(stop_dep_times[stop_dep_times>=a])[0]
                        headway = next_departure - a  # in seconds
                    except:
                        headway = 1*60*60  # headway is at least one hour
                    #print(a,next_departure)
                    row = [r, d, stop, a, headway]
                    df_headway_rows.append(row)

    df_headway = pd.DataFrame(df_headway_rows, columns=['route_id', 'direction_id', 'stop_id', 'traveler_arrival_time', 'headway']).sort_values(by=['route_id','direction_id','stop_id','traveler_arrival_time'])

    df_headway.to_csv(headway_outpath, index=False)
