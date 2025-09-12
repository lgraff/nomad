
def extend_historical_travel_time_data(config, historical_travel_time_df, ratio_name, time_start, time_end, interval_spacing):
    '''Returns df_ext, which stores the travel time (tt) ratio relative to the first departure time in the interval on the level of frc.
       e.g., if the first departure time is 7am, then the tt ratio at 7:05am for frc=2 represents the avg ratio of travel times between 7:05am and 7am for all roads with frc=2.'''
    sec_after_midnight = np.arange(time_start, time_end, interval_spacing)
    HISTORICAL_TRAVEL_TIME_OBSERVATION_SPACING = config['time_factors']['HISTORICAL_TRAVEL_TIME_OBSERVATION_SPACING']
    df_ext = pd.DataFrame(columns=['frc','sec_after_midnight', ratio_name])
    for frc in [2,3,4]:
        df = historical_travel_time_df[historical_travel_time_df['frc'] == frc]
        ratio_rep = np.repeat(df[ratio_name].values, repeats=HISTORICAL_TRAVEL_TIME_OBSERVATION_SPACING/interval_spacing, axis=0)
        frc_arr = frc*np.ones((len(sec_after_midnight)))
        df_ext = pd.concat([df_ext, pd.DataFrame(np.column_stack((frc_arr, sec_after_midnight, ratio_rep)), columns=['frc','sec_after_midnight', ratio_name])], ignore_index=True)
    df_ext['frc'] = df_ext['frc'].astype('int')
    return df_ext

from .travel_time import *
from .reliability import *
from .price import *
from .risk import *
from .discomfort import *
from .get_costs import *