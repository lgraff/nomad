import statsmodels.api as sm
import pandas as pd
import numpy as np

from nomad import conf
from nomad import costs

# We established the predicted crash risk when we built the network model in /data/network/streets.py
# However, a few edges remain becuase they were built subsequently: bikeshare cnx, walk, t_wait, board, and alight

def assign_edge_risk(df_G):
    # Establish risk (predicted crashes) using the crash model calibrated in streets.py
    crash_model = sm.load(conf.crash_model_path)
    df_risk = df_G.copy()
    df_risk.rename(columns={'speed_lim':'SPEED', 'length_m':'length_meters'}, inplace=True) # b/c these are the precise names of the var names in the crash model
    
    # Assumption that crash risk along tnc waiting, boarding, and alighting edges is zero. can be adjusted
    df_risk.loc[df_risk['mode_type'].isin(['t_wait','board','alight']), ['pred_crash']] = 0   
    
    # For the remaining edges, use the crash model
    df_risk.loc[df_risk['pred_crash'].isna(),'pred_crash'] = crash_model.predict(df_risk)
    df_risk = df_risk[['source','target','mode_type','pred_crash']]
    
    # Assume risk is constant for all departure times. So we extend pred_crash by the number of intervals
    NUM_INTERVALS = conf.NUM_INTERVALS
    # pred_crash is the predicted number of crashes in a two-year period (we used two years of data to calibrate regression). 
    # if we define risk as predicted number of crashes per day, then we make the calc: pred_crash / (num_years * days_in_year)
    days_in_year = 365
    num_years = 2 
    risk_arr = np.repeat(df_risk['pred_crash'].values.reshape(-1,1), repeats=NUM_INTERVALS, axis=1)
    df_intervals = pd.DataFrame(risk_arr, columns=[f'i{i}' for i in range(NUM_INTERVALS)], dtype='float16') / (num_years * days_in_year)
    df_edge_info = df_risk[['source','target','mode_type']]
    df_risk_dynamic = pd.concat([df_edge_info, df_intervals], axis=1)
    df_risk_dynamic.sort_values(by=['source','target','mode_type'], inplace=True)
    
    return df_risk_dynamic