import statsmodels.api as sm
import pandas as pd

from nomad import conf

def assign_edge_risk(df_G, df_cost, config):
    '''Establish risk (predicted crashes) using the crash model calibrated in streets.py.'''

    # Load the crash model
    crash_model = sm.load(config['paths']['data']['crash_model'])

    # Ensure column names match with covariate names from the crash model
    df_risk_calc = df_G.copy()
    df_risk_calc.rename(columns={'speed_lim':'SPEED', 'length_m':'length_meters'}, inplace=True) # b/c these are the precise names of the var names in the crash model

    # Assume some parameters for tnc waiting, boarding, and alighting edges. Can be adusted. These parameters will ensure that crash risk along these edges is effectively zero. 
    df_risk_calc.loc[df_risk_calc['mode_type'].isin(['t_wait','board','alight']), ['SPEED','length_meters','frc']] = [0,0,4]

    # Use crash model to predict 2-year crash risk (we used two years of data to calibrate regression). 
    mask = df_risk_calc['pred_crash'].isna()
    df_risk_calc.loc[mask,'pred_crash'] = crash_model.predict(df_risk_calc[mask]) 

    # If we define risk as predicted number of crashes per day, then we make the calc: pred_crash / (num_years * days_in_year)
    days_in_year = 365
    num_years = 2 
    df_risk_calc['pred_crash'] = df_risk_calc['pred_crash'] / (days_in_year * num_years)

    # Adjust by risk index and CMF
    risk_crash_index = config['risk_crash_idx']
    CMF = config['CMF']
    df_risk_calc['pred_crash'] = df_risk_calc.apply(lambda row: row['pred_crash'] * risk_crash_index[row['mode_type']], axis=1)
    df_risk_calc['pred_crash'] = df_risk_calc['pred_crash'] * df_risk_calc['bikeway_type'].apply(lambda x: CMF.get(x, 1))

    df_cost = df_cost.merge(df_risk_calc[['source','target','mode_type','pred_crash']], how='left', on=['source','target','mode_type'])

    return df_cost