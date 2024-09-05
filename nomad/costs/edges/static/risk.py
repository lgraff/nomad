import statsmodels.api as sm
import pandas as pd

from nomad import conf

def assign_edge_risk(df_G, df_cost):
    # establish risk (predicted crashes) using the crash model calibrated in streets.py
    crash_model = sm.load(conf.crash_model_path)
    df_risk_calc = df_G.copy()
    df_risk_calc.rename(columns={'speed_lim':'SPEED', 'length_m':'length_meters'}, inplace=True) # b/c these are the precise names of the var names in the crash model
    # assumption that crash risk along tnc waiting, boarding, and alighting edges is effectively zero. can be adjusted
    df_risk_calc.loc[df_risk_calc['edge_type'].isin(['t_wait','board','alight']), ['SPEED','length_meters','frc']] = [0,0,4]
    df_risk_calc.loc[df_G['pred_crash'].isna(),'pred_crash'] = crash_model.predict(df_risk_calc)
    df_cost = df_cost.merge(df_risk_calc[['source','target','mode_type','pred_crash']], how='left', on=['source','target','mode_type'])
    return df_cost