
import pandas as pd

from nomad import conf

def assign_edge_price(df_tt_dynamic):
    '''The price of an edge is the sum of its usage (mileage- + minute-based fees) + any fixed fees.'''
    price_params = conf.PRICE_PARAMS

    # Do calculations in numpy
    fee_per_min_arr = df_tt_dynamic.apply(lambda x: price_params[x['mode_type']]['ppmin'], axis=1).values.reshape(-1,1)
    fee_per_mile_arr = df_tt_dynamic.apply(lambda x: price_params[x['mode_type']]['ppmile'], axis=1).values.reshape(-1,1)
    fixed_fee_arr = df_tt_dynamic.apply(lambda x: price_params[x['mode_type']]['fixed'], axis=1).values.reshape(-1,1)
    length_arr = df_tt_dynamic['length_m'].values.reshape(-1,1)

    tt_cols = [col for col in df_tt_dynamic.columns if col.startswith('i')]
    tt_arr = df_tt_dynamic[tt_cols].values

    total_fee_arr = fixed_fee_arr + (fee_per_min_arr * tt_arr / 60) + (fee_per_mile_arr * length_arr / conf.MILE_TO_METERS)

    interval_colnames = [f'i{i}' for i in range(total_fee_arr.shape[1])]
    df_intervals_price = pd.DataFrame(total_fee_arr, columns=interval_colnames)
    df_edge_info = df_tt_dynamic[['source','target','mode_type','length_m']].reset_index(drop=True)
    df_price_dynamic = pd.concat([df_edge_info, df_intervals_price], axis=1)

    # special case, fix manually: scooter transfer
    df_price_dynamic.loc[((df_price_dynamic['mode_type'] == 'w') & (df_price_dynamic['target'].str.startswith('sc'))), interval_colnames] = conf.PRICE_PARAMS['sc_tx']['fixed']

    # in lieu of adding node costs to prevent consecutive walking segments, which is compuationally constrained on this machine, add 2.75 for ps-ps
    # df_price_dynamic.loc[((df_price_dynamic['source'].str.startswith('ps')) & (df_price_dynamic['target'].str.startswith('ps'))), interval_colnames] = conf.PRICE_PARAMS['board']['fixed']

    df_price_dynamic.sort_values(by=['source','target','mode_type'], inplace=True)
    
    return df_price_dynamic