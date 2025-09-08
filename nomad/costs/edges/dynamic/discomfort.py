
import numpy as np
import pandas as pd

# Assume discomfort is constant for all departure times. So we extend pred_crash by the number of intervals
def assign_edge_discomfort(config, df_G):
    disc_params = config['discomfort_params']
    df_disc = df_G.copy()
    NUM_INTERVALS = config['time_factors']['NUM_INTERVALS']
    df_disc['discomfort'] = df_disc.apply(lambda row: row['length_m']/1000 * disc_params[row['mode_type']], axis=1)

    disc_arr = np.repeat(df_disc['discomfort'].values.reshape(-1,1), repeats=NUM_INTERVALS, axis=1)
    df_intervals = pd.DataFrame(disc_arr, columns=[f'i{i}' for i in range(NUM_INTERVALS)], dtype='float16')
    df_edge_info = df_disc[['source','target','mode_type']]
    df_disc_dynamic = pd.concat([df_edge_info, df_intervals], axis=1)
    df_disc_dynamic.sort_values(by=['source','target','mode_type'], inplace=True)

    return df_disc_dynamic
