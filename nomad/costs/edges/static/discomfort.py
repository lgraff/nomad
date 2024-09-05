from nomad import conf

def assign_edge_discomfort(df_cost):
    '''Assign discomfort to each edge. Return df, keyed by edge, with discomfort time as an attribute.'''
    # define: segment-level discomfort = segment_discomfort_idx * segment_length (km)
    disc_idxs = conf.DISCOMFORT_PARAMS
    df_cost['discomfort'] = df_cost.apply(lambda x: x['length_m']/1000*disc_idxs[x['mode_type']], axis=1)
    return df_cost