from nomad import costs
from nomad import conf

def get_edge_cost_df(G_sn, hour, minute):
    '''Add costs to the edges for the hour:minute departure time, adding all costs incrementally.'''
    # Get pandas df from nx supernetwork
    #df_G = costs.nx_to_df(sn_path) 

    df_G = costs.edges.nx_to_df(G_sn)

    # Assign edge costs sequentially
    df_tt = costs.assign_edge_travel_time(df_G, hr=hour, minute=minute)
    df_edge_costs_sn = costs.edges.static.assign_edge_reliability(df_G, df_tt, hr=hour, minute=minute)  # reliability is, for the most part, derived from avg tt
    df_edge_costs_sn = costs.edges.staic.assign_edge_price(df_edge_costs_sn) # price is derived from avg tt and length
    df_edge_costs_sn = costs.edges.static.assign_edge_risk(df_G, df_edge_costs_sn)
    df_edge_costs_sn = costs.edges.static.assign_edge_discomfort(df_edge_costs_sn)
    betas = conf.BETAS
    # Get GTC
    df_edge_costs_sn.loc[:,'GTC'] = betas['tt']*df_edge_costs_sn['avg_tt_sec'] + betas['rel']*df_edge_costs_sn['reliability'] + betas['x']*df_edge_costs_sn['price'] + betas['risk']*df_edge_costs_sn['pred_crash'] + betas['disc']*df_edge_costs_sn['discomfort']
    df_edge_costs_sn['edge'] = tuple(zip(df_edge_costs_sn.source, df_edge_costs_sn.target))
    return df_edge_costs_sn