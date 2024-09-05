import pickle

from nomad import conf
from nomad import costs


def assign_edge_costs(G_sn):
    df_G = costs.edges.nx_to_df(G_sn)

    df_tt_dynamic = costs.edges.dynamic.assign_edge_travel_time(df_G, conf.TIME_START, conf.TIME_END, conf.INTERVAL_SPACING)
    df_rel_dynamic = costs.edges.dynamic.assign_edge_reliability(df_tt_dynamic, conf.TIME_START, conf.TIME_END, conf.INTERVAL_SPACING)  # derived from travel time
    df_price_dynamic = costs.edges.dynamic.assign_edge_price(df_tt_dynamic)  # derived from travel time
    df_risk_dynamic = costs.edges.dynamic.assign_edge_risk(df_G)  
    df_disc_dynamic = costs.edges.dynamic.assign_edge_discomfort(df_G)

    return df_tt_dynamic, df_rel_dynamic, df_price_dynamic, df_risk_dynamic, df_disc_dynamic

# def assign_edge_gtc()