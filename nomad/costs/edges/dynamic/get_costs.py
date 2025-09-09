
from nomad import costs
from nomad import utils

def assign_edge_costs(G_sn):
    config = G_sn.config
    
    df_G = utils.nx_to_df(G_sn)

    df_tt_dynamic = costs.edges.dynamic.assign_edge_travel_time(config, df_G)
    df_rel_dynamic = costs.edges.dynamic.assign_edge_reliability(config, df_tt_dynamic)  # derived from travel time
    df_price_dynamic = costs.edges.dynamic.assign_edge_price(config, df_tt_dynamic)  # derived from travel time
    df_risk_dynamic = costs.edges.dynamic.assign_edge_risk(config, df_G)  
    df_disc_dynamic = costs.edges.dynamic.assign_edge_discomfort(config, df_G)

    return df_tt_dynamic, df_rel_dynamic, df_price_dynamic, df_risk_dynamic, df_disc_dynamic

# def assign_edge_gtc()