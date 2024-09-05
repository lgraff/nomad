from nomad import conf


def calc_price(fixed_price, price_per_min, price_per_mile, num_secs, num_meters):
    num_miles = num_meters / 1609        # 1609 meters in a mile
    num_min = num_secs / 60              # 60 seconds in a minute
    price_total = fixed_price + (price_per_min * num_min) + (price_per_mile * num_miles)
    return price_total

def assign_edge_price(df_cost):
    price_params = conf.PRICE_PARAMS
    df_cost['price'] = df_cost.apply(lambda x: calc_price(price_params[x['mode_type']]['fixed'], 
                                                          price_params[x['mode_type']]['ppmin'],
                                                          price_params[x['mode_type']]['ppmile'],
                                                          x['avg_tt_sec'],
                                                          x['length_m']),
                                                          axis=1)
    # manual adjustment for scooter transfer edges: replace the scooter transfer price with $1 to embed fixed cost of scooter into transfer
    sc_tx = (df_cost['mode_type'] == 'w') & (df_cost['target'].str.startswith('sc'))
    df_cost.loc[sc_tx, 'price'] = price_params['sc_tx']['fixed']
    return df_cost


# def calc_usage_fee(fee_per_min, fee_per_mile, num_secs, num_meters):
#     num_miles = num_meters / 1609        # 1609 meters in a mile
#     num_min = num_secs / 60              # 60 seconds in a minute
#     usage_fee = (fee_per_min * num_min) + (fee_per_mile * num_miles)
#     return usage_fee

# def calc_total_fee(fixed_fee, usage_fee):
#     total_fee = fixed_fee + usage_fee
#     return total_fee
