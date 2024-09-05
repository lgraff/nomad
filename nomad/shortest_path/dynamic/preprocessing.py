
import numpy as np

import MNMAPI
from nomad import conf
from nomad import costs
from nomad import shortest_path as sp
from nomad.costs.nodes import dynamic

def edit_config(folder, graph_name, num_rows_link_file, num_rows_node_file):
    with open(folder / 'config.conf', 'w') as f:
        f.write('[Network] \n')
        f.write('network_name = ' + graph_name + '\n')
        f.write('num_of_link = ' + str(num_rows_link_file) + '\n')
        f.write('num_of_node = ' + str(num_rows_node_file) + '\n')

def prepare_graph_file(tdsp_folder, G_sn):
    # Prepare files for compatiblity with MAC-POSTS
    ### Create graph topology file
    df_G = costs.edges.nx_to_df(G_sn).sort_values(by=['source','target','mode_type']).reset_index(drop=True)
    df_G.rename(columns={'source':'source_named', 'target':'target_named'}, inplace=True)
    nid_map = get_nid_map(df_G)
    inv_nid_map = dict(zip(nid_map.values(), nid_map.keys()))
    df_G[['source','target']] = df_G[['source_named','target_named']].applymap(lambda x: inv_nid_map[x]).reset_index(drop=True)
    df_G.insert(0, 'linkID', df_G.index)  # add link ID

    filename = 'graph'
    np.savetxt(tdsp_folder / filename, df_G[['linkID','source','target']].values, fmt='%d', delimiter=' ')
    # add a header EdgeId	FromNodeId	ToNodeId
    f = open(tdsp_folder / filename, 'r')
    log = f.readlines()
    f.close()
    log.insert(0, 'EdgeId	FromNodeId	ToNodeId\n')
    f = open(tdsp_folder / filename, 'w')
    f.writelines(log)
    f.close() 

    ### Create linkID map and its inverse
    # linkID_map = dict(zip(df_G['linkID'], tuple(zip((df_G['source']),df_G['target']))  ))
    # inv_linkID_map = dict(zip(linkID_map.values(), linkID_map.keys()))
    return df_G #, inv_linkID_map

def get_nid_map(df_G):
    node_set = sorted(list(set(df_G['source_named']).union(set(df_G['target_named']))))
    nid_map = dict(zip(range(len(node_set)), node_set))
    return nid_map

def get_link_id_map(df_G):
    linkID_map = dict(zip(df_G['linkID'], tuple(zip((df_G['source_named']),df_G['target_named']))  ))
    return linkID_map

def prepare_node_files(tdsp_folder, df_node_cost_dynamic):
    ### Create node cost array
    INTERVAL_SPACING = conf.INTERVAL_SPACING
    NUM_INTERVALS = conf.NUM_INTERVALS
    interval_columns = [f'i{i}' for i in range(NUM_INTERVALS)]
    td_node_cost = df_node_cost_dynamic[['node_id_via','linkID_in','linkID_out']+interval_columns].values    # nodeID, inLinkID, outLinkID, cost
    
    # save node cost as plain txt
    filename = 'td_node_cost'
    np.savetxt(tdsp_folder / filename, td_node_cost, fmt='%d %d %d ' + (NUM_INTERVALS-1)*'%f ' + '%f')
    f = open(tdsp_folder / filename, 'r')
    log = f.readlines()
    f.close()
    log.insert(0, "node_ID in_link_ID out_link_ID td_cost\n")
    f = open(tdsp_folder / filename, 'w')
    f.writelines(log)
    f.close()

    # save node tt as plain txt
    # This file stores the time-dependent movement-based node travel time cost. It is a requirement of DOT algorithm in MAC-POSTS. For our purposes, it has just one row of zeros.'''   
    filename = 'td_node_tt'
    num_rows = td_node_cost.shape[0]
    zeros_columns = np.zeros((num_rows, NUM_INTERVALS))
    td_node_tt = np.concatenate((td_node_cost[:,[0,1,2]], zeros_columns), axis=1)
    np.savetxt(tdsp_folder / filename, td_node_tt, fmt='%d %d %d ' + (NUM_INTERVALS-1)*'%f ' + '%f')
    f = open(tdsp_folder / filename, 'r')
    log = f.readlines()
    f.close()
    log.insert(0, 'node_ID in_link_ID out_link_ID td_tt\n')
    f = open(tdsp_folder / filename, 'w')
    f.writelines(log)
    f.close()

def prepare_tt_file(tdsp_folder, linkID_arr, df_tt_dynamic):
    ### Create time-dep (td) travel time (tt) array 
    #td_link_tt = np.hstack((linkID_arr, td_link_tt))
    INTERVAL_SPACING = conf.INTERVAL_SPACING
    NUM_INTERVALS = conf.NUM_INTERVALS
    interval_columns = [f'i{i}' for i in range(NUM_INTERVALS)]
    td_link_tt = np.around(df_tt_dynamic[interval_columns].values.astype('float') / INTERVAL_SPACING)
    # replace zeros with ones (because zeros can mess up TDSP computation)
    td_link_tt[td_link_tt == 0] = 1
    td_link_tt = np.hstack((linkID_arr, td_link_tt))
    # save as plain txt
    filename = 'td_link_tt'
    np.savetxt(tdsp_folder / filename, td_link_tt, fmt='%d ' + (NUM_INTERVALS-1)*'%f ' + '%f')
    f = open(tdsp_folder / filename, 'r')
    log = f.readlines()
    f.close()
    log.insert(0, 'link_ID td_tt\n')
    f = open(tdsp_folder / filename, 'w')
    f.writelines(log)
    f.close()

def prepare_gtc_file(tdsp_folder, filename, linkID_arr, gtc_arr):
    ### Create time-dep (td) link cost array
    #filename = 'td_link_cost'
    NUM_INTERVALS = conf.NUM_INTERVALS

    ##linkID_arr = df_G['linkID'].to_numpy().reshape((-1,1))
    td_link_cost = np.hstack((linkID_arr, gtc_arr))
    # save as plain txt
    ##filename = 'td_link_cost'
    np.savetxt(tdsp_folder / filename, td_link_cost, fmt='%d ' + (NUM_INTERVALS-1)*'%f ' + '%f')
    f = open(tdsp_folder / filename, 'r')
    log = f.readlines()
    f.close()
    log.insert(0, 'link_ID td_cost\n')
    f = open(tdsp_folder / filename, 'w')
    f.writelines(log)
    f.close()

def prepare_tdsp_api(G, BETAS, tdsp_folder, link_cost_filename):
    '''Inputs: graph, beta parameters, tdsp_folder to store data, link cost filename.
       Output: tdsp_api, mapping dict from node ID (key) to node name (value)'''
    # Parameters
    NUM_INTERVALS = conf.NUM_INTERVALS  
    interval_columns = [f'i{i}' for i in range(NUM_INTERVALS)]

    # Get graph
    # Get all time-dep edge costs
    df_tt_dynamic, df_rel_dynamic, df_price_dynamic, df_risk_dynamic, df_disc_dynamic = costs.edges.dynamic.assign_edge_costs(G)
    # Get time-dep node costs
    df_node_cost_dynamic = costs.nodes.dynamic.get_node_cost_df(G, conf.NUM_INTERVALS)

    # Prepare tdsp files and get link and node IDs
    df_G = prepare_graph_file(tdsp_folder, G)
    nid_map = get_nid_map(df_G)
    inv_nid_map = dict(zip(nid_map.values(), nid_map.keys()))
    linkID_map = get_link_id_map(df_G)
    inv_linkID_map = dict(zip(linkID_map.values(), linkID_map.keys()))
    
    ### Create tt file 
    linkID_arr = df_G['linkID'].to_numpy().reshape((-1,1))
    prepare_tt_file(tdsp_folder, linkID_arr, df_tt_dynamic)
    
    ### Create node files
    df_node_cost_dynamic['node_id_via'] = df_node_cost_dynamic['node_via'].map(lambda x: inv_nid_map[x]) 
    df_node_cost_dynamic['link_in'] = tuple(zip(df_node_cost_dynamic['node_from'], df_node_cost_dynamic['node_via']))
    df_node_cost_dynamic['link_out'] = tuple(zip(df_node_cost_dynamic['node_via'], df_node_cost_dynamic['node_to']))
    df_node_cost_dynamic[['linkID_in', 'linkID_out']] = df_node_cost_dynamic[['link_in', 'link_out']].applymap(lambda x: inv_linkID_map[x])
    sp.prepare_node_files(tdsp_folder, df_node_cost_dynamic)
    
    ### Prepare gtc file
    gtc_arr = BETAS['tt'] * df_tt_dynamic[interval_columns].values.astype(np.float16) + BETAS['rel'] * df_rel_dynamic[interval_columns].values.astype(np.float16) + BETAS['x'] * df_price_dynamic[interval_columns].values.astype(np.float16) + BETAS['risk'] * df_risk_dynamic[interval_columns].values.astype(np.float16) + BETAS['disc'] * df_disc_dynamic[interval_columns].values.astype(np.float16)
    prepare_gtc_file(tdsp_folder, link_cost_filename, linkID_arr, gtc_arr)
    
    ### Edit the config file
    num_rows_link_file = len(df_G)
    num_rows_node_file = len(df_node_cost_dynamic)
    edit_config(tdsp_folder, 'graph', num_rows_link_file, num_rows_node_file)

    # Invoke TDSP api from mac-posts
    tdsp_api = MNMAPI.tdsp_api()
    tdsp_api.initialize(str(tdsp_folder), NUM_INTERVALS, len(df_G), len(df_node_cost_dynamic))
    tdsp_api.read_td_cost_txt(str(tdsp_folder), 'td_link_tt', 'td_node_cost', link_cost_filename, 'td_node_cost')

    print('TDSP api has successfully read the files')

    return tdsp_api