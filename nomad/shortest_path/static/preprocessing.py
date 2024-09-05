import itertools
import networkx as nx

from nomad import conf

def get_cost_subsets(mode_subset, df_edge_cost, df_node_cost):
    '''Create edge and node cost subsets, only inclusive of the edges and nodes corresponding to the selected mode combination.
       Return: a subset of the edge cost df; a subset of the node cost df.'''
    # Edge cost subset
    edges_include = list(itertools.chain(*[conf.modes_to_edge_type[m] for m in mode_subset])) + ['w']
    df_edge_cost_subset = df_edge_cost[df_edge_cost['mode_type'].isin(edges_include)].copy() # filter the df_edge_cost_subset by the included modes/edges
    df_edge_cost_subset['edge'] = tuple(zip(df_edge_cost_subset.source, df_edge_cost_subset.target))  # add the edge as a tuple
    # Node cost subset
    nodes = set(df_edge_cost_subset.source.unique().tolist() + df_edge_cost_subset.target.unique().tolist())  # node set
    df_node_cost_subset = df_node_cost[((df_node_cost['node_from'].isin(nodes)) & (df_node_cost['node_via'].isin(nodes)) & (df_node_cost['node_to'].isin(nodes)))]

    return (df_edge_cost_subset, df_node_cost_subset) #, name2idx)

def get_node_idx_map(df_edge_cost_subset):
    '''Convert node name to node index (numerical)
       Return a map from node name to node index.'''
    nodes = set(df_edge_cost_subset.source.unique().tolist() + df_edge_cost_subset.target.unique().tolist())  # node set
    name2idx = dict(zip(nodes, range(len(nodes)))) # make map from node name to index
    return name2idx    

def get_G_idx(df_edge_cost_subset, name2idx):
    '''Create graph subset, only inclusive of the edges corresponding to the selected mode combination.
       Return: a new graph, G_idx, where all nodes are numerical.'''
    df_edge_cost_subset.loc[:,'source_idx'] = df_edge_cost_subset.loc[:,'source'].map(lambda x: name2idx[x])
    df_edge_cost_subset.loc[:,'target_idx'] = df_edge_cost_subset.loc[:,'target'].map(lambda x: name2idx[x])
    G_idx = nx.from_pandas_edgelist(df_edge_cost_subset, source='source_idx', target='target_idx', edge_attr='GTC', create_using=nx.DiGraph)
    G_idx.remove_nodes_from(list(nx.isolates(G_idx))) # remove isolated nodes (those without neighbors) - do we need this step?
    
    return G_idx

def get_node_cost_idx(df_node_cost_subset, name2idx):
    '''Get node costs in index form.
       Return a dict of the form {(node_from, node_via, node_to): node_cost}.'''
    df_node_cost_subset.loc[:,['node_from','node_via','node_to']] = df_node_cost_subset[['node_from','node_via','node_to']].applymap(lambda x: name2idx[x])
    node_cost_dict_keys = tuple(zip(df_node_cost_subset.node_from, df_node_cost_subset.node_via, df_node_cost_subset.node_to))
    node_cost_dict_vals = df_node_cost_subset.cost
    node_cost_idx = dict(zip(node_cost_dict_keys, node_cost_dict_vals))
    return node_cost_idx