import networkx as nx 
import numpy as np
import re
import pickle

from nomad import utils
from nomad import conf

def nn(dist_to_all_nodes, nid_map, travel_mode):
    '''Find the nearest neighbor of a given node, when provided the distance matrix from that node to all other nodes.
       Return the nearest neighbor id, its name, and the distance between the node of interest and nearest neighbor.
    '''
    nid_map_travel_mode = [key for key,val in nid_map.items() if val.startswith(travel_mode)]   # this is a list of IDs
    # subset dist matrix for the nodes in the component network of the travel mode of interest
    dist_subset = dist_to_all_nodes[ nid_map_travel_mode]
    # # find the node in the component network of interest that is nearest to the input node of interest
    # #print(dist_subset)
    nn_dist = np.amin(dist_subset)
    nn_idx = np.argmin(dist_subset)
    # now map back to the original node ID
    original_nn_id = nid_map_travel_mode[nn_idx]
    original_nn_name = nid_map[original_nn_id]
    return (original_nn_id, original_nn_name, nn_dist)

class Supernetwork:
    def __init__(self, unimodal_graphs, fix_pre, flex_pre):
        '''Union all the unimodal graphs. Define which nodes are fixed and flex for purpose of creating transfer edges in add_transfer_edges().'''
        self.networks = unimodal_graphs
        self.graph = nx.union_all(unimodal_graphs)
        self.fix_pre = fix_pre  # which *nodes* are fixed in the supernetwork
        self.flex_pre = flex_pre   # which *nodes* are flex in the supernewtork
    
    def save_graph(self, output_path):
        with open(output_path, 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
          
    def print_mode_types(self):
        '''Print the graphs that are incuded in the network.'''
        print(self.networks)
    
    def add_coord_matrix(self):
        '''Add the matrix that contains the coordinates of each node in the supernetwork.'''
        coords_dict = nx.get_node_attributes(self.graph, 'pos')
        nid_map = dict(zip(range(len(coords_dict.keys())), list(coords_dict.keys())))  # idx2name
        coord_matrix = np.array(list(coords_dict.values()))
        #self.coord_dict= coords_dict
        self.nid_map = nid_map
        self.coord_matrix = coord_matrix
        #return (nid_map, coord_matrix)
    
    def add_gcd_dist_matrix(self):
        '''Add the matrix that contains the great circle distance between all pairs of nodes.'''
        # gcd distance matrix: gcd_dist[i,j] is the great circle distance from node i to node j
        gcd_dist = np.empty([len(self.nid_map), len(self.nid_map)], dtype='float16')  # great circle dist in meters 
        for i in range(gcd_dist.shape[0]):
            dist_to_all_nodes = utils.calc_great_circle_dist(self.coord_matrix[i], self.coord_matrix)  # calc distance from node i to all other nodes
            gcd_dist[i] = dist_to_all_nodes  # calc_great_circle_dist is a function defined above 
        self.gcd_dist = gcd_dist
        #return gcd_dist
        
    def separate_nidmap_fix_flex(self):
        '''Separate the node ID map by fixed nodes and flexible nodes.'''
        self.nid_map_fixed = {key:val for key,val in self.nid_map.items() if utils.mode(val) in self.fix_pre}
        self.nid_map_flex = {key:val for key,val in self.nid_map.items() if utils.mode(val) in self.flex_pre}
    
    def define_pmx(self, pmx):
        '''Define the permitted mode changes within the supernetwork.'''
        self.pmx = pmx
       
    def add_transfer_edges(self, W):
        '''See: Algorithm 1, Graff et al. (2024).'''
        # Generate scooter transfer data, assuming real data unavailable
        sc_costs = utils.gen_data(self)
    
        etype = 'transfer'
        trans_edges = {}
        
        # Build transfer edges
        for i in list(self.nid_map_fixed.keys()):
            #attrs = {}
            i_name = self.nid_map[i]  # map back to node name
            catch = utils.wcz(i, self.gcd_dist, W)  # catchment zone around i (includes both fixed and flex nodes)
            # build fixed-fixed transfer edge
            for j in catch:
                if j in self.nid_map_fixed.keys():
                    j_name = self.nid_map[j]  # map back to node name
                    if (utils.mode(i_name), utils.mode(j_name)) in self.pmx:         # if mode switch allowed between i and j
                        # build the transfer edge
                        edge = (i_name, j_name)
                        # find the walking time associated with transfer edge, call it walk_cost
                        #walk_time = self.gcd_dist[i,j] / conf.config_data['Speed_Params']['walk']   # (seconds)  # dist[m] / speed [m/s] / 60 s/min  --> [min]
                        #wait_time = 0
                        # TO DO: account for a no-cost public transit transfer in node-movement cost file
                        fixed_price = 0   # already account for PT fixed cost in the boarding edges              
                        attr_dict = {'length_m':self.gcd_dist[i,j], 'mode_type':'w', 'etype':etype} 
                        trans_edges[edge] = attr_dict

            # now build: transfers from fixed-flex or flex-fixed                          
            # but first, remove 'pv' nodes to prevent arbitary transfers to the pv network
            # 'pv' nodes are only considered flexible for the sake of OD connectors  
            flex_pre_tx = self.flex_pre.copy()
            if 'pv' in self.flex_pre:
                flex_pre_tx.remove('pv') 
            
            # Steps: 
            # 1) find the nearest neighbor in each flex network
            # 2) build transfer from fixed node to the nn in the flex network, if tx is permitted
            # 3) build reverse transfer from nn in the flex network to the fixed node, if tx is permitted
            # note: TNC is a special case b/c we will also build waiting edge e.g. bsd --> t_virtual --> t_physical (walking + waiting)
            # note: scooter is alsp a special case b/c we have already generated (simulated) scoooter transfer data
            if flex_pre_tx:  # check that list is not NoneType
                for m in flex_pre_tx:    
                    if ((utils.mode(i_name), m) in self.pmx) | ((m, utils.mode(i_name)) in self.pmx):    # if mode switch allowed between i and m
                        #print(i, m)
                        nnID, nnName, nnDist = utils.nn(i, self.gcd_dist, m, self.nid_map)  # tuple in the form of (node_id, node_name, dist)
                        if nnID in catch:  # still the rule holds where you can only transfer if within WCZ
                            k_name = nnName
                            edge_in = (i_name, k_name)                
                            edge_out = (k_name, i_name)
                            # walk_time = nnDist / conf.config_data['Speed_Params']['walk']  # (seconds)       dist[m] / speed [m/s] 
                            # TODO: do we want to add a "wait time" associated with scooter/bikeshare unlocking? that seems like almost too granular though
                            # consider: add fixed price (approx) of zipcar? 
                            attr_dict = {'length_m':nnDist, 'mode_type':'w', 'etype':etype} 

                            # special cases: TNC and scooter
                            if m == 't':
                                # transfer TO tnc: walk_edge + wait_edge
                                if (utils.mode(i_name), utils.mode(k_name)) in self.pmx:
                                    # add virtual TNC node to the graph. TODO: later we will add to the nidmap
                                    t_virtual = 'tw' + re.sub(r'[a-zA-Z]', '', k_name)
                                    self.graph.add_node(t_virtual, node_type='tw', pos=self.graph.nodes[k_name]['pos'],
                                                        nwk_type = 't')
                                    # add virtual waiting edge
                                    t_wait_edge = (t_virtual, k_name)
                                    t_wait_attr_dict = {'mode_type':'t_wait', 'etype':etype}
                                    trans_edges[t_wait_edge] = t_wait_attr_dict
                                    trans_edges[(i_name,t_virtual)] = attr_dict # transfer (walking) edge
                                # when transferring FROM tnc, use nearest neighbor distance b/c ride tnc to closest pickup point for next mode
                                if (utils.mode(k_name), utils.mode(i_name)) in self.pmx:  
                                    trans_edges[edge_out] = attr_dict 
                            elif m == 'sc':
                                if (utils.mode(i_name), utils.mode(k_name)) in self.pmx:
                                    scoot_attr_dict = sc_costs[i_name]
                                    trans_edges[edge_in] = scoot_attr_dict # when transferring TO scooter, assign costs that were generated (simulated) using fake historical data      
                                # when transferring FROM scooter, use nearest neighbor distance b/c ride scooter to closest pickup point for next mode
                                if (utils.mode(k_name), utils.mode(i_name)) in self.pmx:  
                                    trans_edges[edge_out] = attr_dict 
                            else:
                                if (utils.mode(i_name), utils.mode(k_name)) in self.pmx:
                                    trans_edges[edge_in] = attr_dict
                                if (utils.mode(k_name), utils.mode(i_name)) in self.pmx:  
                                    trans_edges[edge_out] = attr_dict 
        self.transfer_edges = trans_edges  # for testing purposes so we can access the tx edges directly
        # now add the transfer edges to the supernetwork
        trans_edges = [(e[0], e[1], trans_edges[e])for e in trans_edges.keys()]
        self.graph.add_edges_from(trans_edges)

    def add_od_nodes(self, org_nodes, dst_nodes):
        '''Add origin/destination nodes to the graph of the supernetwork object in place. Each o/d is labeled by its index in the gdf.
           Function get_node_idx2geo_dict() is subsequently used to relable index to census ID
        '''
        # first add all origin nodes
        for i, o_coord in enumerate(list(org_nodes)):
            self.graph.add_nodes_from([('org'+str(i), {'pos': tuple(o_coord), 'nwk_type':'od', 'node_type':'od'})]) # add the org nodes to the graph along with their positions 
            self.nid_map[max(self.nid_map.keys())+1] = 'org'+str(i) # add them to the nid_map
        # then add all destination nodes 
        for i, d_coord in enumerate(list(dst_nodes)):
            self.graph.add_nodes_from([('dst'+str(i), {'pos': tuple(d_coord), 'nwk_type':'od', 'node_type':'od'})]) # add the org nodes to the graph along with their positions 
            self.nid_map[max(self.nid_map.keys())+1] = 'dst'+str(i) # add them to the nid_map

    def add_org_cnx(self, org_coords):
        '''Add origin connection edges to the graph of the supernetwork object.'''
        nid_map = self.nid_map
        coord_matrix = self.coord_matrix  # twait nodes are not added to nid map *at this stage*. nor are the orgs
        
        # First generate all scooter data (assuming true data is unavailable)
        sc_costs = utils.gen_data(self, od_cnx=True) 
        
        # Add the org connectors
        for i, o_coord in enumerate(list(org_coords)):   # o_coords is an n x 2 numpy array
            org_cnx_edges = {} # dict whose key is the org cnx edge and value is the edge's attribute dict
            dist_to_all_nodes = utils.calc_great_circle_dist(np.array(o_coord), coord_matrix) # dist from org to modal graph nodes
            i_name = 'org' + str(i)
            W_od_cnx = conf.W_od_cnx * conf.MILE_TO_METERS
            catch = np.where(dist_to_all_nodes <= W_od_cnx)[0].tolist()

            #print('-----')
            # build od connector edge for each FIXED node in the catchment zone 
            for j in catch:
                if j in self.nid_map_fixed.keys():
                    j_name = self.nid_map[j]  # map back to node name
                    #print(j_name, mode(j_name))
                    if ((utils.mode(j_name) == 'k') | (utils.mode(j_name) == 'kz')):   
                        continue  # exceptions 2 and 5
                    edge = (i_name, j_name)  # build org connector
                    #walk_time = (dist_to_all_nodes[j] / conf.config_data['Speed_Params']['walk'])  # walking traversal time [sec] of edge
                    #wait_time = 0 
                    # note that wait time is not included. this is because we're dealing with fixed modes. only TNC has wait time
                    # wait time for PT is embedded in alighting edges
                    attr_dict = {'length_m':dist_to_all_nodes[j], 'mode_type':'w', 'etype':'od_cnx'}
                    org_cnx_edges[edge] = attr_dict
                    
                    # some fixed modes do not have a node within the wcz. for these modes, we will instead connect the 
                    # org to the nearest neighbor node of for these specific fixed modes. consider this like relaxing the wcz constraint
                    catch_node_names = [self.nid_map[c] for c in catch]
                    catch_fixed_modes = [re.sub(r'[^a-zA-Z]', '', cname) for cname in catch_node_names]
                    #print(catch_fixed_modes)
                    
                    # # which fixed mode does not have a node in the wcz?
                    # rem_fixed_modes = set(self.fix_pre) - set(catch_fixed_modes)
                    # if rem_fixed_modes:
                    #     for rm in rem_fixed_modes:
                    #         if ((rm == 'k') | (rm == 'kz')):   # exceptions 2/5
                    #             continue
                    #         # nn calc
                    #         nnID, nnName, nnDist = nn(dist_to_all_nodes, nid_map, rm)     
                    #         r_name = nnName
                    #         #cnx_edge_length = nnDist
                    #         #walk_time = cnx_edge_length / conf.config_data['Speed_Params']['walk']  # [sec]
                    #         wait_time = 0
                    #         attr_dict = {'length_m':nnDist, 'mode_type':'w', 'etype':'od_cnx'}
                    #         edge = (i_name, r_name)  # build org connector
                    #         org_cnx_edges[edge] = attr_dict

                # build od connector edge for the nearest flexible node (relax constraints that needs to be in wcz)
                # also includes an org connector from org to nearest PV node, butils does NOT include a dst connector from dst to nearest PV node
                for m in self.flex_pre:
                    nnID, nnName, nnDist = nn(dist_to_all_nodes, nid_map, m) 
                    k_name = nnName
                    attr_dict = {'length_m':nnDist, 'mode_type':'w', 'etype':'od_cnx'}         
                    # do this separately for scooters/TNCs and other flex modes. butils have not yet generated TNC data to account for variable pickup wait times
                    if m == 't':
                        # build edge org -- t_wait -- t
                        # add virtual TNC node to the graph. TODO: later we will add to the nid map
                        t_virtual = 'tw' + re.sub(r'[a-zA-Z]', '', k_name)
                        self.graph.add_node(t_virtual, node_type='tw', pos=self.graph.nodes[k_name]['pos'], nwk_type = 't')
                        # add virtual waiting edge
                        t_wait_edge = (t_virtual, k_name)
                        t_wait_attr_dict = {'mode_type':'t_wait', 'etype':'od_cnx'}
                        org_cnx_edges[t_wait_edge] = t_wait_attr_dict  # waiting edge
                        org_cnx_edges[(i_name,t_virtual)] = attr_dict  # transfer (walking) edge 
                    elif m == 'sc':
                        # add edge org -- sc and check the scooter cost dict 
                        edge = (i_name, k_name)
                        sc_cost_dict = sc_costs[i_name]
                        sc_cost_dict['etype'] = 'od_cnx'   
                        org_cnx_edges[edge] = sc_cost_dict
                    else:
                        # build i to k
                        org_cnx_edges[(i_name, k_name)] = attr_dict 
            #org_cnx_edges_all = org_cnx_edges_all | org_cnx_edges
            org_cnx_edges = [(e[0], e[1], org_cnx_edges[e]) for e in org_cnx_edges.keys()] # convert org_cnx_edges to proper form so that they can be added to graph
            self.graph.add_edges_from(org_cnx_edges) # TODO: test with 3 origins


    def add_dst_cnx(self, dst_coords):  
        '''Add destination connection edges to the graph of the supernetwork object''' 
        nid_map = self.nid_map
        #inv_nid_map = dict(zip(self.nid_map.values(), self.nid_map.keys())) 

        coord_matrix = self.coord_matrix  # twait nodes are not added to nid map *at this stage*. nor are the dsts (??)
        #dst_cnx_edges_all = {}
        # add the dest connectors
        for i, d_coord in enumerate(list(dst_coords)):   # d_coords is an n x 2 numpy array
            dst_cnx_edges = {}
            dist_to_all_nodes = utils.calc_great_circle_dist(np.array(d_coord), coord_matrix) # dist from org to modal graph node
            i_name = 'dst' + str(i)
            W_od_cnx = conf.W_od_cnx * conf.MILE_TO_METERS 
            catch = np.where(dist_to_all_nodes <= W_od_cnx)[0].tolist()

            #print('-----')
            # build od connector edge for each FIXED node in the catchment zone 
            for j in catch:
                if j in self.nid_map_fixed.keys():
                    j_name = self.nid_map[j]  # map back to node name
                    #print(j_name, mode(j_name))
                    if (utils.mode(j_name) == 'zd'):   
                        continue  # exceptions 2 and 5
                    edge = (j_name, i_name)  # build dst connector
                    attr_dict = {'length_m':dist_to_all_nodes[j], 'mode_type':'w', 'etype':'od_cnx'}
                    dst_cnx_edges[edge] = attr_dict
                    
                    # some fixed modes do not have a node within the wcz. for these modes, we will instead connect the 
                    # org to the nearest neighbor node of for these specific fixed modes. consider this like relaxing the wcz constraint
                    catch_node_names = [self.nid_map[c] for c in catch]
                    catch_fixed_modes = [re.sub(r'[^a-zA-Z]', '', cname) for cname in catch_node_names]
                    #print(catch_fixed_modes)
                    
                    # # which fixed mode does not have a node in the wcz?
                    # rem_fixed_modes = set(self.fix_pre) - set(catch_fixed_modes)
                    # if rem_fixed_modes:
                    #     for rm in rem_fixed_modes:
                    #         if (rm == 'zd'):   
                    #             continue
                    #         # nn calc
                    #         nnID, nnName, nnDist = nn(dist_to_all_nodes, nid_map, rm) 
                    #         r_name = nnName
                    #         #cnx_edge_length = nnDist
                    #         #walk_time = cnx_edge_length / conf.config_data['Speed_Params']['walk']  # [sec]
                    #         wait_time = 0
                    #         attr_dict = {'length_m':nnDist, 'mode_type':'w', 'etype':'od_cnx'}
                    #         edge = (r_name, i_name)  # build org connector
                    #         dst_cnx_edges[edge] = attr_dict

                # build od connector edge for the nearest flexible node (relax constraints that needs to be in wcz)
                # does NOT include a dst connector from dst to nearest PV node
                for m in self.flex_pre:
                    if m == 'pv':  # exception 3
                        continue
                    nnID, nnName, nnDist = nn(dist_to_all_nodes, nid_map, m) 
                    k_name = nnName
                    attr_dict = {'length_m':nnDist, 'mode_type':'w', 'etype':'od_cnx'}         
                    # do this separately for scooters/TNCs and other flex modes. butils have not yet generated TNC data to account for variable pickup wait times
                    if m == 't':
                        # build dst -- t
                        edge = (k_name, i_name)
                        dst_cnx_edges[edge] = attr_dict 
                    elif m == 'sc':
                        # build dst - sc
                        edge = (k_name, i_name)
                        dst_cnx_edges[edge] = attr_dict 
                    else:
                        # build i to k
                        dst_cnx_edges[(k_name, i_name)] = attr_dict 
            #dst_cnx_edges_all = dst_cnx_edges_all | dst_cnx_edges
            dst_cnx_edges = [(e[0], e[1], dst_cnx_edges[e]) for e in dst_cnx_edges.keys()] # convert od_cnx_edges to proper form so that they can be added to graph    
            self.graph.add_edges_from(dst_cnx_edges) # TODO: test that this works

    def add_direct_od_cnx(self, org_coords, dst_coords):
        '''Add direct edge from org-dst to the graph of the  supernetwork object if the distance bewtween them is less than W_od.'''
        od_dist_matrix = np.zeros((len(org_coords), len(dst_coords)))
        for i in range(len(org_coords)):
            od_dist_matrix[i,:] = utils.calc_great_circle_dist(np.array(org_coords)[i], np.array(dst_coords))
        # check: for each o-d pair, is their distance less than W_od? if so, build transfer directly
        allowed_od_transfer = np.argwhere((od_dist_matrix * conf.CIRCUITY_FACTOR / conf.MILE_TO_METERS) <= conf.W_od)
        od_cnx_edges = []
        for o,d in allowed_od_transfer:
            attr_dict = {'length_m': od_dist_matrix[o,d], 'mode_type':'w', 'etype':'od_cnx'} 
            od_cnx_edges.append(('org'+str(o), 'dst'+str(d), attr_dict)) 
        self.graph.add_edges_from(od_cnx_edges)

    def add_twait_nodes(self):
        '''Add t_wait nodes to the node ID (nid) map in place since the nodes have been added to the graph but not to the nid map.'''
        tw_nodes = [n for n in list(self.graph.nodes) if n.startswith('tw')]
        for tw in tw_nodes:
            self.nid_map[max(self.nid_map.keys())+1] = tw

    @classmethod
    def from_graphs_dict(cls, all_graphs_dict, modes_included):
        # this dict defines which graphs correspond to each mode type 
        #all_graphs_dict = {'t':G_tnc, 'pv':G_pv, 'pb':G_pb, 'bs':G_bs, 'pt':G_pt, 'sc':G_sc, 'z':G_z}
        
        # Dict that defines the node prefixes corresponding to each mode type 
        all_modes_nodes = {'bs':['bs', 'bsd'], 'pt':['ps','rt'], 't':['t'], 'sc':['sc'], 'pv':['pv','k'], 'pb':['pb'], 'z':['zd','z','kz']}
        
        # Define which nodes are fixed and which come from flexible networks 
        all_fix_pre = ['bsd','ps','k', 'zd', 'kz']  
        all_flex_pre = ['t', 'pb', 'pv', 'sc'] 
        
        fix_pre_included = [n for m in modes_included for n in all_modes_nodes[m] if n in all_fix_pre]
        flex_pre_included = [n for m in modes_included for n in all_modes_nodes[m] if n in all_flex_pre]
        
        # this dict defines which modes and nodes are included in the supernetwork
        #modes_nodes_included = {k:v for k,v in all_modes_nodes.items() if k in modes_included}
        
        graphs_included = [all_graphs_dict[m] for m in modes_included]  
        
        # Permitted mode changes
        pmx = [('ps','ps'),('bsd','ps'),('ps','bsd'),('ps','t'),('t','ps'),('t','bsd'),('bsd','t'), ('k','ps'),('k','t'),('k','bsd'),('ps','pb'),
               ('pb','ps'),('ps','sc'),('sc','ps'),('k','sc'), ('bsd','sc'), ('sc','bsd'), ('ps','zd'), ('bsd','zd'), ('t','zd'), ('sc','zd'),
               ('kz','ps'),('kz','t'),('kz','bsd'),('kz','sc')]  
        
        # Initialize the network
        G_sn = cls(graphs_included, fix_pre_included, flex_pre_included)
        G_sn.print_mode_types()
        G_sn.add_coord_matrix()
        G_sn.add_gcd_dist_matrix()
        G_sn.separate_nidmap_fix_flex()
        G_sn.define_pmx(pmx)
        
        # Add transfer edges
        W_tx = conf.W_tx * conf.MILE_TO_METERS
        G_sn.add_transfer_edges(W_tx)
        
        print('supernetwork built')      
        #G_sn.save_object(output_path)

        return G_sn