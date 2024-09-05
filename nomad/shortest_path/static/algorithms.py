
import networkx as nx
import logging
import heapq
import numpy as np
import itertools

from nomad import conf
from nomad import shortest_path as sp

# for bellman speedup, see: https://arxiv.org/pdf/1111.5414.pdf
def bellman_ford(nodes, edges, node_costs, source, weight_name):
    # Initialization
    dist = np.ones((len(nodes),1)) * np.inf
    pred = [None] * len(nodes)
    dist[source] = 0
    C = {source}
    while len(C) > 0:
        dist_prev = dist.copy()  # keep track of prior iteration's distances
        for u in C:
            #print('iteration')
            # Relax edges
            for (u,v) in edges.keys():
                # first check if a node cost applies
                node_cost = node_costs[(pred[u],u,v) if (pred[u],u,v) in node_costs.keys() else 0]
                # then calculate new distance
                # print(u,v)
                # print(dist[u])
                # print(edges[(u,v)]['weight']) #[weight_name]])
                # print(node_cost)
                new_dist_v = dist[u] + edges[(u,v)][weight_name] + node_cost # (previous dist to u) + (dist u-v) + (dist q-u-v via u)
                if new_dist_v < dist[v]:
                    dist[v] = new_dist_v
                    pred[v] = u
        # for which vertices did D[v] change?
        C = set(np.where(dist_prev != dist)[0])
    return dist, pred

def get_neighbors(G, node):
    neighbors = [n for n in G.neighbors(node)]
    return neighbors

def dijkstra(G, node_costs, weight_name, source, target):
    #edges = dict(G.edges)
    #nodes = list(G.nodes)

    # Initialization
    num_nodes = len(list(G.nodes))
    dist = np.ones((num_nodes,1)) * np.inf
    pred = [None] * num_nodes
    dist[source] = 0

    # create priority queue
    pq = []
    heapq.heappush(pq, (dist[source], source))

    while len(pq) > 0:
        d, u = heapq.heappop(pq)
        if u == target:
            break
        neighbors_u = get_neighbors(G, u)

        # Relax edges of neighbors of u
        for v in neighbors_u:
            # first check if a node cost applies
            node_cost = node_costs[(pred[u],u,v)] if (pred[u],u,v) in node_costs.keys() else 0
            # then calculate new distance
            new_dist_v = dist[u] + G.edges[u,v][weight_name] + node_cost # (previous dist to u) + (dist u-v) + (dist q-u-v via u)
            if new_dist_v < dist[v]:
                dist[v] = new_dist_v
                pred[v] = u
                heapq.heappush(pq, (dist[v], v))
    return dist, pred

def extract_shortest_path(pred, dest):
    current = dest
    path = []
    while current != None:
        path.append(current)
        current = pred[current]
    return(path[::-1])

def run_shortest_path(G_idx, node_cost_idx, weight_name, source, target):
    dist, pred = sp.dijkstra(G_idx, node_cost_idx, weight_name, source, target)
    shortest_path = sp.extract_shortest_path(pred, target)
    total_gtc = np.nan if len(shortest_path) == 1 else dist[target][0]
    return (shortest_path, total_gtc)
