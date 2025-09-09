
import networkx as nx
import logging
import heapq
import numpy as np
import itertools


def dijkstra(G, node_costs, weight_name, source, target):
    num_nodes = len(G.nodes)
    dist = [float('inf')] * num_nodes  # Use a simple Python list for distances
    pred = [None] * num_nodes  # Predecessor list
    dist[source] = 0

    # Priority queue: stores (distance, node)
    pq = [(0, source)]

    while pq:
        d, u = heapq.heappop(pq)
        
        # If the node distance is greater than the known shortest distance, skip it
        if d > dist[u]:
            continue
        
        # Early exit if we reach the target
        if u == target:
            break

        for v in G.neighbors(u):
            # Calculate node cost (with a default of 0)
            node_cost = node_costs.get((pred[u], u, v), 0)
            # Calculate new distance
            new_dist_v = dist[u] + G[u][v][weight_name] + node_cost  # (previous dist to u) + (dist u-v) + (dist q-u-v via u)
            if new_dist_v < dist[v]:
                dist[v] = new_dist_v
                pred[v] = u
                heapq.heappush(pq, (new_dist_v, v))

    return dist, pred

# Example Usage
# shortest_path, total_cost = bidirectional_dijkstra(G_idx, node_cost_idx, 'GTC', source, target)
# print(shortest_path, total_cost)

# def run_shortest_path(G_idx, node_cost_idx, weight_name, source, target):
#     dist, pred = sp.dijkstra(G_idx, node_cost_idx, weight_name, source, target)
#     shortest_path = sp.extract_shortest_path(pred, target)
#     total_gtc = np.nan if len(shortest_path) == 1 else dist[target][0]
#     return (shortest_path, total_gtc)

def dijkstra_digraph_one_to_all(G, node_costs, weight_name, source):

    # Initialize distances and predecessors
    dist = {node: float('inf') for node in G.nodes}
    pred = {node: None for node in G.nodes}

    dist[source] = 0
    pq = [(0, source)]  # Priority queue with (distance, node)
    visited = set()

    while pq:
        d, u = heapq.heappop(pq)

        if u in visited:
            continue

        for v in G.neighbors(u):
            edge_cost = G[u][v].get(weight_name, float('inf'))
            node_cost = node_costs.get(v, 0)  # Additional node cost if available
            new_dist = dist[u] + edge_cost + node_cost

            if new_dist < dist[v]:
                dist[v] = new_dist
                pred[v] = u
                heapq.heappush(pq, (new_dist, v))

        visited.add(u)

    return dist, pred


def reconstruct_path_single(dist, pred, target, node_costs):
    """Reconstruct the path from predecessors."""
    path = []
    node = target
    visited = set()  # Set to track visited nodes and detect cycles

    while node is not None:
        if node in visited:
            print(f"Cycle detected in path at node {node}")
            return None  # Return None to indicate an error or failure
        visited.add(node)
        path.append(node)
        node = pred[node]

    path.reverse()

    # Adjust the total cost by adding the node costs at the end
    min_cost = dist[target]
    total_cost = min_cost  
    for node in path:
        # Add node cost (from (pred, node, node) tuple)
        total_cost += node_costs.get((pred[node], node, node), 0)

    return path, total_cost

def bidirectional_dijkstra_multidigraph(G, node_costs, weight_name, source, target):
    """
    Bidirectional Dijkstra's Algorithm for shortest path in a MultiDiGraph.
    
    Parameters:
    - G: NetworkX MultiDiGraph.
    - node_costs: Dictionary of additional node costs (optional).
    - weight_name: Name of the edge weight attribute.
    - source: Source node.
    - target: Target node.
    
    Returns:
    - shortest_path: List of nodes representing the shortest path.
    - total_cost: Total generalized travel cost (GTC) of the shortest path.
    """

    num_nodes = len(G.nodes)

    # Initialize distances and predecessors
    dist_f = [float('inf')] * num_nodes
    dist_b = [float('inf')] * num_nodes
    pred_f = [None] * num_nodes
    pred_b = [None] * num_nodes

    dist_f[source] = 0
    dist_b[target] = 0

    pq_f = [(0, source)]
    pq_b = [(0, target)]

    visited_f = set()
    visited_b = set()

    meeting_node = None
    min_cost = float('inf')

    while pq_f and pq_b:
        # Forward search
        if pq_f:
            d_f, u_f = heapq.heappop(pq_f)
            if u_f in visited_f:
                continue
            visited_f.add(u_f)

            for v_f in G.neighbors(u_f):
                min_edge_cost = min(edge_data[weight_name] for edge_data in G[u_f][v_f].values())
                #node_cost = node_costs.get((pred_f[u_f], u_f, v_f), 0) if pred_f[u_f] else 0
                new_dist_f = dist_f[u_f] + min_edge_cost # + node_cost


                if new_dist_f < dist_f[v_f]:
                    dist_f[v_f] = new_dist_f
                    pred_f[v_f] = u_f
                    heapq.heappush(pq_f, (new_dist_f, v_f))

                if v_f in visited_b:
                    total_cost = dist_f[v_f] + dist_b[v_f]
                    if total_cost < min_cost:
                        min_cost = total_cost
                        meeting_node = v_f

        # Backward search
        if pq_b:
            d_b, u_b = heapq.heappop(pq_b)
            if u_b in visited_b:
                continue
            visited_b.add(u_b)

            for v_b in G.predecessors(u_b):  # Backward uses predecessors
                min_edge_cost = min(edge_data[weight_name] for edge_data in G[v_b][u_b].values())
                #node_cost = node_costs.get((pred_b[u_b], u_b, v_b), 0) if pred_b[u_b] else 0
                new_dist_b = dist_b[u_b] + min_edge_cost # + node_cost

                if new_dist_b < dist_b[v_b]:
                    dist_b[v_b] = new_dist_b
                    pred_b[v_b] = u_b
                    heapq.heappush(pq_b, (new_dist_b, v_b))

                if v_b in visited_f:
                    total_cost = dist_f[v_b] + dist_b[v_b]
                    if total_cost < min_cost:
                        min_cost = total_cost
                        meeting_node = v_b

        # Stop if the frontiers meet
        if meeting_node is not None:
            break

    if meeting_node is None:
        print("No path found.")
        return None, float('nan')

    # Reconstruct path
    shortest_path = reconstruct_path_bidirectional(pred_f, pred_b, meeting_node)
    
    # Adjust the total cost by adding the node costs at the end
    total_cost = min_cost  # Start with the total cost from the bidirectional search
    for node in shortest_path:
        # Add node cost (from (pred, node, node) tuple)
        total_cost += node_costs.get((pred_f[node], node, node), 0)

    return shortest_path, min_cost


def validate_predecessors(pred):
    for idx, value in enumerate(pred):
        if value is not None and value == idx:
            print(f"Self-loop detected at node {idx}")
        if value is not None and pred[value] == idx:
            print(f"Cycle detected: {idx} -> {value} -> {idx}")


def reconstruct_path_bidirectional(pred_f, pred_b, meeting_node):
    # Reconstruct forward path
    forward_path = []
    node = meeting_node
    visited_f = set()  # Set to track visited nodes in the forward path

    while node is not None:
        if node in visited_f:
            print(f"Cycle detected in forward path at node {node}")
            return None  # Return None to indicate an error or failure
        visited_f.add(node)
        forward_path.append(node)
        node = pred_f[node]
    
    forward_path.reverse()

    # Reconstruct backward path
    backward_path = []
    node = meeting_node
    visited_b = set()  # Set to track visited nodes in the backward path

    while node is not None:
        if node in visited_b:
            print(f"Cycle detected in backward path at node {node}")
            return None  # Return None to indicate an error or failure
        visited_b.add(node)
        backward_path.append(node)
        node = pred_b[node]

    # Merge paths, avoid naming the meeting node twice
    return forward_path + backward_path[1:]


# def run_shortest_path_bidirectional(G_idx, node_cost_idx, weight_name, source, target):
#     '''Call bidirectional_dijkstra.'''
#     shortest_path, min_cost = bidirectional_dijkstra_multidigraph(G_idx, node_cost_idx, weight_name, source, target)
#     return shortest_path, min_cost

