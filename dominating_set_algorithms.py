# Minimum Connected Dominating Set
import networkx as nx

def MCDS(G):
  u = max(G.degree(), key=lambda x: x[1])[0]
  covered = [u]
  cds = []

  while len(covered) < G.number_of_nodes():
    candidates = list(set(covered) - set(cds))
    r = None
    max_neighbors = -1

    for v in candidates:
      neighbors = [n for n in G.neighbors(v) if n not in covered]
      num_neighbors = len(neighbors)

      if num_neighbors > max_neighbors:
        max_neighbors = num_neighbors
        r = v

    cds.append(r)

    for n in G.neighbors(r):
      if n not in covered:
        covered.append(n)

  return cds


# Connected m-Dominating Set

def CmDS(G, m):
  sorted_nodes = sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True)
  C = []

  for i in range(G.number_of_nodes()):
    N_c = len([v for v in G.neighbors(sorted_nodes[i]) if v in C])
    if N_c < m:
      C.append(sorted_nodes[i])

  dominatees = [v for v in sorted_nodes if v not in C]
  while not nx.is_connected(nx.subgraph(G, C)):
    C.append(dominatees[0])
    dominatees = dominatees[1:]

  return C

def Optimization(G, C, m):
  C_sub = nx.subgraph(G, C)
  C_sorted_nodes = sorted(C_sub.nodes(), key=lambda x: G.degree(x))

  for i in range(len(C)):
    N_c = len([v for v in G.neighbors(C_sorted_nodes[i]) if v in C])
    if N_c >= m:
      dominatees_i = [v for v in G.neighbors(C_sorted_nodes[i]) if v not in C]
      N_j = [len([v for v in G.neighbors(j) if v in C]) - 1 for j in dominatees_i]
      if len(N_j) > 0 and min(N_j) >= m:
        C.remove(C_sorted_nodes[i])

  return C

def dominating_set(G, m, minimal=False, optimize = False):
  if minimal == True:
    ds = MCDS(G)
  else:
    ds = CmDS(G, m)
    if optimize:
      ds = Optimization(G, ds, m)
  
  return ds

def CmDS_fast(G, m):
    # Sort exactly as original
    sorted_nodes = sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True)
    
    # We maintain a list for ordering, and a set for O(1) lookups
    C_list = []
    C_set = set()
    
    for node in sorted_nodes:
        # Generator sum is faster than list comprehension allocation
        N_c = sum(1 for v in G.neighbors(node) if v in C_set)
        if N_c < m:
            C_list.append(node)
            C_set.add(node)

    dominatees = [v for v in sorted_nodes if v not in C_set]
    
    idx = 0
    # nx.is_connected works perfectly with sets, no need to convert to list
    while not nx.is_connected(G.subgraph(C_set)):
        new_node = dominatees[idx]
        C_list.append(new_node)
        C_set.add(new_node)
        idx += 1 # Avoids the O(N) slicing of dominatees[1:]

    return C_list

def Optimization_fast(G, C_list, m):
    C_set = set(C_list)
    C_sorted_nodes = sorted(C_list, key=lambda x: G.degree(x))

    for node in C_sorted_nodes:
        # 1. Node itself must have >= m neighbors in C
        N_c = sum(1 for v in G.neighbors(node) if v in C_set)
        
        if N_c >= m:
            dominatees_i = [v for v in G.neighbors(node) if v not in C_set]
            
            # 2. Enforce the original `len(N_j) > 0` condition
            if len(dominatees_i) > 0:
                
                # 3. Check if min(N_j) >= m, but with early exit for speed
                can_remove = True
                for j in dominatees_i:
                    # Count neighbors of j in C
                    count_j_in_C = sum(1 for v in G.neighbors(j) if v in C_set)
                    
                    # Original logic: min(N_j) >= m where N_j = count - 1
                    if (count_j_in_C - 1) < m:
                        can_remove = False
                        break # Fails condition, skip checking the rest of dominatees
                        
                if can_remove:
                    C_set.remove(node)

    # Return as a list, preserving the order of the nodes that survived
    return [x for x in C_list if x in C_set]

def dominating_set_fast(G, m, minimal=False, optimize = False):
  if minimal == True:
    ds = MCDS(G)
  else:
    ds = CmDS_fast(G, m)
    if optimize:
      ds = Optimization_fast(G, ds, m)
  
  return ds