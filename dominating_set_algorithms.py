# Minimum Connected Dominating Set
import networkx

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
  while not networkx.is_connected(networkx.subgraph(G, C)):
    C.append(dominatees[0])
    dominatees = dominatees[1:]

  return C

def Optimization(G, C, m):
  C_sub = networkx.subgraph(G, C)
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