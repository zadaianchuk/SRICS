import networkx as nx
import numpy as np



def get_graph(adj_matrix):
    N = adj_matrix.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(N))
    edges  = []
    for i in range(N):
        for j in range(N):
            if adj_matrix[i, j] != 0:
                edges.append((i,j))
    G.add_edges_from(edges)
    return G


def get_subgoals(adj_matrix, source=0):
    """
    returns subsets that have unique path from sourse to meta node with all nodes in subset
    """
    G = get_graph(adj_matrix)
    N = G.number_of_nodes()
    un_sets = [frozenset({0})]
    for i in range(0, N):
        pathes = []
        for path in nx.all_simple_paths(G, source=source, target=i):
            pathes.append(path)
        un_set = set().union(*[set(path) for path in pathes])
        un_sets.append(frozenset(un_set))
    un_sets = set(un_sets)
    subsets = [list(x) for x in list(un_sets)]
    subgoals = [subset for subset in subsets if subset]
    return subgoals

def get_parents(subgoals, N=5):
    """
    returns subsets that have unique path from sourse to meta node with all nodes in subset
    """
    all_parents = {} 
    for i in range(0, N):
        parents = []
        for subgoal in subgoals:
            if i in subgoal:
                idx = subgoal.index(i)
                parents.extend(subgoal[:idx])
        subgoal_parents_idx = np.zeros((1, N))
        subgoal_parents_idx[:, parents] = 1
        all_parents[i] = subgoal_parents_idx

    return all_parents

if __name__ == '__main__':
    weights = np.array(
                [[1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [1, 0, 1, 1, 0],
                [1, 0, 1, 1, 0],
                [1, 0, 0, 0, 1],
                ])
    subgoals = get_subgoals(weights, source=0)
    print(subgoals)
    parents = get_parents(subgoals)
    print(parents)


