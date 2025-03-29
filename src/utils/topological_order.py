import networkx as nx

class TopologicalOrder:
    @staticmethod
    def get_topological_order(vertices,edges):
        G = nx.DiGraph()
        G.add_nodes_from(vertices)
        G.add_edges_from(edges)
        alap_values = {node: float('inf') for node in G.nodes}

        exit_nodes = [n for n, d in G.out_degree() if d == 0]

        
        for node in exit_nodes:
            alap_values[node] = 0

        return list(nx.topological_sort(G))
        
