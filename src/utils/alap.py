import networkx as nx
import heapq
from src.utils.util_graph import UtilGraph
import pygraphviz as pgv
class ALAP:
    @staticmethod
    def get_alap_values(vertices,edges):
        A = pgv.AGraph(strict=False, directed=True)

        A.add_edges_from(edges)

        A.layout(prog='dot')

        positions = {}

        for node in A.nodes():
            pos = A.get_node(node).attr['pos']
            x, y = map(float, pos.split(','))
            positions[node] = y

        sorted_nodes = sorted(positions.items(), key=lambda item: item[1], reverse=True)

        levels = {}
        current_level = -1
        previous_y = None

        for node, y in sorted_nodes:
            if previous_y is None or y != previous_y:
                current_level += 1
            levels[str(node)] = int(current_level)
            previous_y = y

        return levels
    @staticmethod
    def get_output_heights(edges = None ,nx_graph = None):
        if edges:
            graph = nx.DiGraph()
            graph.add_edges_from(edges)
        else: 
            graph = nx_graph
        roots = [node for node in graph.nodes if graph.in_degree(node) == 0]
        output_heigths = {node: float('-inf') for node in graph.nodes if graph.out_degree(node) == 0}
        for root in roots:
            ALAP.calculate_depth(graph,root,output_heigths)
        return output_heigths
    @staticmethod
    def get_roots(edges):
        if edges:
            graph = nx.DiGraph()
            graph.add_edges_from(edges)
        roots = [node for node in graph.nodes if graph.in_degree(node) == 0]
        return roots
    @staticmethod
    def calculate_depth(graph, root,output_heigths):
        def backtracking(node, depth, visited,output_heigths):
            if graph.out_degree(node) == 0:
                if node in output_heigths:
                    output_heigths[node] = max(output_heigths[node],depth)
                else:
                    output_heigths[node] = depth
                visited.add(node)
                return None
            visited.add(node)
            for neighbor in graph.neighbors(node):
                if neighbor not in visited:
                    backtracking(neighbor, depth + 1, visited.copy(),output_heigths)
        return backtracking(root, 0, set(),output_heigths)
    
