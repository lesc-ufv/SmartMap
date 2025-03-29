from src.enums.enum_operation import EnumOperation
from src.graphs.graph_interface import GraphInterface
import networkx as nx

class NetworkXGraph(GraphInterface):
    def __init__(self,path_to_dot_file=None, vertices = None, edges = None):
        if path_to_dot_file:
            self.graph = nx.drawing.nx_pydot.read_dot(path_to_dot_file)
        elif vertices:
            self.graph = nx.DiGraph()
            self.graph.add_edges_from(edges)
            self.graph.add_nodes_from(vertices)

    def calc_in_vertex(self):
        in_vertex = {}
        for node in self.graph.nodes:
            in_vertex[node] = list(self.graph.predecessors(node))
        return in_vertex
    
    def calc_out_vertex(self):
        out_vertex = {}
        for node in self.graph.nodes:
            out_vertex[node] = list(self.graph.successors(node))
        return out_vertex
    
    def num_nodes(self):
        return len(self.graph.nodes)
    
    def get_vertices(self):
        return self.graph.nodes()

    def get_edges(self):
        return self.graph.edges()
    
    def create_by_nodes_and_edges(self,vertices: list, edges: list):
        return NetworkXGraph(None,vertices,edges)
    
    


    
