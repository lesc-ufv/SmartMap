from src.graphs.graph_interface import GraphInterface
from src.utils.util_graph import UtilGraph
from src.enums.enum_operation import EnumOperation
from src.utils.util_dfg import UtilDFG

class DFG:
    def __init__(self, graph: GraphInterface,node_to_operation:dict, placement_order:list, value_to_fill = -1): 
        self.graph = graph
        self.vertices,self.real_to_reseted_node,self.reseted_to_real_node = UtilGraph.reset_vertices_labels(self.graph.get_vertices())
        self.edges = UtilGraph.transform_edges_labels_by_dict(self.graph.get_edges(),self.real_to_reseted_node)
        self.node_to_operation :dict = {self.real_to_reseted_node[node]:op for node,op in node_to_operation.items()}
        self.in_vertices : dict = None
        self.out_vertices :dict = None
        self.node_to_pe : dict = UtilGraph.init_dict_node_to_something(self.graph.num_nodes(),value_to_fill)
        self.placement_order :list = [self.real_to_reseted_node[node] for node in placement_order]
        self.max_root = UtilDFG.get_max_root(self.edges)

        
    @property
    def num_nodes(self):
        return self.graph.num_nodes()
    def get_real_nodes(self):
        return self.graph.get_vertices()
    
    def get_pe_assigned_to_node(self,node_id):
        return self.node_to_pe[node_id]
    
    def get_in_vertices_by_node_id(self,node_id):
        if self.in_vertices:
            return self.in_vertices[node_id]
        self.in_vertices = {self.real_to_reseted_node[node]:[self.real_to_reseted_node[in_vertice] for in_vertice in in_node] for node,in_node \
                            in self.graph.calc_in_vertex().items()}

        return self.in_vertices[node_id]
    
    def get_out_vertices_by_node_id(self,node_id):
        if self.out_vertices:
            return self.out_vertices[node_id]
        self.out_vertices = {self.real_to_reseted_node[node]:[self.real_to_reseted_node[out_vertice] for out_vertice in out_node] for node,out_node \
                            in self.graph.calc_out_vertex().items()}
        return self.out_vertices[node_id]
    
    def get_operation_by_node_id(self,node_id):
        return self.node_to_operation[node_id][EnumOperation.index_id()]
    
    def node_has_self_cycle(self,node_id):
        node_out_vertices = self.get_out_vertices_by_node_id(node_id)
        return node_id in node_out_vertices 