from src.graphs.graph_interface import GraphInterface
from src.graphs.mapping_graph_interface import MappingGraphInterface
from src.utils.alap import ALAP
from src.utils.util_graph import UtilGraph
from src.graphs.dfgs.dfg import DFG
from src.utils.zizag import Zigzag

class YOTODFG(MappingGraphInterface):
    def __init__(self, graph:GraphInterface,node_to_operation:dict): 
        self.special_value = -1
        self.zigzag_edges = Zigzag.get_zigzag_order(graph.get_edges())[0]
        placement_order = [*self.zigzag_edges[0]] + [v for (_,v) in self.zigzag_edges[1:]]
        self.base_dfg : DFG = DFG(graph,node_to_operation,placement_order,self.special_value)
        self.alap_values  = {self.base_dfg.real_to_reseted_node[k]:v for (k,v) in ALAP.get_alap_values(graph.get_vertices(),graph.get_edges()).items()}
        self.sched_order = {node:i for i,node in enumerate(self.base_dfg.placement_order)}
        self.vertices_to_scheduled_modulo_time_slice = UtilGraph.init_dict_node_to_something(self.base_dfg.num_nodes,self.special_value)
        self.edges_index = UtilGraph.generate_edges_index_by_edges(self.base_dfg.edges)
        
    @property
    def in_vertices(self):
        return self.base_dfg.in_vertices
    @property
    def out_vertices(self):
        return self.base_dfg.out_vertices
    def get_pe_assigned_to_node(self,node_id):
        return self.base_dfg.node_to_pe[node_id]
   
    def get_next_node_to_be_mapped(self,node_id = None):
        order = self.sched_order[node_id] if node_id is not None else -1
        return self.base_dfg.placement_order[order + 1] if order + 1 < len(self.base_dfg.vertices) else None
    
    def get_previous_mapped_node(self,node_id):
        order = self.sched_order[node_id] if node_id is not None else -1
        return self.base_dfg.placement_order[order - 1] if order - 1 >= 0 else None

    def get_nodes_to_pe(self):
        return self.base_dfg.node_to_pe

    def len_vertices(self):
        return len(self.base_dfg.vertices)

    def assign_PE_to_vertex(self,pe_id,node_id):
        self.base_dfg.node_to_pe[node_id] = pe_id
    
    def all_nodes_was_mapped(self):
        for pe in self.base_dfg.node_to_pe.values():
            if pe == self.special_value:
                return False
        return True

    def assign_scheduled_modulo_time_slice_to_vertex(self,scheduled_modulo_time_slice,node_id):
        self.vertices_to_scheduled_modulo_time_slice[node_id] = scheduled_modulo_time_slice

    def number_vertices_mapped_same_modulo_time_slice_by_node_id(self,node_id):
        modulo_time_slice_vertex = self.vertices_to_scheduled_modulo_time_slice[node_id]
        if modulo_time_slice_vertex == self.special_value:
            return 0
        count = 0
        for value in self.vertices_to_scheduled_modulo_time_slice.values():
            if modulo_time_slice_vertex == value:
                count += 1
        return count - 1

    def get_feature_by_node_id(self, node_id):    
        return {
            'id':node_id,
            'sched_order':self.sched_order[node_id],
            'sched_modulo':self.vertices_to_scheduled_modulo_time_slice[node_id],
            'in_degree':len(self.base_dfg.get_in_vertices_by_node_id(node_id)), #type: ignore
            'out_degree': len(self.base_dfg.get_out_vertices_by_node_id(node_id)), #type: ignore
            'op_code': self.base_dfg.get_operation_by_node_id(node_id),
            'has_self_cycle': int(self.base_dfg.node_has_self_cycle(node_id)),
            'num_vertex_same_modulo': self.number_vertices_mapped_same_modulo_time_slice_by_node_id(node_id),
            'assigned_PE_id': self.base_dfg.get_pe_assigned_to_node(node_id)
        }
          
    def get_node_features(self,type):
        if type == 'list':
            return self.get_list_node_features()
        elif type == 'dict':
            return self.get_dict_node_features()
        
    def get_dict_node_features(self):
        vertexes_features = {}
        for vertex in range(len(self.base_dfg.vertices)):
            vertexes_features[vertex] = self.get_feature_by_node_id(vertex)
        return vertexes_features
    
    def get_list_node_features(self):
        vertexes_features = []
        for vertex in range(len(self.base_dfg.vertices)):
            vertexes_features.append(list(self.get_feature_by_node_id(vertex).values()))
        return vertexes_features

    def get_edges_index(self):
        return self.edges_index
    