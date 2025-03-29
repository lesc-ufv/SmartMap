from src.graphs.graph_interface import GraphInterface
from src.graphs.cgras.cgra import CGRA
from src.graphs.mapping_graph_interface import MappingGraphInterface
from src.utils.util_graph import UtilGraph
from src.enums.enum_functionalitie_pe import EnumFunctionalitiePE
class CGRATraversal(MappingGraphInterface):

    def __init__(self,graph:GraphInterface,dim_arch,pe_to_functionalities,interconnect_style,arch_name,*interconnection_generators):
        self.fill_value = -1
        self.cgra = CGRA(graph,dim_arch,pe_to_functionalities,arch_name,interconnect_style,self.fill_value,*interconnection_generators)
        self.edges_index = UtilGraph.generate_edges_index_by_edges(self.cgra.graph.get_edges())
        self.in_vertex = self.cgra.graph.calc_in_vertex()
        self.out_vertex = self.cgra.graph.calc_out_vertex()
    def len_vertices(self):
        return self.cgra.graph.num_nodes()
    def get_cgra_dims(self):
        return self.cgra.dim_arch

    def get_interconnection_names(self):
        return self.cgra.interconnection_generator_names

    def get_interconnection_style(self):
        return self.cgra.interconnect_style

    def get_graph(self):
        return self.cgra.graph

    def get_pe_to_functionalities(self):
        return self.cgra.pe_to_functionalities

    def get_pes_to_node_id(self):
        return self.cgra.pe_to_dfg_node

    def get_arch_name(self):
        return self.cgra.arch_name
        
    def get_pes_to_routing(self):
        return self.cgra.pes_to_routing
    def get_edges_index(self):
        return self.edges_index
    
    def get_feature_by_node_id(self,pe_id):
        functionalities = self.cgra.pe_to_functionalities[pe_id]
        id_mapped_node = self.cgra.pe_to_dfg_node[pe_id]
        return {
            'id': pe_id,
            'in-degree': len(self.in_vertex[pe_id]),
            'out-degree': len(self.out_vertex[pe_id]),
            'has_logical': int(functionalities[EnumFunctionalitiePE.LOGICAL.value] != EnumFunctionalitiePE.NONE),
            'has_arithmetic': int(functionalities[EnumFunctionalitiePE.ARITHMETIC.value] != EnumFunctionalitiePE.NONE),
            'has_memory_access': int(functionalities[EnumFunctionalitiePE.MEMORY_ACCESS .value] != EnumFunctionalitiePE.NONE),
            'id_mapped_dfg_node':  id_mapped_node if isinstance(id_mapped_node,int) else -2
        }

    def get_node_features(self,type):
        if type == 'list':
            return self.get_list_node_features()
        elif type == 'dict':
            return self.get_dict_node_features()

        
    def get_dict_node_features(self):
        node_features = {}
        for vertex in self.cgra.graph.get_vertices():
            node_features[vertex] = self.get_feature_by_node_id(vertex)
        return node_features
    def get_list_node_features(self):
        node_features = []
        for vertex in self.cgra.graph.get_vertices():
            node_features.append(list(self.get_feature_by_node_id(vertex).values()))
        return node_features

    
    def get_interconnect_style(self):
        return self.cgra.interconnect_style

    def assign_node_to_pe(self,node_id,pe_id):
        self.cgra.pe_to_dfg_node[pe_id] = node_id
    
    def get_free_interconnections(self):
        return self.cgra.free_connections

    def get_out_vertices(self):
        return self.cgra.out_vertices   
    
    def update_free_interconnections(self,new_free_interconnections):
        self.cgra.free_connections = new_free_interconnections
    
    def update_pes_to_routing(self,new_pes_to_routing):
        self.cgra.pes_to_routing = new_pes_to_routing
    
    def get_pe_pos_by_pe_id(self,pe_id):
        return self.cgra.reseted_to_real_nodes[pe_id]
    
    def generate_mask(self):
        return [1 if self.cgra.pe_to_dfg_node[i] == self.fill_value else -float('inf') for i in range(len(self.cgra.graph.get_vertices()))] 

    def generate_legal_actions(self):
        return [i for i in range(len(self.cgra.graph.get_vertices())) if self.cgra.pe_to_dfg_node[i] == self.fill_value]

    def get_used_pes(self):
        return [pe for pe in self.cgra.pe_to_dfg_node.keys() if self.cgra.pe_to_dfg_node[pe] != self.fill_value]
    
    def get_node_assigned_to_pe(self,pe):
        return self.cgra.get_node_assigned_to_pe(pe)