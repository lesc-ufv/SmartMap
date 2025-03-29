from src.graphs.graph_interface import GraphInterface
from src.utils.util_graph import UtilGraph


class CGRA:
    def __init__(self,graph:GraphInterface,dim_arch,pe_to_functionalities,arch_name,interconnect_style,fill_value,*interconnection_generators):
        self.arch_name = arch_name
        self.dim_arch = dim_arch
        self.pe_to_functionalities = pe_to_functionalities
        self.interconnect_style = interconnect_style
        self.fill_value = fill_value
        self.interconnection_generator_names = [interconnection_func.__name__ for interconnection_func in interconnection_generators]
        self.pos_vertices = [(i,j) for j in range(dim_arch[1]) for i in range(dim_arch[1])]
        self.pos_edges = []
        for interconnection_func in interconnection_generators:
            for pos_pe in self.pos_vertices:
                pos_adj_pes = interconnection_func(pos_pe,dim_arch)
                for pos_adj_pe in pos_adj_pes:
                    edge = [pos_pe,pos_adj_pe]
                    if edge not in self.pos_edges:
                        self.pos_edges.append(edge)
        self.int_vertices, self.real_to_reseted_nodes, self.reseted_to_real_nodes = UtilGraph.reset_vertices_labels(self.pos_vertices)
        self.int_edges = UtilGraph.transform_edges_labels_by_dict(self.pos_edges,self.real_to_reseted_nodes)
        self.graph :GraphInterface = graph.create_by_nodes_and_edges(self.int_vertices,self.int_edges)
        self.pe_to_dfg_node = UtilGraph.init_dict_node_to_something(self.int_vertices,fill_value)
        self.out_vertices = self.graph.calc_out_vertex()
        self.int_vertices = self.graph.calc_in_vertex()
        self.pes_to_routing :dict[tuple[int],list[tuple[int]]] = {}
        self.free_connections = self.graph.calc_out_vertex() 

    def get_node_assigned_to_pe(self,pe):
        return self.pe_to_dfg_node[pe]
    def all_pes_was_used(self):
        for v in self.pe_to_dfg_node.values():
            if v == self.fill_value:
                return False
        return True