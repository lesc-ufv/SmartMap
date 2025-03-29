from abc import ABC,abstractmethod

class GraphInterface(ABC):
    @abstractmethod
    def create_by_nodes_and_edges(self,vertices:list,edges:list):
        pass
    @abstractmethod
    def calc_in_vertex():
        pass
    
    @abstractmethod
    def calc_out_vertex():
        pass
    
    @abstractmethod
    def num_nodes():
        pass

    @abstractmethod
    def get_vertices():
        pass

    @abstractmethod
    def get_edges():
        pass