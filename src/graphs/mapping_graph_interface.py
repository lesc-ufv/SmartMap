from abc import ABC,abstractmethod

class MappingGraphInterface(ABC):
    @abstractmethod
    def get_node_features():
        pass

    @abstractmethod
    def get_feature_by_node_id():
        pass

    @abstractmethod
    def get_edges_index():
       pass

    