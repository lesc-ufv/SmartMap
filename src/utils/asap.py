class ASAP:
    def __init__(self):
        pass
    def get_asap_values(self,vertexes,in_vertices):
        """
        Calculates the ASAP (as soon as possible) value of the nodes, which determines the earliest possible scheduling order based on their 
        predecessors.

        Args:
            vertexes (list): List of vertices.
            in_vertices (dict): A dictionary where the keys represent vertices from the 'vertexes' argument and the values contain 
            their incoming degree nodes.        
        
        Returns:
            dict[str,int]: Keys are the vertices in 'vertexes', and values represent their ASAP values.

        Tested:
            False
        """
        asap_values : dict[int,int] = {}
        while len(asap_values) != len(vertexes):
            for node in vertexes:
                if len(in_vertices[node]) == 0: 
                    asap_values[node] = 1
                elif self.__predecessor_nodes_was_scheduled(node,asap_values,in_vertices):
                    asap_values[node] = max([(asap_values[n] + 1) for n in in_vertices[node] if n!= node]) 
        return asap_values
        
    def __predecessor_nodes_was_scheduled(self,node:int,asap_values:dict[int,int],in_vertices) -> bool:
        """
            Checks whether all predecessor nodes of a node have been scheduled.
            Args:
                node (int): Vertex to be checked.
                asap_values (dict): Dictionary where keys are the vertex labels and values represent their ASAP values.
                in_vertices (dict): A dictionary where keys are the vertex labels and values contain 
                their incoming degree nodes.
            Returns:
                bool: True if all node's predecessors are scheduled, False otherwise.
            Tested:
                Not necessary.
        """

        for n in in_vertices[node]:
            if n != node:
                boolean = self.__backtracking(n,asap_values,in_vertices)
                if not boolean:
                    return False
        return True

    def __backtracking(self,node:int,asap_values,in_vertices) -> bool:
        """
            Verifies recursively if all node's predecessors have been scheduled.
            Auxiliary function for predecessor_nodes_was_scheduled. 
            Args:
                node (int): Vertex to be checked.
                asap_values (dict): Dictionary where keys are the vertex labels and values represent their ASAP values.
                in_vertices (dict): A dictionary where keys are the vertex labels and values contain 
                their incoming degree nodes.
            Returns:
                bool: True if all node's predecessors are scheduled, False otherwise.
            Tested:
                Not necessary.
        """
        if asap_values.get(node) is None:
            return False
        if asap_values[node] == 1:
            return True
        for n in in_vertices[node]:
            if n != node:
                boolean = self.__backtracking(n, asap_values,in_vertices)
                if not boolean:
                    return False
        return True 