from typing import Union, List, Tuple, Dict


class UtilGraph:
    """
    Class to facilitate the use of commom operations in graphs.
    """
    @staticmethod
    def init_dict_node_to_something(vertices_or_num_vertices: Union[int, list], fill_value=-1) -> Dict[int, int]:
        """
        Initializes a dictionary mapping each node to an initial value. If vertices_or_num_vertices is provided as a
        list of vertices, the dictionary will map the nodes within it to the value in fill_value. If provided with a
        number of vertices, the dictionary will be generated considering vertices labeled from 0 to (num_vertices - 1)."

        Args:
            vertices_or_num_vertices (Union[int, list]): A list of vertices or the total number of vertices.
            fill_value (int): Value to initialize the values of the dictionary.

        Returns:
            dict: A dictionary containing pairs of (k, v) where k is the node label and v equals the fill_value.

        Tested:
            True
        """
        node_to_pe = {}
        if isinstance(vertices_or_num_vertices, int):
            for node_id in range(vertices_or_num_vertices):
                node_to_pe[node_id] = fill_value
        else:
            for node_id in vertices_or_num_vertices:
                node_to_pe[node_id] = fill_value
        return node_to_pe

    @staticmethod
    def reset_vertices_labels(vertices: List[int]) -> Tuple[List[int], Dict[int, int], Dict[int, int]]:
        """
        Relabels the vertices from 0 to (number of vertices - 1).

        Args:
            vertices (list): List of vertices.

        Returns:
            list: List with the relabeled vertices.
            dict: Pairs of (k, v) where k is the original label and v is the corresponding reset label of the vertices.
            dict: Pairs of (k, v) where k is the reset label and v is the original label of the vertices.

        Tested:
            True
        """
        real_to_reset = {}
        reset_to_real = {}
        new_vertices = []
        for count, vertex in enumerate(vertices):
            real_to_reset[vertex] = count
            reset_to_real[count] = vertex
            new_vertices.append(count)
        return new_vertices, real_to_reset, reset_to_real

    @staticmethod
    def transform_edges_labels_by_dict(edges: List[Tuple[int, int]], old_to_new_nodes: Dict[int, int]) -> List[Tuple[int, int]]:
        """
        Relabels the nodes in edges using a dictionary containing pairs (k, v) where k is the original label and v is the new label.

        Args:
            edges (list): List of pairs of nodes.
            old_to_new_nodes (dict): Dictionary where key is the original label and value is the new label.

        Returns:
            list: List of pairs of nodes with the new labels.

        Tested:
            True
        """
        return [(old_to_new_nodes[u], old_to_new_nodes[v]) for u, v in edges]

    @staticmethod
    def generate_edges_index_by_edges(edges: List[Tuple[int, int]]) -> List[List[int]]:
        """
        Generates the edges index from a list of edges.

        Args:
            edges (list): List of pairs of nodes.

        Returns:
            list[list,list]: A list containing two lists where the first contains the source nodes and the second contains the corresponding
            destination nodes. For example, if node_a = list[0][i] and node_b = list[1][i], it indicates that node_a has an edge pointing to node_b.

        Tested:
            True
        """
        edges_index = [[], []]
        for src, dst in edges:
            edges_index[0].append(src)
            edges_index[1].append(dst)
        return edges_index

    @staticmethod
    def generate_dict_in_or_out_vertices(vertices: List[int], edges: List[Tuple[int, int]], in_or_out: str) -> Dict[int, List[int]]:
        """
        Creates a dictionary that maps each node to its in-degree or out-degree vertices.

        Args:
            vertices (list): List of vertices.
            edges (list): List of pairs of vertices.
            in_or_out (str): Must be "in" or "out", indicating whether to generate a dictionary
            containing in-degree or out-degree vertices, respectively.

        Returns:
            dict: Dictionary mapping each node according to the value in the in_or_out variable.

        Raises:
            ValueError: If in_or_out is not "in" or "out".

        Tested:
            True
        """
        def split_in_edge(edge: Tuple[int, int]) -> Tuple[int, int]:
            return edge

        def split_out_edge(edge: Tuple[int, int]) -> Tuple[int, int]:
            return edge[1], edge[0]

        if in_or_out == 'in':
            split_func = split_in_edge
        elif in_or_out == 'out':
            split_func = split_out_edge
        else:
            raise ValueError('in_or_out parameter must be "in" or "out"')

        node_dict = {vertex: [] for vertex in vertices}
        for edge in edges:
            u, v = split_func(edge)
            node_dict[v].append(u)

        return node_dict
