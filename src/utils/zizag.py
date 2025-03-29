import networkx as nx
import random 
class Zigzag:
    @staticmethod
    def get_zigzag_order( edges, make_shuffle: bool = False):
        """
        Returns a list of edges according to the zigzag algorithm.

        Args:
            make_shuffle (bool): Whether to shuffle the output list.

        Returns:
            Tuple containing three lists: edges_str, edges_raw, and convergence.
        """
        g = nx.DiGraph()
        g.add_edges_from(edges)
        
        output_list = [[node, 'IN'] for node in g.nodes() if g.out_degree(node) == 0]

        if make_shuffle:
            random.shuffle(output_list)

        stack = output_list.copy()
        edges = []
        visited = set()
        convergence = []

        fan_in = {node: list(g.predecessors(node)) for node in g.nodes()}
        fan_out = {node: list(g.successors(node)) for node in g.nodes()}

        if make_shuffle:
            for node in g.nodes():
                random.shuffle(fan_in[node])
                random.shuffle(fan_out[node])

        while stack:
            a, direction = stack.pop(0)  # get the top1
            visited.add(a)

            if direction == 'IN':  # direction == 'IN'

                if fan_out[a]:  # Case 3
                    b = fan_out[a][-1]
                    stack.insert(0, [a, 'IN'])
                    stack[:0] = [[a, 'IN']] * len(fan_in[a])
                    stack.insert(0, [b, 'OUT'])
                    fan_out[a].remove(b)
                    fan_in[b].remove(a)
                    if b in visited:
                        convergence.append([a, b])
                    edges.append([a, b, 'OUT'])

                elif fan_in[a]:  # Case 2
                    b = fan_in[a][-1]
                    stack.insert(0, [a, 'IN'])
                    stack[:0] = [[b, 'IN']] * len(fan_in[a])
                    fan_in[a].remove(b)
                    fan_out[b].remove(a)
                    if b in visited:
                        convergence.append([a, b])
                    edges.append([a, b, 'IN'])

            else:  # direction == 'OUT'

                if fan_in[a]:  # Case 3
                    b = fan_in[a][0]
                    stack.insert(0, [a, 'OUT'])
                    stack[:0] = [[a, 'OUT']] * len(fan_out[a])
                    stack.insert(0, [b, 'IN'])
                    fan_in[a].remove(b)
                    fan_out[b].remove(a)
                    if b in visited:
                        convergence.append([a, b])
                    edges.append([a, b, 'IN'])

                elif fan_out[a]:  # Case 2
                    b = fan_out[a][0]
                    stack.insert(0, [a, 'OUT'])
                    stack[:0] = [[b, 'OUT']] * len(fan_out[a])
                    fan_out[a].remove(b)
                    fan_in[b].remove(a)
                    if b in visited:
                        convergence.append([a, b])
                    edges.append([a, b, 'OUT'])

        return Zigzag.clear_edges(edges), Zigzag.clear_edges(edges, False), convergence
        # edges = Zigzag.clear_ed?ges(edges)
        # return edges

    @staticmethod
    def clear_edges(edges, remove_placed_edges: bool = True):
        """
        Removes duplicate edges from the list.

        Args:
            edges (List[List[str]]): List of edges.
            remove_placed_edges (bool): Whether to remove placed edges.

        Returns:
            List[List[str]]: List of unique edges.
        """
        dic = set()
        dic.add(edges[0][0])
        new_edges = []
        for edge in edges:
            n1, n2 = edge[:2]
            if n2 not in dic or not remove_placed_edges:
                dic.add(n2)
                new_edges.append([n1, n2])
        return new_edges

    @staticmethod
    def get_graph_annotations(edges, cycle):
        def func_key(val1: str, val2: str) -> str:
            """
            Concatenate two strings to form a key.

            :param val1: The first string.
            :type val1: str
            :param val2: The second string.
            :type val2: str
            :return: The concatenated key.
            :rtype: str
            """
            return (val1,val2)

        dic_cycle = {}

        # Initialization dictionary
        for edge in edges:
            key: str = func_key(edge[0], edge[1])
            dic_cycle[key] = []

        for elem_cycle_begin, elem_cycle_end in cycle:
            walk_key = []
            found_start = False
            count = 0
            value1 = ''

            for edge in reversed(edges):
                if elem_cycle_begin == edge[1] and not found_start:
                    value1 = edge[0]
                    key = func_key(value1, elem_cycle_begin)
                    walk_key.insert(0, key)
                    dic_cycle[key].append([elem_cycle_end, count])
                    count += 1
                    found_start = True

                elif found_start and (value1 == edge[1] or elem_cycle_end == edge[0]):
                    value1, value2 = edge[0], edge[1]
                    key = func_key(value1, value2)
                    if value1 != elem_cycle_end and value2 != elem_cycle_end:
                        walk_key.insert(0, key)
                        dic_cycle[key].append([elem_cycle_end, count])
                        count += 1
                    else:
                        # Go back and update values
                        for k in range(count // 2):
                            dic_actual = dic_cycle[walk_key[k]]
                            for dic_key, (node, count) in enumerate(dic_actual):
                                if node == elem_cycle_end:
                                    dic_actual[dic_key][1] = k + 1
                        break  # to the next on the vector CYCLE
        return dic_cycle