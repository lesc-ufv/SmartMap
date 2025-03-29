import networkx as nx
from typing import Union
from src.enums.enum_operation import EnumOperation
from src.utils.alap import ALAP
import pygraphviz as pgv
class UtilDFG:
    @staticmethod
    def get_max_root(edges):
        value = edges[0][0]

        A = pgv.AGraph(strict=False, directed=True)

        A.add_edges_from(edges)

        A.layout(prog='dot')

        positions = {}

        for node in A.nodes():
            pos = A.get_node(node).attr['pos']
            x, y = map(float, pos.split(','))
            positions[node] = y

        sorted_nodes = sorted(positions.items(), key=lambda item: item[1], reverse=True)
        if isinstance(value,str):
            return str(sorted_nodes[0][0])
        elif isinstance(value,int):
            return int(sorted_nodes[0][0])

    @staticmethod
    def generate_node_to_operation_by_networkx_graph(graph,att_name):
        node_to_operation = {}
        for vertex,label in nx.get_node_attributes(graph, att_name).items():
            node_to_operation[vertex] = EnumOperation.enum_operation_by_type(label).value
        return node_to_operation
    @staticmethod
    def balance_graph(nx_graph:nx.DiGraph):
        A = pgv.AGraph(strict=False, directed=True)
        num_vertices = len(nx_graph.nodes())
        A.add_edges_from(list(nx_graph.edges()))

        A.layout(prog='dot')

        positions = {}

        for node in A.nodes():
            pos = A.get_node(node).attr['pos']
            x, y = map(float, pos.split(','))
            positions[node] = y

        sorted_nodes = sorted(positions.items(), key=lambda item: item[1], reverse=True)

        levels = {}
        current_level = -1
        previous_y = None

        for node, y in sorted_nodes:
            if previous_y is None or y != previous_y:
                current_level += 1
            levels[node] = current_level
            previous_y = y

        for node in list(nx_graph.nodes()): 
            cur_time_slice = levels[node]
            for father in A.predecessors(node):
                if levels[father]  + 1 != cur_time_slice:
                    count = cur_time_slice - levels[father]  - 1
                    nx_graph.remove_edge(father,node)
                    temp_father = father
                    
                    while count > 0:
                        while True:
                            new_node = f'add{num_vertices}'
                            if new_node not in nx_graph.nodes():
                                break
                            else:
                                num_vertices += 1
                        nx_graph.add_node(new_node,**{'opcode':'add'})
                        nx_graph.add_edge(temp_father,new_node)
                        temp_father = new_node
                        count -= 1
                        num_vertices+=1
                    nx_graph.add_edge(temp_father,node)


        return nx_graph
    @staticmethod
    def num_nodes_for_graph_be_balanced(vertices,edges):
        A = pgv.AGraph(strict=False, directed=True)

        A.add_edges_from(edges)

        A.layout(prog='dot')

        positions = {}

        for node in A.nodes():
            pos = A.get_node(node).attr['pos']
            x, y = map(float, pos.split(','))
            positions[node] = y

        sorted_nodes = sorted(positions.items(), key=lambda item: item[1], reverse=True)

        levels = {}
        current_level = -1
        previous_y = None

        for node, y in sorted_nodes:
            if previous_y is None or y != previous_y:
                current_level += 1
            levels[node] = current_level
            previous_y = y

        total_nodes = 0
        for node in vertices: 
            cur_time_slice = levels[node]
            for father in A.predecessors(node):
                if levels[father]  + 1 != cur_time_slice:
                    total_nodes += cur_time_slice - levels[father]  - 1
        return total_nodes

    @staticmethod
    def dfg_is_balanced(vertices,edges):
        def schedule(node,G,cur_schedule,node_to_scheduled_time_slice,fifo):
            node_to_scheduled_time_slice[node] = max(0,cur_schedule)
            visited[node] = True
            fifo.append(node)
            for father in G.predecessors(node):
                if father not in node_to_scheduled_time_slice:
                    schedule(father,G,cur_schedule -1,node_to_scheduled_time_slice,fifo)


        root = UtilDFG.get_max_root(edges)
        G = nx.DiGraph()
        G.add_edges_from(edges)
        G.add_nodes_from(vertices)
        node_to_scheduled_time_slice = {root: 0}
        visited = {root:True}
        fifo = [root]
        while fifo:
            cur_node = fifo.pop(0)
            for out_node in G.successors(cur_node):
                if out_node not in visited:
                    node_to_scheduled_time_slice[out_node] = 1 + node_to_scheduled_time_slice[cur_node]
                    for father in G.predecessors(out_node):
                        if father not in node_to_scheduled_time_slice:
                            schedule(father,G,node_to_scheduled_time_slice[out_node]  - 1,node_to_scheduled_time_slice,fifo)
                    visited[out_node] = True
                    fifo.append(out_node)
                else:
                    if cur_node != out_node:
                        node_to_scheduled_time_slice[out_node] =  max(node_to_scheduled_time_slice[out_node],node_to_scheduled_time_slice[cur_node] + 1)
        
        for node in vertices: 
            cur_time_slice = node_to_scheduled_time_slice[node]
            for father in G.predecessors(node):
                if node_to_scheduled_time_slice[father]  + 1 != cur_time_slice:
                    return False
        return True

            
            

