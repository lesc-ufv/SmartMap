



from pathlib import Path
import pygraphviz as pgv
import networkx as nx
path = Path("benchmarks/e2e_benchmark/")

files = [str(file) for file in path.rglob('*.dot') if file.is_file()]
print(len(files))
for file in files:
    G1 = nx.drawing.nx_pydot.read_dot(file)
    new_graph = nx.DiGraph()
    map_nodes = {}
    new_vertices = [ ]
    for i,node in enumerate(G1.nodes()):
        new_vertices.append(f'add{i}')
        map_nodes[node] = f'add{i}'
    new_edges = [(map_nodes[edge[0]],map_nodes[edge[1]]) for edge in G1.edges()]
    new_graph.add_edges_from(new_edges)
    new_graph.add_nodes_from(new_vertices)
    nx.set_node_attributes(new_graph, 'add', 'opcode')
    nx.drawing.nx_pydot.write_dot(new_graph,file)