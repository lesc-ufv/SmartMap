from pathlib import Path
import pygraphviz as pgv
import networkx as nx
path = Path("benchmarks/changed_MCTS_benchmark/")

files = [str(file) for file in path.rglob('*.dot') if file.is_file()]
print(len(files))
for file in files:
    print(file)
    G1 = nx.drawing.nx_pydot.read_dot(file)
    G = pgv.AGraph(directed=True)
    G.add_nodes_from(list(G1.nodes()))
    G.add_edges_from(list(G1.edges()))
    G.draw(file.replace('.dot','.png'), format = 'png', prog = 'dot')
