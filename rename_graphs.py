



from pathlib import Path
import pygraphviz as pgv
import networkx as nx
import os
path_to_dataset = "benchmarks/changed_MCTS_benchmark/"
path = Path(path_to_dataset)

files = [str(file) for file in path.rglob('*.dot') if file.is_file()]
for file in files:
    print(file)
    G1 = nx.drawing.nx_pydot.read_dot(file)
    num_nodes= len(G1.nodes())
    num_edges = len(G1.edges())
    dfg_name = f'V_{num_nodes}_E_{num_edges}.dot'
    os.rename(file, os.path.dirname(file) + os.sep+ dfg_name)
    os.remove(file.replace('.dot','.png'))
     

