from src.utils.util_eval import UtilEval
from pathlib import Path
from pathlib import Path
from src.graphs.graphs.networkx_graph import NetworkXGraph
from src.utils.util_dfg import UtilDFG
from src.utils.util_initialize import UtilInitialize
from src.utils.util_configs import UtilConfigs
from src.mapping_history import MappingHistory
import torch
from src.mcts.mcts import MCTS
import numpy
import os
import time
from src.enums.enum_mode import EnumMode
from src.enums.enum_interconnections import EnumInterconnections
from src.utils.util_module import UtilModule
import sys
from configs.config_mapzero import ConfigMapzero
from src.utils.util_mapping import UtilMapping
from src.entities.mapping_results import MappingResults
from configs.config_smartmap import ConfigSmartMap
from model_launcher_map import ModelLauncher
import copy
from src.utils.util_checkpoint import UtilCheckpoint
def print_placement(state):
    n_rows,n_cols = state.cgra.cgra.dim_arch
    matrix = [[-1 for j in range(n_cols)] for i in range(n_rows)]
    map_nodes = state.dfg.base_dfg.reseted_to_real_node
    pe_routing_items = list(state.cgra.cgra.pes_to_routing.items())
    node_to_scheduled_time_slice = state.node_to_scheduled_time_slice
  
    for k,v in state.cgra.get_pes_to_node_id().items():
        i_pos = k // n_rows
        j_pos = k % n_cols
        time_r = None
        if v == -2:
            time_r = -1
            for k_t,v_r in pe_routing_items:
                if k in v_r:
                    pos_pe = 0
                    pos_r = 1
                    while k!= v_r[pos_r]:
                        pos_r += 1
                    pe = v_r[pos_pe]
                    r_pe = v_r[pos_r]
                    len_routing = pos_r - pos_pe
                    node_pos_pe = state.cgra.cgra.pe_to_dfg_node[pe]
                    time_r = -1 if node_to_scheduled_time_slice is None else (state.node_to_scheduled_time_slice[node_pos_pe] + len_routing )
                    

        matrix[i_pos][j_pos] = f'{k} | ' + (str(map_nodes[v]) if v != -1 and v != -2 else "R" if v == -2 else str(v)) + ' | t = ' + (f'{time_r}' if time_r else  (f'{state.node_to_scheduled_time_slice[v]}' if node_to_scheduled_time_slice and v != -1 else '-1'))
    
    def get_max_lengths(data):
        max_init = 0
        max_final = 0
        max_mid = 0
        for row in data:
            for item in row:
                init,mid,final =  item.split('|')
                max_init = max(max_init, len(init.strip()))
                max_final = max(max_final, len(final.strip()))
                max_mid = max(max_mid, len(mid.strip()))
        return max_init, max_mid, max_final

    max_init, max_mid, max_final = get_max_lengths(matrix)

    formatted_rows = [
        [f"{item.split('|')[0].strip():<{max_init}} | {item.split('|')[1].strip():<{max_mid}} | {item.split('|')[2].strip():<{max_final}} "
        for item in row]
        for row in matrix
    ]

    for row in formatted_rows:
        print('\t'+' '.join(f"[{item}] " for item in row))

def format_dict(data):
    max_key_length = max(len(f"{key}") for key in data)
    max_value_length = max(len(f"{value}") for value in data.values())

    formatted_entries = []
    for key, value in data.items():
        key_str = f"{key}".ljust(max_key_length)
        value_str = f"{value}".ljust(max_value_length)
        formatted_entries.append(f"\t{key_str} | {value_str}")

    for entry in formatted_entries:
        print(entry)

        
@torch.no_grad
def infer(model_config):
    config = model_config.get_config()
    if config.seed:
        numpy.random.seed(config.seed)
        torch.manual_seed(config.seed)

    cgra = model_config.get_cgra()
    model = config.class_model(*config.model_instance_args)
    environment = config.environment
    model_path = model_config.get_path_to_ckpt_model()

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path ,map_location=config.device))
    else:
        print()
        print(f'Model not found in checkpoint path {model_path}. Please train the model before running inference.\n')
        return None

    path = Path(config.path_to_test_dataset)
    files = [str(file) for file in path.rglob('*.dot') if file.is_file()]
    arch_dims = config.arch_dims
    num_pes = arch_dims[0]*arch_dims[1]
    states = [] 

    for file in files:
        dfg_graph = NetworkXGraph(file)
        if len(dfg_graph.get_vertices()) + UtilDFG.num_nodes_for_graph_be_balanced(list(dfg_graph.get_vertices()),list(dfg_graph.get_edges())) > num_pes:
            continue
        node_to_operation = UtilDFG.generate_node_to_operation_by_networkx_graph(dfg_graph.graph,"opcode")
        dfg = UtilInitialize.initialize_dfg_from_class_name(config.dfg_class_name,dfg_graph,node_to_operation)
        initial_state = UtilConfigs.get_mapping_state_by_model_name(config.model_name,dfg,cgra,dfg.get_next_node_to_be_mapped(),
                                        config.distance_func)
        dfg_name = os.path.basename(file)
        states.append([initial_state,dfg_name])
    for state,dfg_name in states:
        with open("/dev/tty", "w") as terminal:
            print(f"\nMapping {dfg_name}\n",file=terminal)
        checkpoint = UtilCheckpoint.get_checkpoint()
        checkpoint['weights'] = model.get_weights()
        model_launcher = ModelLauncher(model,dfg_name,config,state,environment,checkpoint,None)
        print(50*'-'+f" Mapping {dfg_name} "+50*'-')
        print()
        
        init_mapping_time = time.time()
        model_launcher.map(True)
        mapping = model_launcher.mapping_history

        model_cp = copy.deepcopy(model)
        model_cp.load_state_dict(model_launcher.checkpoint['weights'])
        if mapping is None:
            print("\nSolution not found during finetune. Using the finetuned model to generate the final mapping.\n")
            mapping, max_tree_depths, mcst_simu_flag, num_backtrackings = UtilEval.play_game(config,model_cp,state,environment,config.num_max_expansion_test)
            final_mapping_time = time.time()
            mapping_time = final_mapping_time - init_mapping_time
            if mcst_simu_flag:
                print('Solution found before finish the MCTS simulations.')
            sm_wo_finetune = mapping.observation_history[-1].mapping_is_valid

        else:
            final_mapping_time = time.time()
            mapping_time = model_launcher.map_time

            mapping_wo_finetune, max_tree_depths, mcst_simu_flag, num_backtrackings = UtilEval.play_game(config,model_cp,state,environment,config.num_max_expansion_test)
            sm_wo_finetune = mapping_wo_finetune.observation_history[-1].mapping_is_valid

        final_state = mapping.observation_history[-1]
        unsuccessful_reason = UtilMapping.get_unsuccesful_reason(final_state)
        mapping_is_valid = 1 if final_state.mapping_is_valid else 0
        used_PEs = 0
        for dfg_node in final_state.cgra.cgra.pe_to_dfg_node.values():
            if dfg_node != -1:
                used_PEs += 1

        mapping_text = 'Successful' if mapping_is_valid == 1 else "Unsuccessful"
        routing_penalty = sum(mapping.reward_history)
        print(f"{mapping_text} Mapping | Routing Penalty: {routing_penalty} | Mapping Time: {mapping_time:.3f} sec | Num Simulations: {config.num_simulations}")
        if not mapping_is_valid:
            print(f"Unsuccessful Mapping Reason: {unsuccessful_reason}")

        print() 
        print('Placement and Timing:')
        print('[PE | Placed DFG node or routing PE (R) | Scheduled Time]\n')
        print_placement(final_state)
        print()

        print("Routing:")
        print("(Source PE, Destination PE) | Routing path passing through PEs: [Source PE, PE_x, ..., PE_y,Destination PE]\n")
        format_dict(final_state.cgra.cgra.pes_to_routing)
        print()

        mean_max_tree_depths = numpy.mean(max_tree_depths)


        mean_visited_rate,mean_visited_nodes,mean_mean_expanded_nodes = UtilEval.calculate_mean_expanded_nodes(model_cp,config,state,environment,config.num_max_expansion_test,config.max_moves) 
        
        print("Here, the MCTS results are obtained by mapping using the last weights from finetune, just for comparison with zero shot results.")
        print(f'MCTS results: Succesfull Mapping: {sm_wo_finetune}.Mean visited rate: {mean_visited_rate*100:.3f}% | Mean visited nodes: {mean_visited_nodes:.3f} | 2xMean expanded nodes: {mean_mean_expanded_nodes:.3f} | Mean max tree depth {mean_max_tree_depths:.3f} | Number of Backtrackings: {num_backtrackings}')
        
        mapping_results = MappingResults(config.model_name.value,arch_dims,model_config.type_interconnnections.value,dfg_name.replace('.dot',''),EnumMode.FINETUNE.value)
        mapping_results.mapping_is_valid = final_state.mapping_is_valid
        mapping_results.routing_penalty = routing_penalty
        mapping_results.unsuccessful_reason = unsuccessful_reason
        mapping_results.mapping_time = mapping_time
        mapping_results.mean_visited_rate = mean_visited_rate
        mapping_results.mean_num_visited_nodes = mean_visited_nodes
        mapping_results.mean_mean_expanded_nodes = mean_mean_expanded_nodes
        mapping_results.mean_max_tree_depths = mean_max_tree_depths
        mapping_results.num_simulations = config.num_simulations
        mapping_results.used_PEs = used_PEs
        mapping_results.num_backtrackings = num_backtrackings
        mapping_results.save_csv()

        
        print()
        print(50*'-'+f" End Mapping {dfg_name} "+50*'-')
        print('\n\n')
    
if __name__ == "__main__":
    args = sys.argv
    config_module = args[1]
    config_class = args[2]
    interconnection_style = args[3]
    interconnection_style = EnumInterconnections(interconnection_style)
    arch_dims = tuple(map(int,args[4].split('x')))
    mode = EnumMode(args[5])
    model_config = UtilModule.load_class_from_file(config_module,config_class,interconnection_style,arch_dims,mode)
    infer(model_config)