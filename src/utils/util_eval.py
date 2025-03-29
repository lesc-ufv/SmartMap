from pathlib import Path
from src.graphs.graphs.networkx_graph import NetworkXGraph
from src.utils.util_dfg import UtilDFG
from src.utils.util_initialize import UtilInitialize
from src.utils.util_configs import UtilConfigs
from src.mapping_history import MappingHistory
import torch
from src.mcts.mcts import MCTS
import numpy
from src.utils.util_dfg import UtilDFG
import os
class UtilEval:
    @staticmethod
    def calculate_mean_expanded_nodes(model,config,observation,environment,num_max_expansion,max_moves):
        """
        Play one game with actions based on the Monte Carlo tree search at each moves.
        """
        done = False
        count = 0
        fifo = []
        mean_num_expanded_nodes_list = []
        visited_nodes_list = []
        visited_rates = []
        with torch.no_grad():
            while (
                not done and count <= max_moves
            ):
            # Choose the action
                root, mcts_info = MCTS(config).run(
                    model,
                    observation,
                    observation.get_legal_actions(),
                    False,
                    num_max_expansion=num_max_expansion,
                    get_first_valid_solution=True
                )
                total_nodes = 0
                non_visited_nodes = 0
                cur_num_expanded_nodes_list = []

                fifo.append(root)
                while len(fifo) != 0:
                    node = fifo.pop(0)
                    total_nodes += 1
                    if node.visit_count != 0:
                        cur_num_expanded_nodes_list.append(len(node.children) )
                    else: 
                        non_visited_nodes += 1

                    for action,next_node in node.children.items():
                        fifo.append(next_node)

                num_visited_nodes = total_nodes - non_visited_nodes
                rate = num_visited_nodes/total_nodes

                visited_rates.append(rate)
                visited_nodes_list.append(num_visited_nodes)
                mean_num_expanded_nodes_list.append(numpy.mean(cur_num_expanded_nodes_list))

                if mcts_info['trajectory'] is not None:
                    return numpy.mean( visited_rates),numpy.mean( visited_nodes_list),numpy.mean(  mean_num_expanded_nodes_list),

                count = 0
                invalid_actions = set()
                while count < len(root.children):
                    action = UtilEval.select_action(
                        root,invalid_actions
                    )

                    observation, reward, done = environment.step(root.state,action)
                    if not root.state.is_bad_reward(reward):
                        break
                    count+=1
                    invalid_actions.add(action)
                
        return numpy.mean(visited_rates),numpy.mean(visited_nodes_list),numpy.mean(mean_num_expanded_nodes_list),
    @staticmethod
    def get_initial_eval_states(config,cgra):
        path = Path(config.path_to_test_dataset)
        files = [str(file) for file in path.rglob('*.dot') if file.is_file()]
        arch_dims = config.arch_dims
        num_pes = arch_dims[0]*arch_dims[1]
        states = [] 
        for file in files:
            dfg_graph = NetworkXGraph(file)
            len_graph = len(dfg_graph.get_vertices())

            if len_graph + UtilDFG.num_nodes_for_graph_be_balanced(dfg_graph.get_vertices(),dfg_graph.get_edges()) > num_pes:
                print(f"\n{os.path.basename(file)} cannot be mapped into a {arch_dims[0]}x{arch_dims[1]} architecture.\n")
                continue
            node_to_operation = UtilDFG.generate_node_to_operation_by_networkx_graph(dfg_graph.graph,"opcode")
            dfg = UtilInitialize.initialize_dfg_from_class_name(config.dfg_class_name,dfg_graph,node_to_operation)
            initial_state = UtilConfigs.get_mapping_state_by_model_name(config.model_name,dfg,cgra,dfg.get_next_node_to_be_mapped(),
                                            config.distance_func)
            states.append(initial_state)
        return states
    
    @staticmethod
    def play_game(config,model,initial_state,environment,num_max_expansion = None):
        """
        Play one game with actions based on the Monte Carlo tree search at each moves.
        """
        mapping_history = MappingHistory()
        observation = initial_state
        mapping_history.action_history.append(0)
        mapping_history.observation_history.append(observation)
        mapping_history.reward_history.append(0)

        done = False
        max_tree_depths = []
        backtracking_count = 0
        with torch.no_grad():
            while (
                not done and len(mapping_history.action_history) <= config.max_moves
            ):
                #  Choose the action
                root, mcts_info = MCTS(config).run(
                    model,
                    observation,
                    observation.get_legal_actions(),
                    False,
                    override_root_with=None,
                    num_max_expansion=num_max_expansion,
                    get_first_valid_solution=True
                )
               

                max_tree_depths.append(mcts_info['max_tree_depth'])
                if mcts_info['trajectory'] is not None:
                    trajectory = mcts_info['trajectory']
                    for node in trajectory[1:]:
                        mapping_history.action_history.append(node.father_action)
                        mapping_history.store_search_statistics(node,config.action_space)
                        mapping_history.observation_history.append(node.state)
                        mapping_history.reward_history.append(node.reward)
                    return mapping_history, max_tree_depths,True,backtracking_count
                count = 0
                invalid_actions = set()
                while count < len(root.children):
                    action = UtilEval.select_action(
                        root,invalid_actions
                    )

                    observation, reward, done = environment.step(root.state,action)
                    if not root.state.is_bad_reward(reward):
                        break
                    count+=1
                    backtracking_count += 1
                    invalid_actions.add(action)


                mapping_history.store_search_statistics(root, config.action_space)

                mapping_history.action_history.append(action)
                mapping_history.observation_history.append(observation)
                mapping_history.reward_history.append(reward)

        return mapping_history, max_tree_depths, False, backtracking_count
    @staticmethod
    def select_action(node,invalid_actions):
        visit_count = []
        actions = []
        for action,child in node.children.items():
            if action not in invalid_actions:
                visit_count.append(child.visit_count)
                actions.append(action)   

        visit_counts = numpy.array(
           visit_count, dtype="int32"
        )

        action = actions[numpy.argmax(visit_counts)]

        return action
    