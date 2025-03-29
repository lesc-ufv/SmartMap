import math
import numpy
from src.config import Config
import torch
from src.mcts.node import Node
from src.utils.min_max_stats import MinMaxStats
from src.rl.states.mapping_state_interface import MappingStateInterface
from src.config import Config
class MCTS:
    """
    Core Monte Carlo Tree Search algorithm.
    To decide on an action, we run N simulations, always starting at the root of
    the search tree and traversing the tree according to the UCB formula until we
    reach a leaf node.
    """

    def __init__(self, config : Config):
        self.config :Config = config
 
    def run(
        self,
        model,
        state:MappingStateInterface,
        legal_actions,
        add_exploration_noise,
        override_root_with=None,
        num_max_expansion = None,
        get_first_valid_solution = False
    ):
        """
        At the root of the search tree we use the representation function to obtain a
        hidden state given the current observation.
        We then run a Monte Carlo Tree Search using only action sequences and the model
        learned by the network.
        """
        if override_root_with:
            root = override_root_with
            root_predicted_value = None
        else:
            root = Node(0)
            current_state = state
            (
                policy_logits,
                root_predicted_value,
                reward,
            ) = model.initial_inference([current_state])

            assert set(legal_actions).issubset(
                set(self.config.action_space)
            ), "Legal actions should be a subset of the action space."
            reward = reward.item()
            root.expand(
                legal_actions,
                reward,
                policy_logits,
                current_state,
                current_state.is_end_state,
                num_max_expansion
            )

        if add_exploration_noise:
            root.add_exploration_noise(
                dirichlet_alpha=self.config.root_dirichlet_alpha,
                exploration_fraction=self.config.root_exploration_fraction,
            )
        max_tree_depth = 0
        min_max_stats = MinMaxStats()
        
        for _ in range(self.config.num_simulations):
            node = root
            search_path = [node]
            current_tree_depth = 0

            while node.expanded():
                current_tree_depth += 1
                action, node = self.select_child(node, min_max_stats)
                search_path.append(node)

            parent = search_path[-2]
            value, reward, policy_logits, hidden_state = model.recurrent_inference(
                [parent.state],
            [action],
            )
            value = value.item()
            reward = reward.item()

            node.expand(
                hidden_state[0].get_legal_actions(),
                reward,
                policy_logits,
                hidden_state[0],
                hidden_state[0].is_end_state,
                num_max_expansion
            )

            self.backpropagate(search_path, value, min_max_stats)

            max_tree_depth = max(max_tree_depth, current_tree_depth)
            last_node = search_path[-1]
            if get_first_valid_solution and last_node.state.mapping_is_valid:
                assert last_node.state.is_end_state
                trajectory = [last_node]
                cur_node = last_node
                while cur_node.father is not None:
                    trajectory.append(cur_node.father)
                    cur_node = cur_node.father
                trajectory.reverse()
                extra_info = {
                    "max_tree_depth": max_tree_depth,
                    "root_predicted_value": root_predicted_value,
                    "trajectory": trajectory,                
                }
                return root, extra_info
        extra_info = {
            "max_tree_depth": max_tree_depth,
            "root_predicted_value": root_predicted_value,
            "trajectory": None
        }
        return root, extra_info

    def select_child(self, node, min_max_stats):
        """
        Select the child with the highest UCB score.
        """
        max_ucb = max(
            self.ucb_score(node, child, min_max_stats)
            for action, child in node.children.items()
        )
        action = numpy.random.choice(
            [
                action
                for action, child in node.children.items()
                if self.ucb_score(node, child, min_max_stats) == max_ucb
            ]
        )
        return action, node.children[action]

    def ucb_score(self, parent, child, min_max_stats):
        """
        The score for a node is based on its value, plus an exploration bonus based on the prior.
        """
        pb_c = (
            math.log(
                (parent.visit_count + self.config.pb_c_base + 1) / self.config.pb_c_base
            )
            + self.config.pb_c_init
        )
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior

        if child.visit_count > 0:
            # Mean value Q
            value_score = min_max_stats.normalize(
                child.reward
                + self.config.discount
                * (child.value())
            )
        else:
            value_score = 0

        return prior_score + value_score

    def backpropagate(self, search_path, value, min_max_stats):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            min_max_stats.update(node.reward + self.config.discount * node.value())

            value = node.reward + self.config.discount * value
    

