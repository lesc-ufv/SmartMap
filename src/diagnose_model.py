import matplotlib.pyplot as plt
import numpy
import seaborn
import torch
from src.trajectory_info import Trajectoryinfo
from src.self_play import SelfPlay
from src.mcts.mcts import MCTS
from src.mcts.node import  Node

class DiagnoseModel:
    """
    Tools to understand the learned model.

    Args:
        weights: weights for the model to diagnose.

        config: configuration class instance related to the weights.
    """

    def __init__(self, model, checkpoint, config):
        self.config = config

        # Initialize the network
        self.model = model
        self.model.set_weights(checkpoint["weights"])
        self.model.to(
            torch.device("cuda" if torch.cuda.is_available() and config.train_on_gpu else "cpu")
        )  # on GPU if available since the DataParallel objects in MuZeroNetwork requires that
        self.model.eval()


    def get_real_trajectories(
        self, first_obs, horizon, environment,plot=True,
    ):
        """
        First, MuZero plays a game but uses its model instead of using the environment.
        Then, MuZero plays the optimal trajectory according precedent trajectory but performs it in the
        real environment until arriving at an action impossible in the real environment.
        It does an MCTS too, but doesn't take it into account.
        All information during the two trajectories are recorded and displayed.
        """

        real_trajectory_info = Trajectoryinfo("Real trajectory", self.config)
        trajectory_divergence_index = None
        real_trajectory_end_reason = "Reached horizon"

        # Illegal moves are masked at the root
        root, mcts_info = MCTS(self.config).run(
            self.model,
            first_obs,
            first_obs.get_legal_actions(),
            False,
            num_max_expansion=self.config.num_max_expansion_test,
            get_first_valid_solution=True
        )
        self.plot_mcts(root, plot)
        real_trajectory_info.store_info(root, mcts_info, None, numpy.NaN)
        for i in range(horizon):
            state = root.state
            action = SelfPlay.select_action(root, 0)      
            # Follow virtual trajectory until it reaches an illegal move in the real env
            if action not in first_obs.get_legal_actions():
                action = SelfPlay.select_action(root, 0)
                if trajectory_divergence_index is None:
                    trajectory_divergence_index = i
                    real_trajectory_end_reason = f"Virtual trajectory reached an illegal move at timestep {trajectory_divergence_index}."
                break  # Comment to keep playing after trajectory divergence

            observation, reward, done = environment.step(state,action)
            if done:
                real_trajectory_end_reason = "Real trajectory reached Done"
                break
            root, mcts_info = MCTS(self.config).run(
                self.model,
                observation,
                observation.get_legal_actions(),
                False,
                num_max_expansion=self.config.num_max_expansion_test,
                get_first_valid_solution=True
            )
            real_trajectory_info.store_info(root, mcts_info, action, reward)

        if plot:
            # virtual_trajectory_info.plot_trajectory()
            real_trajectory_info.plot_trajectory()
            print(real_trajectory_end_reason)

        return (
            # virtual_trajectory_info,
            real_trajectory_info,
            trajectory_divergence_index,
        )

    def close_all(self):
        plt.close("all")

    def plot_mcts(self, root, plot=True):
        """
        Plot the MCTS, pdf file is saved in the current directory.
        """
        try:
            from graphviz import Digraph
        except ModuleNotFoundError:
            print("Please install graphviz to get the MCTS plot.")
            return None

        graph = Digraph(comment="MCTS", engine="neato")
        graph.attr("graph", rankdir="LR", splines="true", overlap="false")
        id = 0

        def traverse(node, action, parent_id, best):
            nonlocal id
            node_id = id
            graph.node(
                str(node_id),
                label=f"DFG Node: {node.state.dfg.base_dfg.reseted_to_real_node[node.state.id_node_to_be_mapped] if node.state.id_node_to_be_mapped is not None else None}\n Action: {action}\nValue: {node.value():.2f}\nVisit count: {node.visit_count}\nPrior: {node.prior:.2f}\nReward: {node.reward:.2f}",
                color="orange" if best else "black",
            )
            id += 1
            if parent_id is not None:
                graph.edge(str(parent_id), str(node_id), constraint="false")

            if len(node.children) != 0:
                best_visit_count = max(
                    [child.visit_count for child in node.children.values()]
                )
            else:
                best_visit_count = False
            
            for action, child in node.children.items():
                if child.visit_count != 0:
                    traverse(
                        child,
                        action,
                        node_id,
                        True
                        if best_visit_count and child.visit_count == best_visit_count
                        else False,
                    )

        traverse(root, None, None, True)
        graph.node(str(0), color="red")
        graph.render("mcts", view=plot, cleanup=True, format="pdf")
        return graph
    
    