
import numpy
import matplotlib.pyplot as plt
import seaborn
class Trajectoryinfo:
    """
    Store the information about a trajectory (rewards, search information for every step, ...).
    """

    def __init__(self, title, config):
        self.title = title + ": "
        self.config = config
        self.action_history = []
        self.reward_history = []
        self.prior_policies = []
        self.policies_after_planning = []
        # Not implemented, need to store them in every nodes of the mcts
        self.prior_values = []
        self.values_after_planning = [[numpy.NaN] * len(self.config.action_space)]
        self.prior_root_value = []
        self.root_value_after_planning = []
        self.prior_rewards = [[numpy.NaN] * len(self.config.action_space)]
        self.mcts_depth = []

    def store_info(self, root, mcts_info, action, reward, new_prior_root_value=None):
        if action is not None:
            self.action_history.append(action)
        if reward is not None:
            self.reward_history.append(reward)
        self.prior_policies.append(
            [
                root.children[action].prior
                if action in root.children.keys()
                else numpy.NaN
                for action in self.config.action_space
            ]
        )
        self.policies_after_planning.append(
            [
                root.children[action].visit_count / self.config.num_simulations
                if action in root.children.keys()
                else numpy.NaN
                for action in self.config.action_space
            ]
        )
        self.values_after_planning.append(
            [
                root.children[action].value()
                if action in root.children.keys()
                else numpy.NaN
                for action in self.config.action_space
            ]
        )
        self.prior_root_value.append(
            mcts_info["root_predicted_value"].item()
            if not new_prior_root_value
            else new_prior_root_value.item()
        )
        self.root_value_after_planning.append(root.value())
        self.prior_rewards.append(
            [
                root.children[action].reward
                if action in root.children.keys()
                else numpy.NaN
                for action in self.config.action_space
            ]
        )
        self.mcts_depth.append(mcts_info["max_tree_depth"])

    def plot_trajectory(self):
        name = "Prior policies"
        print(name, self.prior_policies, "\n")
        plt.figure(self.title + name)
        ax = seaborn.heatmap(
            self.prior_policies,
            mask=numpy.isnan(self.prior_policies),
            annot=True,
        )
        ax.set(xlabel="Action", ylabel="Timestep")
        ax.set_title(name)

        name = "Policies after planning"
        print(name, self.policies_after_planning, "\n")
        plt.figure(self.title + name)
        ax = seaborn.heatmap(
            self.policies_after_planning,
            mask=numpy.isnan(self.policies_after_planning),
            annot=True,
        )
        ax.set(xlabel="Action", ylabel="Timestep")
        ax.set_title(name)

        if 0 < len(self.action_history):
            name = "Action history"
            print(name, self.action_history, "\n")
            plt.figure(self.title + name)
            ax = seaborn.heatmap(
                numpy.transpose([self.action_history]),
                mask=numpy.isnan(numpy.transpose([self.action_history])),
                xticklabels=False,
                annot=True,
            )
            ax.set(ylabel="Timestep")
            ax.set_title(name)

        name = "Values after planning"
        print(name, self.values_after_planning, "\n")
        plt.figure(self.title + name)
        ax = seaborn.heatmap(
            self.values_after_planning,
            mask=numpy.isnan(self.values_after_planning),
            annot=True,
        )
        ax.set(xlabel="Action", ylabel="Timestep")
        ax.set_title(name)

        name = "Prior root value"
        print(name, self.prior_root_value, "\n")
        plt.figure(self.title + name)
        ax = seaborn.heatmap(
            numpy.transpose([self.prior_root_value]),
            mask=numpy.isnan(numpy.transpose([self.prior_root_value])),
            xticklabels=False,
            annot=True,
        )
        ax.set(ylabel="Timestep")
        ax.set_title(name)

        name = "Root value after planning"
        print(name, self.root_value_after_planning, "\n")
        plt.figure(self.title + name)
        ax = seaborn.heatmap(
            numpy.transpose([self.root_value_after_planning]),
            mask=numpy.isnan(numpy.transpose([self.root_value_after_planning])),
            xticklabels=False,
            annot=True,
        )
        ax.set(ylabel="Timestep")
        ax.set_title(name)

        name = "Prior rewards"
        print(name, self.prior_rewards, "\n")
        plt.figure(self.title + name)
        ax = seaborn.heatmap(
            self.prior_rewards, mask=numpy.isnan(self.prior_rewards), annot=True
        )
        ax.set(xlabel="Action", ylabel="Timestep")
        ax.set_title(name)

        if 0 < len(self.reward_history):
            name = "Reward history"
            print(name, self.reward_history, "\n")
            plt.figure(self.title + name)
            ax = seaborn.heatmap(
                numpy.transpose([self.reward_history]),
                mask=numpy.isnan(numpy.transpose([self.reward_history])),
                xticklabels=False,
                annot=True,
            )
            ax.set(ylabel="Timestep")
            ax.set_title(name)

        name = "MCTS depth"
        print(name, self.mcts_depth, "\n")
        plt.figure(self.title + name)
        ax = seaborn.heatmap(
            numpy.transpose([self.mcts_depth]),
            mask=numpy.isnan(numpy.transpose([self.mcts_depth])),
            xticklabels=False,
            annot=True,
        )
        ax.set(ylabel="Timestep")
        ax.set_title(name)

        plt.show(block=False)