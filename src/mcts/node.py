
import torch
import numpy
class Node:
    def __init__(self, prior):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.state = None
        self.reward = 0
        self.father = None
        self.father_action = None
    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(self, actions, reward, policy_logits, state, is_end_state,num_max_expansion = None):
        """
        We expand a node using the value, reward and policy prediction obtained from the
        neural network.
        """
        self.reward = reward
        self.state = state
        action_to_index_legal_action = state.get_action_to_index_policy_logits()
        if not is_end_state:
            policy_values = torch.softmax(
                torch.tensor([policy_logits[0][action_to_index_legal_action[a]] for a in actions]), dim=0
            )
            
            num_expansion = min(num_max_expansion if num_max_expansion else float('inf'),len(actions))
            policy_values,indices = torch.topk((policy_values), num_expansion)
            policy_values = policy_values.tolist()
            policy = {actions[a]: policy_values[i] for i, a in enumerate(indices)}

            for action, p in policy.items():
                self.children[action] = Node(p)
                self.children[action].father = self
                self.children[action].father_action = action

    def add_exploration_noise(self, dirichlet_alpha, exploration_fraction):
        """
        At the start of each search, we add dirichlet noise to the prior of the root to
        encourage the search to explore new actions.
        """
        actions = list(self.children.keys())
        noise = numpy.random.dirichlet([dirichlet_alpha] * len(actions))
        frac = exploration_fraction
        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac