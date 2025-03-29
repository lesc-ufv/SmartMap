import numpy
from copy import deepcopy
class MappingHistory:
    """
    Store only usefull information of a self-play game.
    """
    def __init__(self):
        self.observation_history = []
        self.action_history = []
        self.reward_history = []
        self.child_visits = []
        self.root_values = []
        self.reanalysed_predicted_root_values = None
        # For PER
        self.priorities = None
        self.game_priority = None
    
    def print_history(self):
        for i in range(len(self.action_history)):
            print(f'State {i}')
            self.observation_history[i].print_state()
            if i< len(self.root_values):
                print('Root Value')
                print(self.root_values[i])
            print()
            if i + 1 < len(self.action_history):
                print('Action | Reward')
                print(self.action_history[i+1],self.reward_history[i+1])
                print()
            if i < len(self.child_visits):
                print('Child Visits')
                print(self.child_visits[i])
                print()
            
        
        print('Reanalysed predicted root values')
        print(self.reanalysed_predicted_root_values)
        print()

        print('Priorities')
        print(self.priorities)
        print()
        
        print('Game Priority')
        print(self.game_priority)
        print()

    def store_search_statistics(self, root, action_space):
        # Turn visit count from root into a policy
        if root is not None:
            state = root.state
            action_to_index_policy_logits = state.get_action_to_index_policy_logits()
            sum_visits = sum(child.visit_count for child in root.children.values())
            child_visits = [0 for _ in range(len(action_space))]
            
            for a in root.children:
                child_visits[action_to_index_policy_logits[a]] = root.children[a].visit_count / sum_visits            
            
            self.child_visits.append(child_visits)
            self.root_values.append(root.value())
        else:
            self.root_values.append(None)
    def get_stacked_observations(
        self, index, num_stacked_observations, action_space_size
    ):
        """
        Generate a new observation with the observation at the index position
        and num_stacked_observations past observations and actions stacked.
        """
        index = index % len(self.observation_history)

        return self.observation_history[index]
