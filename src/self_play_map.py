import math
import time

import numpy
import ray
import torch
from src.mcts.mcts import MCTS
from src.mapping_history import MappingHistory
from src.config import Config
from src.utils.util_eval import UtilEval
@ray.remote
class SelfPlay:
    """
    Class which run in a dedicated thread to play games and save them to the replay-buffer.
    """

    def __init__(self, initial_checkpoint, model, environment, config,initial_state, seed):
        self.config :Config = config
        # Fix random generator seed
        config.seed = seed
        numpy.random.seed(seed)
        torch.manual_seed(seed)

        # Initialize the network
        self.model = model
        self.environment = environment
        self.model.set_weights(initial_checkpoint["weights"])
        self.model.to(torch.device("cuda" if self.config.selfplay_on_gpu else "cpu"))
        self.model.eval()
        self.initial_state = initial_state
        self.test_mode = None

    def continuous_self_play(self, shared_storage,replay_buffer,test_mode=False,get_first_valid_solution=True):
        self.test_mode = test_mode
        num_max_expansion = self.config.num_max_expansion_test if test_mode else self.config.num_max_expansion_train
        while ray.get(
            shared_storage.get_info.remote("training_step")
        ) < self.config.training_steps and not ray.get(
            shared_storage.get_info.remote("terminate")
        ):
            self.model.set_weights(ray.get(shared_storage.get_info.remote("weights")))
            
            if not test_mode:
                game_history = self.play_game(
                    self.config.visit_softmax_temperature_fn(
                        ray.get(
                            shared_storage.get_info.remote("training_step")
                        ),
                        
                        self.config.training_steps)
                    ,
                    self.config.temperature_threshold,
                    get_first_valid_solution=get_first_valid_solution,
                    num_max_expansion=num_max_expansion
                )
                replay_buffer.save_game.remote(game_history, shared_storage)

            else:
                # Take the best action (no exploration) in test mode
                game_history = self.play_game(
                    0,
                    self.config.temperature_threshold,
                    get_first_valid_solution = get_first_valid_solution,
                    num_max_expansion=num_max_expansion
                )

                # Save to the shared storage
                shared_storage.set_info.remote(
                    {
                        "episode_length": len(game_history.action_history) - 1,
                        "routing_penalty": sum(game_history.reward_history),
                        "mean_value": numpy.mean(
                            [value for value in game_history.root_values if value]
                        ),
                        "mean_reward":numpy.mean(
                            [value for value in game_history.reward_history]),
                    }
                )

            if game_history.observation_history[-1].mapping_is_valid:
                shared_storage.set_info.remote({'mapping':game_history,'terminate':True})
               
            if not test_mode and self.config.self_play_delay:
                    time.sleep(self.config.self_play_delay)
            

            if not test_mode and self.config.ratio:
                training_step = ray.get(shared_storage.get_info.remote("training_step"))
                played_games = ray.get(shared_storage.get_info.remote("num_played_games"))
                norm = self.config.adjust_ratio(training_step, played_games)

                while (
                 norm*training_step
                    / max(
                        1, played_games
                    )
                    < self.config.ratio
                    and ray.get(shared_storage.get_info.remote("training_step"))
                    < self.config.training_steps
                    and not ray.get(shared_storage.get_info.remote("terminate"))
                ):
                    training_step = ray.get(shared_storage.get_info.remote("training_step"))
                    played_games = ray.get(shared_storage.get_info.remote("num_played_games"))
                    norm = self.config.adjust_ratio(training_step, played_games)

            


    def play_game(
        self, temperature, temperature_threshold, get_first_valid_solution,num_max_expansion = None
    ):
        """
        Play one game with actions based on the Monte Carlo tree search at each moves.
        """
        mapping_history = MappingHistory()
        observation = self.initial_state
        mapping_history.action_history.append(0)
        mapping_history.observation_history.append(observation)
        mapping_history.reward_history.append(0)

        done = False

        with torch.no_grad():
            while (
                not done and len(mapping_history.action_history) <= self.config.max_moves
            ):
            # Choose the action
                root, mcts_info = MCTS(self.config).run(
                    self.model,
                    observation,
                    observation.get_legal_actions(),
                    False if self.test_mode else not temperature == 0,
                    override_root_with=None,
                    num_max_expansion=num_max_expansion,
                    get_first_valid_solution=get_first_valid_solution,
                )
                 
                if mcts_info['trajectory'] is not None:
                    assert get_first_valid_solution
                    trajectory = mcts_info['trajectory']
                    for node in trajectory[1:]:
                        mapping_history.action_history.append(node.father_action)
                        mapping_history.store_search_statistics(node, self.config.action_space)
                        mapping_history.observation_history.append(node.state)
                        mapping_history.reward_history.append(node.reward)
                    # print('Solution found before finish the search')
                    return mapping_history

                if not self.test_mode:
                    action = self.select_action(
                        root,
                        temperature
                        if not temperature_threshold
                        or len(mapping_history.action_history) < temperature_threshold
                        else 0,
                    )
    
                    observation, reward, done = self.environment.step(root.state,action)
                else:
                    count = 0
                    invalid_actions = set()
                    while count < len(root.children):
                        action = UtilEval.select_action(
                            root,invalid_actions
                        )

                        observation, reward, done = self.environment.step(root.state,action)
                        if not root.state.is_bad_reward(reward):
                            break
                        count+=1
                        invalid_actions.add(action)


                mapping_history.store_search_statistics(root, self.config.action_space)

                # Next batch
                mapping_history.action_history.append(action)
                mapping_history.observation_history.append(observation)
                mapping_history.reward_history.append(reward)
        return mapping_history


    @staticmethod
    def select_action(node, temperature):
        """
        Select action according to the visit count distribution and the temperature.
        The temperature is changed dynamically with the visit_softmax_temperature function
        in the config.
        """
        visit_counts = numpy.array(
            [child.visit_count for child in node.children.values()], dtype="int32"
        )
        actions = [action for action in node.children.keys()]
        if temperature == 0:
            action = actions[numpy.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = numpy.random.choice(actions)
        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(
                visit_count_distribution
            )
            action = numpy.random.choice(actions, p=visit_count_distribution)

        return action


