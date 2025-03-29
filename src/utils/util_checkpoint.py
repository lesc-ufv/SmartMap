import torch
class UtilCheckpoint:
  
    @staticmethod
    def get_checkpoint  ():
        checkpoint = {
        "weights": None,
        "optimizer_state": None,
        "routing_penalty": 0,
        "reward": 0,
        "episode_length": 0,
        "mean_value": 0,
        "training_step": 0,
        "lr": 0,
        "total_loss": 0,
        "value_loss": 0,
        "policy_loss": 0,
        "num_played_games": 0,
        "num_played_steps": 0,
        "num_reanalysed_games": 0,
        "terminate": False,
        "mapping_is_valid": False,
        'mapping':None,
        "mean_reward": 0
        }
        return checkpoint