from src.config import Config
import numpy
import ray
import torch
import math
import os
import pathlib
import pickle
from src.diagnose_model import DiagnoseModel
from src.reanalyse import Reanalyse
from src.trainer import Trainer
from src.shared_storage import SharedStorage
from src.self_play import SelfPlay
from src.replay_buffer import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import copy
from src.cpu_actor import CPUActor
import time
from src.utils.util_replay_buffer import UtilReplayBuffer
import warnings
class ModelLauncher:
    def __init__(self,model,dfg_name,config: Config, initial_state, environment, checkpoint, path_replay_buffer, split_resources_in=1):
        warnings.filterwarnings("ignore", category=FutureWarning)
        self.config = config
        self.model = model
        self.environment = environment


        numpy.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        if self.config.max_num_gpus == 0 and (
            self.config.selfplay_on_gpu
            or self.config.train_on_gpu
            or self.config.reanalyse_on_gpu
        ):
            raise ValueError(
                "Inconsistent Config: max_num_gpus = 0 but GPU requested by selfplay_on_gpu or train_on_gpu or reanalyse_on_gpu."
            )
        if (
            self.config.selfplay_on_gpu
            or self.config.train_on_gpu
            or self.config.reanalyse_on_gpu
        ):
            total_gpus = (
                self.config.max_num_gpus
                if self.config.max_num_gpus is not None
                else torch.cuda.device_count()
            )
        else:
            total_gpus = 0
        self.num_gpus = total_gpus / split_resources_in
        if 1 < self.num_gpus:
            self.num_gpus = math.floor(self.num_gpus)
        
        ray.init(num_gpus=total_gpus, local_mode= False, ignore_reinit_error=True)

        self.checkpoint = checkpoint
  
        os.makedirs(self.config.results_path, exist_ok=True)
        
        self.replay_buffer = {}
        self.path_replay_buffer = path_replay_buffer
        self.initial_state = initial_state

        self.dfg_name = dfg_name
        # Workers
        self.self_play_workers = None
        self.test_worker = None
        self.training_worker = None
        self.reanalyse_worker = None
        self.replay_buffer_worker = None
        self.shared_storage_worker = None

    def train(self, log_in_tensorboard=True):
        """
        Spawn ray workers and launch the training.

        Args:
            log_in_tensorboard (bool): Start a testing worker and log its performance in TensorBoard.
        """
        if log_in_tensorboard or self.config.save_model or self.config.save_replay_buffer:
            os.makedirs(self.config.results_path,exist_ok=True)

        # Manage GPUs
        if 0 < self.num_gpus:
            num_gpus_per_worker = self.num_gpus / (
                self.config.train_on_gpu
                + self.config.num_workers * self.config.selfplay_on_gpu
                + log_in_tensorboard * self.config.selfplay_on_gpu
                + self.config.use_last_model_value * self.config.reanalyse_on_gpu
            )
            if 1 < num_gpus_per_worker:
                num_gpus_per_worker = math.floor(num_gpus_per_worker)
        else:
            num_gpus_per_worker = 0

        # Initialize workers
        self.training_worker = Trainer.options(
            num_cpus=0,
            num_gpus=num_gpus_per_worker if self.config.train_on_gpu else 0,
        ).remote(self.model,self.checkpoint, self.config)

        self.shared_storage_worker = SharedStorage.remote(
            self.checkpoint,
            self.config,
        )
        
        self.shared_storage_worker.set_info.remote("terminate", False)

        self.replay_buffer_worker = ReplayBuffer.remote(
            self.checkpoint, UtilReplayBuffer.get_replay_buffer(self.path_replay_buffer)["buffer"] if self.path_replay_buffer else {} , self.config
        )

        if self.config.use_last_model_value:
            self.reanalyse_worker = Reanalyse.options(
                num_cpus=0,
                num_gpus=num_gpus_per_worker if self.config.reanalyse_on_gpu else 0,
            ).remote(self.checkpoint,self.model, self.config)

        self.self_play_workers = [
            SelfPlay.options(
                num_cpus=0,
                num_gpus=num_gpus_per_worker if self.config.selfplay_on_gpu else 0,
            ).remote(
                self.checkpoint,
                self.model,
                self.environment,
                copy.deepcopy(self.config),
                self.initial_state,
                self.config.seed + seed,
            )
            for seed in range(self.config.num_workers)
        ]

        # Launch workers
        [
            self_play_worker.continuous_self_play.remote(
                self.shared_storage_worker, self.replay_buffer_worker
            )
            for self_play_worker in self.self_play_workers
        ]
        self.training_worker.continuous_update_weights.remote(
            self.replay_buffer_worker, self.shared_storage_worker
        )
        if self.config.use_last_model_value:
            self.reanalyse_worker.reanalyse.remote(
                self.replay_buffer_worker, self.shared_storage_worker
            )

        if log_in_tensorboard:
            self.logging_loop(
                num_gpus_per_worker if self.config.selfplay_on_gpu else 0,
            )


    def logging_loop(self, num_gpus):
        """
        Keep track of the training performance.
        """
        # Launch the test worker to get performance metrics
        self.test_worker = SelfPlay.options(
            num_cpus=0,
            num_gpus=num_gpus,
        ).remote(
            self.checkpoint,
            self.model,
            self.environment,
            self.config,
            self.initial_state,
            self.config.seed + self.config.num_workers,
        )
        self.test_worker.continuous_self_play.remote(
            self.shared_storage_worker, None, True, False
        )

        # Write everything in TensorBoard
        writer = SummaryWriter(self.config.results_path+self.dfg_name)

        print(
            f"\nTraining...\nRun tensorboard --logdir {self.config.results_path +self.dfg_name} and go to http://localhost:6006/ to see in real time the training performance.\n"
        )

        # Save hyperparameters to TensorBoard
        hp_table = [
            f"| {key} | {value} |" for key, value in self.config.__dict__.items()
        ]
        writer.add_text(
            "Hyperparameters",
            "| Parameter | Value |\n|-------|-------|\n" + "\n".join(hp_table),
        )
        # Save model representation
        writer.add_text(
            "Model summary",
            str(self.model).replace("\n", " \n\n"),
        )
        # Loop for updating the training performance
        counter = 0
        keys = [
            "reward",
            "episode_length",
            "mean_value",
            "training_step",
            "lr",
            "routing_penalty",
            "total_loss",
            "value_loss",
            "policy_loss",
            "num_played_games",
            "num_played_steps",
            "num_reanalysed_games",
            "mapping_is_valid",
            "mean_reward"
        ]
        info = ray.get(self.shared_storage_worker.get_info.remote(keys))
        try:
            while info["training_step"] < self.config.training_steps:
                info = ray.get(self.shared_storage_worker.get_info.remote(keys))

                writer.add_scalar(
                    "Routing_penalty/Step",
                    info["routing_penalty"],
                    counter,
                )
                writer.add_scalar(
                    "Mean_value/Step",
                    info["mean_value"],
                    counter,
                )

                writer.add_scalar(
                    "Mean_reward/Step",
                    info["mean_reward"],
                    counter,
                )
        
                writer.add_scalar("Learning_rate/Step", info["lr"], counter)
                writer.add_scalar(
                    "Total_loss/Step", info["total_loss"], counter
                )
                writer.add_scalar("Value_loss/Step", info["value_loss"], counter)
                writer.add_scalar("Policy_loss/Step", info["policy_loss"], counter)
                print(
                    f'Routing Penalty (Test): {info["routing_penalty"]:.2f}. Successful Mapping: {info["mapping_is_valid"]}.  Training step: {info["training_step"]}/{self.config.training_steps}. Played games: {info["num_played_games"]}. Total Loss: {info["total_loss"]:.2f}. Value Loss: {info["value_loss"]:.2f}. Policy Loss: {info["policy_loss"]:.2f}. lr: {info["lr"]:.6f}',
                    end="\r",
                )
                counter += 1
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass

        self.terminate_workers()

        if self.config.save_replay_buffer:
            # Persist replay buffer to disk
            path = self.config.results_path + self.dfg_name + os.sep
            os.makedirs(path,exist_ok=True)

            path += "replay_buffer.pkl"
            print(f"\n\nPersisting replay buffer games to disk at {path}")
            pickle.dump(
                {
                    "buffer": dict(list(self.replay_buffer.items())[-self.config.num_last_buffers_to_save:]) if self.config.num_last_buffers_to_save else self.replay_buffer ,
                    "num_played_games": self.checkpoint["num_played_games"],
                    "num_played_steps": self.checkpoint["num_played_steps"],
                    "num_reanalysed_games": self.checkpoint["num_reanalysed_games"],
                },
                open(path, "wb"),
            )

    def terminate_workers(self):
        """
        Softly terminate the running tasks and garbage collect the workers.
        """
        if self.shared_storage_worker:
            self.shared_storage_worker.set_info.remote("terminate", True)
            self.checkpoint = ray.get(
                self.shared_storage_worker.get_checkpoint.remote()
            )
            self.shared_storage_worker.save_checkpoint.remote()
        

        if self.replay_buffer_worker:
            self.replay_buffer = ray.get(self.replay_buffer_worker.get_buffer.remote())

        print("\nShutting down workers...")

        self.self_play_workers = None
        self.test_worker = None
        self.training_worker = None
        self.reanalyse_worker = None
        self.replay_buffer_worker = None
        self.shared_storage_worker = None
        ray.shutdown()
    def test(
        self,initial_state, num_tests=1, num_gpus=0
    ):
        """
        Test the model in a dedicated thread.

        Args:
            render (bool): To display or not the environment. Defaults to True.

            opponent (str): "self" for self-play, "human" for playing against MuZero and "random"
            for a random agent, None will use the opponent in the config. Defaults to None.

            muzero_player (int): Player number of MuZero in case of multiplayer
            games, None let MuZero play all players turn by turn, None will use muzero_player in
            the config. Defaults to None.

            num_tests (int): Number of games to average. Defaults to 1.

            num_gpus (int): Number of GPUs to use, 0 forces to use the CPU. Defaults to 0.
        """
        self_play_worker = SelfPlay.options(
            num_cpus=0,
            num_gpus=num_gpus,
        ).remote(self.checkpoint, self.model, self.environment,self.config,initial_state, self.config.seed)
        results = []
        for i in range(num_tests):
            print(f"Testing {i+1}/{num_tests}")
            results.append(
                ray.get(
                    self_play_worker.play_game.remote(
                        0,
                        0,
                        True,
                        num_max_expansion =  self.config.num_max_expansion_test,

                    )
                )
            )

        result = numpy.mean([sum(history.reward_history) for history in results])
        
        n_rows,n_cols = self.checkpoint['cgra']['arch_dims']
        matrix = [[-1 for j in range(n_cols)] for i in range(n_rows)]

        for i,history in enumerate(results):
            state = history.observation_history[-1]
            print(f"Results {i+1}/{num_tests}")
            print()
            print(state.cgra.cgra.pes_to_routing)
            map_nodes = state.dfg.base_dfg.reseted_to_real_node
            print('Placement')
            for k,v in state.cgra.get_pes_to_node_id().items():
                i_pos = k // n_rows
                j_pos = k % n_cols
                matrix[i_pos][j_pos] = map_nodes[v] if v != -1 and v != -2 else "R" if v == -2 else v
            for row in matrix:
                print(row)
            print('Roulting Penalty:', result)
        return result


    def diagnose_model(self, horizon):
        """
        Play a game only with the learned model then play the same trajectory in the real
        environment and display information.

        Args:
            horizon (int): Number of timesteps for which we collect information.
        """
        dm = DiagnoseModel(self.model,self.checkpoint, self.config)
        dm.get_real_trajectories(self.initial_state, horizon,self.environment)
        input("Press enter to close all plots")
        dm.close_all()