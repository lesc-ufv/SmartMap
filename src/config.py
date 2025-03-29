class Config:
    def __init__(self,):
        self.seed = None
        self.device = None
        self.train_on_gpu = None
        self.reanalyse_on_gpu = None
        self.optimizer = None
        self.lr_init = None
        self.weight_decay = None
        self.dtype = None
        self.num_simulations = None
        self.discount = None
        self.training_steps = None
        self.results_path = None
        self.momentum = None
        self.ratio = None
        self.self_play_delay = None
        self.use_last_model_value = None
        self.num_workers = None
        self.selfplay_on_gpu = None
        self.PER = None
        self.temperature_threshold = None
        self.replay_buffer_size = None
        self.action_space = None
        self.PER_alpha = None
        self.stacked_observations = None
        self.batch_size = None
        self.lr_decay_rate = None
        self.lr_decay_steps = None
        self.lr_init = None
        self.value_loss_weight = None
        self.checkpoint_interval = None
        self.self_play_delay = None
        self.save_model  = None
        self.training_delay = None
        self.max_num_gpus = None
        self.num_max_expansion_train = None
        self.num_max_expansion_test = None
        self.enum_interconnections_cgra = None
        self.dataset_range = None
        self.path_to_train_dataset = None
        self.path_to_eval_dataset = None
        self.environment = None
        self.model_name = None
        self.path_to_logs = None
        self.arch_dims = None
        self.arch_name = None
        self.class_model = None
        self.model_instance_args = None
        self.interconnection_style = None
        self.interconnections = None
        self.dfg_class_name = None
        self.cgra_class_name = None
        self.distance_func = None
        self.mode = None
        self.save_replay_buffer = None
        self.training_steps = None
        self.epochs = None
        self.td_steps = None
        self.first_ratio_trainstep = 0.3
        self.first_ratio = 0.2
        self.second_ratio_train = 0.6
        self.second_ratio = 0.5
        self.third_ratio = 1
        self.base_train_step = (self.second_ratio_train)/((self.first_ratio**-1)*self.first_ratio_trainstep + (self.second_ratio**-1)*(self.second_ratio_train-self.first_ratio_trainstep)) 
        self.base_train_step = (self.base_train_step**-1)*self.third_ratio
        self.len_action_space = None
        self.num_last_buffers_to_save = None

    def set_training_steps(self,training_steps):
        self.training_steps = training_steps
        self.lr_decay_steps = int(0.05*training_steps)
        if self.lr_decay_steps == 0:
            self.lr_decay_steps = self.training_steps


    def adjust_td_steps(self,training_step):
        len_action_space  = self.len_action_space

        if training_step/self.training_steps < 0.25:
            self.td_steps = int(0.25 * len_action_space)

        elif training_step/self.training_steps < 0.5:
            self.td_steps = int(0.5 * len_action_space)

        elif training_step/self.training_steps < 0.75:
            self.td_steps = int(0.75 * len_action_space)

        else:
            self.td_steps = len_action_space


    def adjust_ratio(self,training_step,played_games ):
        if training_step/self.training_steps < self.first_ratio_trainstep or played_games < self.first_ratio**-1 * self.first_ratio_trainstep * self.training_steps:
            self.ratio = self.first_ratio
            return 1

        if training_step/self.training_steps < self.second_ratio or  played_games < (self.first_ratio**-1 * (self.first_ratio_trainstep) + self.second_ratio**-1 *(self.second_ratio_train-self.first_ratio_trainstep)) * self.training_steps:

            self.ratio = self.second_ratio
            tsb = self.first_ratio_trainstep*self.training_steps
            return (tsb*(5/2 + training_step/tsb -1) )/training_step

        self.ratio = self.third_ratio
        tsb = self.second_ratio_train*self.training_steps
        return (tsb*(self.base_train_step + training_step/tsb - 1) )/training_step

