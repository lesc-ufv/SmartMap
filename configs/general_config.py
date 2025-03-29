from src.config import Config
import torch
from src.utils.softmax_temperature import SoftmaxTemperature
from src.enums.enum_interconnect_style import EnumInterconnectStyle
from src.enums.enum_mode import EnumMode
from src.utils.util_configs_train import UtilConfigsTrain
class GeneralConfig(Config):
    def __init__(self, model_name,arch_name,enum_interconnections_cgra, 
                 environment,replay_buffer_size,num_simulations,dataset_range,
                 arch_dims, class_model,model_instance_args,interconnections,
                 dfg_class_name,cgra_class_name,mode):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        gpu_flag = self.device == 'cuda'
        relative_dir = f'{model_name.value}/{arch_dims[0]}x{arch_dims[1]}_{arch_name}'
        action_space = arch_dims[0]*arch_dims[1]

        self.len_action_space = action_space
        
        self.mode = mode

        self.max_moves = action_space
        self.num_unroll_steps = action_space
        self.action_space = [i for i in range(action_space)]
        self.enum_interconnections_cgra = enum_interconnections_cgra
        self.arch_name = arch_name
        self.seed = 1234
        self.train_on_gpu = gpu_flag
        self.reanalyse_on_gpu = False
        self.selfplay_on_gpu =  False
        self.max_num_gpus = 1
        
        self.self_play_delay = 0
        self.training_delay = 0  
        self.ratio = 0.2

        self.dtype = torch.float32

        self.optimizer = "Adam"
        self.weight_decay = 0.0001
        self.momentum = 0.9
        self.lr_init = 0.01   
        self.lr_decay_rate = 0.95
        self.PER = False
        self.PER_alpha = 1

        self.results_path = f'results/{relative_dir}/'
        
        self.use_last_model_value = True
        self.num_workers = 10
        self.batch_size = 32
        self.discount = 0.997
        self.temperature_threshold = None

        self.replay_buffer_size = 2*self.batch_size
        self.value_loss_weight = 1  
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25
        self.pb_c_base = 19652
        self.pb_c_init = 1.25
        
        self.td_steps = int(0.25 * action_space) if mode == EnumMode.DATA_GENERATION or mode == EnumMode.TEST else action_space
        self.stacked_observations = 0
        
        self.checkpoint_interval = 1

        self.visit_softmax_temperature_fn = SoftmaxTemperature.visit_softmax_temperature_fn
        
        self.num_max_expansion_train = 200
        self.num_max_expansion_test = 100
        self.training_steps = 500
        self.lr_decay_steps = 100
        
        
        self.num_simulations = num_simulations
        self.dataset_range = dataset_range
        self.environment = environment
        self.model_name = model_name
        self.path_to_train_dataset = 'benchmarks/synthetics/'
        self.path_to_test_dataset = 'benchmarks/changed_MCTS_benchmark/'

        self.path_to_logs = f'logs/{relative_dir}/' 
        self.arch_dims = arch_dims
        self.class_model = class_model
        self.model_instance_args = model_instance_args
        self.interconnections = interconnections
        self.interconnection_style = EnumInterconnectStyle.NEIGH_TO_NEIGH
        self.dfg_class_name = dfg_class_name
        self.cgra_class_name = cgra_class_name
        self.num_last_buffers_to_save = 1

        self.save_model = False if mode == EnumMode.DATA_GENERATION or mode == EnumMode.TEST else True
        self.save_replay_buffer = True if mode == EnumMode.DATA_GENERATION else False
        self.epochs = UtilConfigsTrain.get_epochs_by_arch_dims(self.arch_dims)
    
    
        
        