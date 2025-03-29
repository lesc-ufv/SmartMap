import pandas as pd
import os

class MappingResults:
    keys = [
        "model_name", "arch_dims", "arch_interconnections", "dfg_name","test_mode",
        "mapping_is_valid", "routing_penalty", "unsuccessful_reason", "mapping_time",
        "mean_visited_rate", "mean_num_visited_nodes", "mean_mean_expanded_nodes", "mean_max_tree_depths",
        "num_simulations","used_PEs", 'num_backtrackings'
    ]

    def __init__(self, model_name, arch_dims, arch_interconnections, dfg_name,test_mode):
        self.path_to_results = 'results/mapping_results.csv'
        self.model_name = model_name
        self.arch_dims = str(arch_dims)
        self.arch_interconnections = arch_interconnections
        self.dfg_name = dfg_name
        self.test_mode = test_mode 
        if os.path.exists(self.path_to_results):
            self.df = pd.read_csv(self.path_to_results)
        else:
            dic = {k: [] for k in self.keys}
            self.df = pd.DataFrame(dic)
        self.mapping_is_valid = None
        self.routing_penalty = None
        self.unsuccessful_reason = None
        self.mapping_time = None
        self.mean_visited_rate = None
        self.mean_num_visited_nodes = None
        self.mean_mean_expanded_nodes = None
        self.mean_max_tree_depths = None
        self.num_simulations = None
        self.used_PEs = None
        self.num_backtrackings = None

    def update_csv(self):
        filter_mask = (
            (self.df['model_name'] == self.model_name) &
            (self.df['arch_dims'] == self.arch_dims) &
            (self.df['arch_interconnections'] == self.arch_interconnections) &
            (self.df['dfg_name'] == self.dfg_name) &
           (self.df['test_mode'] == self.test_mode)
        )
        row_index = self.df.index[filter_mask]
        
        if len(row_index) == 0:
            new_row = [self.model_name, self.arch_dims, self.arch_interconnections, self.dfg_name,self.test_mode] + [None] * 11
            self.df.loc[len(self.df)] = new_row
            row_index = [len(self.df) - 1]
        
        for key in self.keys:
            value = getattr(self, key)
            if value is not None:
                self.df.loc[row_index[0], key] = value
                
    def save_csv(self):
        self.update_csv()
        self.df.to_csv(self.path_to_results, index=False)