import pandas as pd
import os
class Results:
    keys = ["model_name", "arch_dims", "arch_interconnections", "mcts_depth", "mean_node_expansion",
            "sample_generation_time", "training_time","inference_time"," successful_mapping"]

    def __init__(self,model_name,arch_dims,arch_interconnections):
        self.path_to_results = 'results.csv'
        self.model_name = model_name
        self.arch_dims = arch_dims
        self.arch_interconnections = arch_interconnections
        if os.path.exists(self.path_to_results):
            self.df = pd.read_csv(self.path_to_results)
        else:
            dic = {k:[] for k in self.keys}
            self.df = pd.DataFrame(dic)
        self.df = pd.read_csv
        self.mcts_depth = None
        self.mean_node_expansion = None
        self.sample_generation_time = None
        self.training_time = None
        self.mean_inference_time = None
        self.successful_mapping_rate = None
        self.total_samples = None

    def update_csv(self):
        row = self.df[(self.df == self.model_name) & (self.df == self.arch_dims) & (self.df == self.arch_interconnections)]
        for key in self.keys:
            if row[key] is None:
                row[key] = getattr(self,key)
        self.df[row] = row
        
    def save_csv(self):
        self.df.write_csv(self.path_to_results)

                

        
