import pandas as pd
import os
class TrainingResults:
    keys = ["model_name", "arch_dims", "arch_interconnections", "training_time"]

    def __init__(self,model_name,arch_dims,arch_interconnections):
        self.path_to_results = 'results/train_results.csv'
        self.model_name = model_name
        self.arch_dims = str(arch_dims)
        self.arch_interconnections = arch_interconnections
        if os.path.exists(self.path_to_results):
            self.df = pd.read_csv(self.path_to_results)
        else:
            dic = {k:[] for k in self.keys}
            self.df = pd.DataFrame(dic)
        self.training_time = None
    

    def update_csv(self):
        row = self.df[(self.df['model_name'] == self.model_name) & (self.df['arch_dims'] == self.arch_dims) & (self.df['arch_interconnections'] == self.arch_interconnections)]
        if row.empty:
            self.df.loc[len(self.df)] = [self.model_name,self.arch_dims,self.arch_interconnections] + [None]*1
            row = self.df[(self.df['model_name'] == self.model_name) & (self.df['arch_dims'] == self.arch_dims) & (self.df['arch_interconnections'] == self.arch_interconnections)]
        for key in self.keys:
            if row[key] is None:
                row[key] = getattr(self,key)

        for key in self.keys:
            # if row.iloc[0][key] is None:
            value = getattr(self, key)
            if value is not None:
                self.df.loc[row.index[0], key] = value
                
    def save_csv(self):
        self.update_csv()
        self.df.to_csv(self.path_to_results,index=False)

                

        
