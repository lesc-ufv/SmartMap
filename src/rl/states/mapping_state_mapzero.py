from src.rl.states.mapping_state_interface import MappingStateInterface
import torch
from src.graphs.dfgs.dfg_mapzero import DFGMapZero
from src.graphs.cgras.cgra_mapzero import CGRAMapzero
class MappingStateMapZero(MappingStateInterface):
    def __init__(self,dfg: DFGMapZero, cgra, id_node_to_be_mapped):
        self.dfg = dfg
        self.cgra = cgra
        self.id_node_to_be_mapped = id_node_to_be_mapped
        self.is_end_state = id_node_to_be_mapped is None or cgra.cgra.all_pes_was_used()
        self.legal_actions = None
        self.mapping_is_valid = False
        self.node_to_scheduled_time_slice = None
        self.bad_reward = 100
    
    def is_bad_reward(self,reward):
        return -reward*100 >= self.bad_reward
    def get_mapping_is_valid(self):
        return self.mapping_is_valid

    def set_mapping_is_valid(self,mapping_is_valid):
        self.mapping_is_valid = mapping_is_valid
    
    def get_action_to_index_policy_logits(self):
        return {a:a for a in self.legal_actions}

    def get_dfg_edges_index(self):
        return self.dfg.get_edges_index()
    def get_current_modulo_time_slice(self):
        return 0
    def get_cgra_edges_index(self):
        return self.cgra.get_edges_index()
    
    def get_dfg_features(self):
        return self.dfg.get_node_features('list')
    
    def get_cgra_features(self):
        return self.cgra.get_node_features('list')
    
    def get_feature_node_to_be_mapped(self):
        return list(self.dfg.get_feature_by_node_id(self.id_node_to_be_mapped).values())
    
    def generate_mask(self):
        return self.cgra.generate_mask()

    def get_legal_actions(self):
        if self.legal_actions:
            return self.legal_actions
        self.legal_actions = self.cgra.generate_legal_actions()
        return self.legal_actions

    def get_is_end_state(self):
        return self.is_end_state
    
    def generate_model_args(self):
        if not self.is_end_state:
             
            return [self.get_dfg_features(), 
                self.get_dfg_edges_index(),
               self.get_cgra_features(), 
                self.get_cgra_edges_index(),
                self.get_feature_node_to_be_mapped(),
                self.generate_mask()
                ]
        dfg_features = self.get_dfg_features()
        return [dfg_features, 
                self.get_dfg_edges_index(),
                self.get_cgra_features(), 
                self.get_cgra_edges_index(),
                [-1 for _ in range(len(dfg_features[0]))],
                self.generate_mask()
                ]
        

    def get_interconnect_style(self):
        return self.cgra.get_interconnect_style()
    
    def print_state(self):
        def print_placement(state):
            n_rows,n_cols = state.cgra.cgra.dim_arch
            matrix = [[-1 for j in range(n_cols)] for i in range(n_rows)]
            map_nodes = state.dfg.base_dfg.reseted_to_real_node
            for k,v in state.cgra.get_pes_to_node_id().items():
                i_pos = k // n_rows
                j_pos = k % n_cols
                matrix[i_pos][j_pos] = map_nodes[v] if v != -1 and v != -2 else "R" if v == -2 else v
            for row in matrix:
                print(row)
        def print_dict(dic):
            for k,v in dic.items():
                print(k,v) 
        print("Node to be mapped", self.dfg.base_dfg.reseted_to_real_node[self.id_node_to_be_mapped] if self.id_node_to_be_mapped is not None else None)
        print("Placement:")
        print_placement(self)
        print("Roteamento")
        print_dict(self.cgra.cgra.pes_to_routing)
        print("Legal actions", self.legal_actions)
        print("Features DFG nodes")
        print_dict(self.dfg.get_node_features('dict'))
        print("CGRA features")
        print_dict(self.cgra.get_node_features('dict'))
        print("End state: ",self.is_end_state)
        print()
        print('Scheduled Time Slice:')
        if self.is_end_state:
            sched_time = self.dfg.get_node_features('dict')
            for k,v in sched_time.items():
                map_nodes = self.dfg.base_dfg.reseted_to_real_node
                print(map_nodes[k],v['sched_time'])


            
    
 
