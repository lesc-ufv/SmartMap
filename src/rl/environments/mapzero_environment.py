from copy import deepcopy
from src.rl.environments.interface_environment import InterfaceEnvironment
from src.rl.states.mapping_state_mapzero import MappingStateMapZero
from src.enums.enum_interconnect_style import EnumInterconnectStyle
from src.graphs.dfgs.dfg_mapzero import DFGMapZero
from src.graphs.cgras.cgra_mapzero import CGRAMapzero
from src.utils.util_calculations import UtilCalculations
from src.utils.util_routing import UtilRouting
from src.utils.util_mapping import UtilMapping
import numpy as np

class MapZeroEnvironment(InterfaceEnvironment):
    @staticmethod
    def step(state:MappingStateMapZero,action):
        bad_reward = state.bad_reward
        if not state.is_end_state:
            id_node_to_be_mapped = state.id_node_to_be_mapped
            cp_dfg: DFGMapZero = deepcopy(state.dfg)
            cp_cgra: CGRAMapzero = deepcopy(state.cgra)

            cp_dfg.assign_scheduled_modulo_time_slice_to_vertex(state.get_current_modulo_time_slice(),id_node_to_be_mapped)

            cp_dfg.assign_PE_to_vertex(action,id_node_to_be_mapped)

            cp_cgra.assign_node_to_pe(id_node_to_be_mapped,action)

            pe = action
            used_pes = cp_cgra.get_used_pes()
            pes_to_routing = cp_cgra.get_pes_to_routing()
            free_interconnections = cp_cgra.get_free_interconnections().copy()
            out_vertices_cgra = cp_cgra.get_out_vertices().copy()
            reward = 0
            if state.get_interconnect_style() == EnumInterconnectStyle.NEIGH_TO_NEIGH:
                if len(cp_dfg.in_vertices[id_node_to_be_mapped]) > 0:
                    for father in cp_dfg.in_vertices[id_node_to_be_mapped]:
                        father_pe = cp_dfg.get_pe_assigned_to_node(father)
                        assert father_pe != -1

                        pes_to_routing,free_interconnections,cost = UtilRouting.route(father_pe,state.cgra.cgra.dim_arch,pe,used_pes ,pes_to_routing,free_interconnections,out_vertices_cgra)
                        
                        if cost != 0:
                            len_routing = len(pes_to_routing[(father_pe,pe)]) - 1
                            father_scheduled_time_slice = cp_dfg.get_schedule_time_slice_by_node_id(father)
                            cur_scheduled_time_slice = max(len_routing+father_scheduled_time_slice, 
                                                        cp_dfg.get_schedule_time_slice_by_node_id(id_node_to_be_mapped))
                            cp_dfg.assign_scheduled_time_slice_to_vertex(cur_scheduled_time_slice,id_node_to_be_mapped)

                        if len(pes_to_routing[(father_pe,pe)]) > 2:
                            for route_pe in pes_to_routing[(father_pe,pe)][1:-1]:
                                cp_cgra.assign_node_to_pe(-2,route_pe)
                                used_pes = cp_cgra.get_used_pes()
                                
                        if reward > -bad_reward:
                            reward -= bad_reward if cost == 0 else UtilCalculations.calc_dist_manhattan(cp_cgra.get_pe_pos_by_pe_id(father_pe),cp_cgra.get_pe_pos_by_pe_id(pe))

                    cp_cgra.update_free_interconnections(free_interconnections)
                    cp_cgra.update_pes_to_routing(pes_to_routing)
                else:
                    cp_dfg.assign_scheduled_time_slice_to_vertex(cp_dfg.get_alap_value_by_node(id_node_to_be_mapped),id_node_to_be_mapped)
            else:
                assert False
            next_state = MappingStateMapZero(cp_dfg,cp_cgra,cp_dfg.get_next_node_to_be_mapped(id_node_to_be_mapped))
            if next_state.is_end_state:
                if next_state.dfg.all_nodes_was_mapped():
                    
                    mapping_is_valid,node_to_scheduled_time_slice = UtilMapping.mapping_is_valid_2(next_state.dfg.base_dfg.vertices,
                                                                                                    next_state.dfg.base_dfg.max_root,
                                                                next_state.dfg.base_dfg.node_to_pe,next_state.dfg.in_vertices,
                                                                next_state.dfg.base_dfg.out_vertices,next_state.cgra.get_pes_to_routing())
          
                    
                    if node_to_scheduled_time_slice is not None:
                        next_state.node_to_scheduled_time_slice = node_to_scheduled_time_slice
                        next_state.dfg.vertices_to_scheduled_time_slice = node_to_scheduled_time_slice
                else:
                    mapping_is_valid = False    
                if not mapping_is_valid:
                    reward -= bad_reward
                    
                next_state.set_mapping_is_valid(mapping_is_valid)

            return next_state, reward/100, next_state.is_end_state
        else:
            return deepcopy(state),0, True