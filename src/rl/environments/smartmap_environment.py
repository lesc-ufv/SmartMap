from copy import deepcopy
from src.rl.environments.interface_environment import InterfaceEnvironment
from src.rl.states.mapping_state_yoto import MappingStateYOTO
from src.enums.enum_interconnect_style import EnumInterconnectStyle
from src.graphs.dfgs.dfg_mapzero import DFGMapZero
from src.graphs.cgras.cgra_mapzero import CGRAMapzero
from src.utils.util_calculations import UtilCalculations
from src.utils.util_routing import UtilRouting
from src.utils.util_mapping import UtilMapping
import numpy as np
import math
class SmartMapEnvironment(InterfaceEnvironment):
    @staticmethod
    def step(state:MappingStateYOTO,action,debug=False):
        bad_reward = state.bad_reward
        if not state.is_end_state:
            id_node_to_be_mapped = state.id_node_to_be_mapped
            cp_dfg  = deepcopy(state.dfg)
            cp_cgra = deepcopy(state.cgra)
            if debug:
                print("Current State")
                state.print_state()
                print()
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
                for neigh in cp_dfg.in_vertices[id_node_to_be_mapped]:

                    neigh_pe = cp_dfg.get_pe_assigned_to_node(neigh)
                    if cp_dfg.base_dfg.node_to_pe[neigh] != cp_dfg.special_value and \
                        (neigh_pe,pe) not in cp_cgra.cgra.pes_to_routing:

                        pes_to_routing,free_interconnections,cost = UtilRouting.route(neigh_pe,state.cgra.cgra.dim_arch,pe,used_pes ,pes_to_routing,free_interconnections,out_vertices_cgra)

                        if len(pes_to_routing[(neigh_pe,pe)]) > 2:
                            for route_pe in pes_to_routing[(neigh_pe,pe)][1:-1]:
                                cp_cgra.assign_node_to_pe(-2,route_pe)
                                used_pes = cp_cgra.get_used_pes()

                        reward -= 0 if cost == 0 else (len(pes_to_routing[(neigh_pe,pe)]) - 1)
                #Reversed routing order
                for neigh in cp_dfg.out_vertices[id_node_to_be_mapped]:
                    neigh_pe = cp_dfg.get_pe_assigned_to_node(neigh)
                    if cp_dfg.base_dfg.node_to_pe[neigh] != cp_dfg.special_value and \
                        (pe,neigh_pe) not in cp_cgra.cgra.pes_to_routing:

                        pes_to_routing,free_interconnections,cost = UtilRouting.route(pe,state.cgra.cgra.dim_arch,neigh_pe,used_pes ,pes_to_routing,free_interconnections,out_vertices_cgra)

                        if len(pes_to_routing[(pe,neigh_pe)]) > 2:
                            for route_pe in pes_to_routing[(pe,neigh_pe)][1:-1]:
                                cp_cgra.assign_node_to_pe(-2,route_pe)
                                used_pes = cp_cgra.get_used_pes()

                        reward -= 0 if cost == 0 else (len(pes_to_routing[(pe,neigh_pe)]) - 1)
                 
                cp_cgra.update_free_interconnections(free_interconnections)
                cp_cgra.update_pes_to_routing(pes_to_routing)
               
            else:
                assert False
            next_state = MappingStateYOTO(cp_dfg,cp_cgra,cp_dfg.get_next_node_to_be_mapped(id_node_to_be_mapped))
            if debug:
                print("Next state")
                next_state.print_state()
            if next_state.is_end_state:
                if next_state.dfg.all_nodes_was_mapped():
                    mapping_is_valid,node_to_scheduled_time_slice = UtilMapping.mapping_is_valid_2(next_state.dfg.base_dfg.vertices,next_state.dfg.base_dfg.max_root
                                                    ,next_state.dfg.get_nodes_to_pe(), next_state.dfg.in_vertices, next_state.dfg.out_vertices,
                                                    next_state.cgra.get_pes_to_routing())
                    next_state.node_to_scheduled_time_slice = node_to_scheduled_time_slice
                else:
                     mapping_is_valid = False
                
                if not mapping_is_valid:
                    for (u,v) in next_state.dfg.base_dfg.edges:
                        father_pe = next_state.dfg.base_dfg.node_to_pe[u]
                        child_pe = next_state.dfg.base_dfg.node_to_pe[v]
                        if (father_pe,child_pe) not in next_state.cgra.cgra.pes_to_routing:
                            reward -= bad_reward
                    reward -= bad_reward

                next_state.set_mapping_is_valid(mapping_is_valid)

            if debug:
                print("Action",action,"Reward",reward)
                print()
            return next_state,reward/100, next_state.is_end_state
        else:
            if debug:
                print("End state")
                state.print_state()
            return deepcopy(state),0, True