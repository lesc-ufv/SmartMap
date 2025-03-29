from src.utils.util_dfg import UtilDFG

class UtilMapping:
    @staticmethod
    def mapping_is_valid(vertices,node_to_sched_time_slice,node_to_in_vertices,node_to_pe,pes_to_routing):
        for node in vertices: 
            cur_time_slice = node_to_sched_time_slice[node]
            child_pe = node_to_pe[node]
            for father in node_to_in_vertices[node]:
                father_pe = node_to_pe[father]
                if node_to_sched_time_slice[father] + (len(pes_to_routing[(father_pe,child_pe)]) - 1) != cur_time_slice:
                    return False
        return True
    @staticmethod
    def mapping_is_valid_2(vertices,edges_or_max_root,node_to_pe,node_to_in_vertices,node_to_out_vertices,pes_to_routing):
        for v in pes_to_routing.values():
            if v == []:
                return False,None

        def schedule(node,in_vertices,cur_schedule,node_to_scheduled_time_slice,fifo):
            node_to_scheduled_time_slice[node] = max(0,cur_schedule)
            child_pe = node_to_pe[node]
            visited[node] = True
            fifo.append(node)
            for father in in_vertices[node]:
                if father not in node_to_scheduled_time_slice:
                    father_pe = node_to_pe[father]
                    schedule(father,in_vertices,cur_schedule -(len(pes_to_routing[(father_pe,child_pe)])-1),node_to_scheduled_time_slice,fifo)



        if isinstance(edges_or_max_root,list):
            root = UtilDFG.get_max_root(edges_or_max_root)
        elif isinstance(edges_or_max_root,str) or isinstance(edges_or_max_root,int):
            root = edges_or_max_root
        node_to_scheduled_time_slice = {root: 0}
        visited = {root:True}
        fifo = [root]
        while fifo:
            cur_node = fifo.pop(0)
            for out_node in node_to_out_vertices[cur_node]:
                child_pe = node_to_pe[out_node]

                if out_node not in visited:
                    father_pe = node_to_pe[cur_node]
                    node_to_scheduled_time_slice[out_node] = (len(pes_to_routing[(father_pe,child_pe)]) - 1) + node_to_scheduled_time_slice[cur_node]
                    for father in node_to_in_vertices[out_node]:
                        father_pe = node_to_pe[father]
                        if father not in node_to_scheduled_time_slice:
                            schedule(father,node_to_in_vertices,node_to_scheduled_time_slice[out_node]  - (len(pes_to_routing[(father_pe,child_pe)])-1),node_to_scheduled_time_slice,fifo)
                    visited[out_node] = True
                    fifo.append(out_node)
                else:
                    if cur_node != out_node:
                        father_pe = node_to_pe[cur_node]
                        node_to_scheduled_time_slice[out_node] =  max(node_to_scheduled_time_slice[out_node],node_to_scheduled_time_slice[cur_node] + (len(pes_to_routing[(father_pe,child_pe)])-1))
        assert len(vertices) == len(node_to_scheduled_time_slice)
        return UtilMapping.mapping_is_valid(vertices,node_to_scheduled_time_slice,node_to_in_vertices,node_to_pe,pes_to_routing),node_to_scheduled_time_slice
    
    @staticmethod
    def adjust_schedule(node,in_vertices,cur_schedule,node_to_pe,pes_to_routing,node_to_scheduled_time_slice):
        node_to_scheduled_time_slice[node] = max(0,cur_schedule)
        child_pe = node_to_pe[node]
        for father in in_vertices[node]:
            father_pe = node_to_pe[father]
            node_to_scheduled_time_slice = UtilMapping.adjust_schedule(father,in_vertices, cur_schedule -(len(pes_to_routing[(father_pe,child_pe)])-1),
                                        node_to_pe,pes_to_routing,node_to_scheduled_time_slice)
        return node_to_scheduled_time_slice
    @staticmethod
    def get_unsuccesful_reason(final_state):
        dfg = final_state.dfg

        if final_state.mapping_is_valid:
            return "None"

        if not dfg.all_nodes_was_mapped():
            return "At least one node was not mapped."

        cgra = final_state.cgra

        for v in cgra.cgra.pes_to_routing.values():
            if v == []:
                return "Invalid Routing."

        return "Invalid timing."

    
