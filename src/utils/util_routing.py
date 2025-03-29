from src.utils.util_calculations import UtilCalculations
import heapq
class UtilRouting:   
    @staticmethod
    def route(father_pe,arch_dims, child_pe,used_pes,pes_to_routing, free_interconnections,out_vertices):
        path = UtilRouting.a_star_routing(father_pe, arch_dims,child_pe, used_pes, out_vertices,free_interconnections)
        pes_to_routing[(father_pe,child_pe)] = path
        for i in range(len(path) - 1):
            free_interconnections[path[i]].remove(path[i + 1])
        cost = len(pes_to_routing[(father_pe,child_pe)])
        return pes_to_routing,free_interconnections,cost
    
    @staticmethod
    def a_star_routing(init_state, arch_dims, final_state, used_pes, out_vertexes: dict, free_interconnections):
        def pe_pos(pe_id, row, col):
            return (pe_id // col, pe_id % col)
        
        calc_dist_fn = UtilCalculations.calc_dist_manhattan 

        heap = []
        heapq.heappush(heap, (0, 0, init_state)) 

        visited = set(used_pes)
        if final_state in visited:
            visited.remove(final_state)

        father = {}

        g_cost = {init_state: 0}

        while heap:
            _, g, curr_node = heapq.heappop(heap)

            if curr_node == final_state:
                path = []
                aux = final_state
                while aux != init_state:
                    path.append(aux)
                    aux = father[aux]
                path.append(init_state)
                return path[::-1]

            for neighbor in free_interconnections.get(curr_node, []):
                if neighbor not in visited:
                    cost_g = g + 1 
                    if neighbor not in g_cost or cost_g < g_cost[neighbor]:
                        g_cost[neighbor] = cost_g
                        f = cost_g + calc_dist_fn(
                            pe_pos(neighbor, arch_dims[0], arch_dims[1]),
                            pe_pos(final_state, arch_dims[0], arch_dims[1])
                        )
                        heapq.heappush(heap, (f, cost_g, neighbor))
                        father[neighbor] = curr_node
                        visited.add(neighbor)

        return []

    

    
    @staticmethod
    def bfs_routing(init_state, final_state,used_pes, out_vertexes: dict,free_interconnections):
        fifo = []
        fifo.append(init_state)

        visited = {init_state:True}
        
        for pe in used_pes:
            if pe != final_state:
                visited[pe] = True
        father = {}

        while len(fifo) > 0:
            curr_node = fifo.pop(0)

            if curr_node == final_state:
                aux = final_state
                path = []
                while aux != init_state:
                    path.append(aux)
                    aux = father[aux]
                path.append(aux)
                path.reverse()
                return path

            for neighboor in out_vertexes:
                if neighboor not in visited and neighboor in free_interconnections[curr_node]:
                    fifo.append(neighboor)
                    visited[neighboor] = True
                    father[neighboor] = curr_node
        return []

    

