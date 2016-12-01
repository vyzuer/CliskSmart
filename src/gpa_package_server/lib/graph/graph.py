import random, time

# Class for Nodes
class Node:
    def __init__(self, node_id, seg_id=0, x_pos=0, y_pos=0, b_fix=False, size = 1.0, color_energy=0.0, color=[0.5,0.5,0.5], neighbours=None, radius=0.01, saliency=0.0, gid=-1):
        self.id = node_id        
        self.seg_id = seg_id
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.s_force = [0, 0]
        self.e_force = [0, 0]
        self.disp = [0, 0]
        self.fix = b_fix
        self.size = size
        self.color_energy = color_energy
        self.color = color
        self.radius = radius
        self.saliency= saliency
        self.gid = gid

        if neighbours is None:
            neighbours = []
        self.neighbours = neighbours
        self.energy = 0.0
        self.move = [0, 0]
        
    def get_neighbours(self):
        return self.neighbours

    def get_id(self):
        return self.id

    def set_neighbour(self, node):
        self.neighbours.append(node)

    def delete_neighbour(self, node):
        self.neighbours.remove(node)


# Class for edge
class Edge:
    def __init__(self, node_1, node_2, spring_length):
        self.src = node_1
        self.dst = node_2

        self.spring_length = spring_length

    def get_src(self):
        return self.src

    def get_dst(self):
        return self.dst

    def get_spring_length(self):
        return self.spring_length

# Class for graph
class Graph:
    def __init__(self):
        # build an empty graph
        self.nodes = []
        self.edges = []
        self.energy = 0.0
        self.energy_gain = 0.0
        self.pnodes = []
        self.psuedo_nodes = []
        self.num_people = 0
        self.num_pnodes = 0
        self.positions = None
        self.p_positions = None
    
    def add_psuedo_node(self, node):
        # adds a node to the graph
        self.psuedo_nodes.append(node)
    
    def add_pnode(self, node):
        # adds a node to the graph
        self.pnodes.append(node)
    
    def add_node(self, node):
        # adds a node to the graph
        self.nodes.append(node)
    
    def add_edge(self, node_1, node_2, spring_length=0.0):
        if node_1.id < node_2.id:
            self.edges.append(Edge(node_1, node_2, spring_length))
        else:
            self.edges.append(Edge(node_2, node_1, spring_length))

        node_1.set_neighbour(node_2)
        node_2.set_neighbour(node_1)
    
    def nodes_list(self):
        # returns the list of ids of nodes
        list_of_ids = []
        for node in self.nodes:
            list_of_ids.append(node.id)
        return list_of_ids
    
    def get_node(self, node_id):
        for node in self.nodes:
            if node.id == node_id:
                return node

        return None

    def get_nodes(self):
        return self.nodes

    def get_edges(self):
        # returns the list of edges ([(id, id), (id, id), ...]
        return self.edges
    
    def count_nodes(self):
        # prints the number of nodes
        return len(self.nodes)
    
    def count_edges(self):
        # prints the number of nodes
        return len(self.edges)
    
    def print_nodes(self):
        # prints the list of nodes
        for x in self.nodes:
            print 'Node', x.id
            for n_node in x.neighbours:
                print '\tEdge', n_node.id

    
    def print_edges(self):
        # prints the list of edges
        to_print = '['
        count = 0
        for edge in self.edges:
            to_print = to_print + '(' + str(edge.src.id) + ',' + str(edge.dst.id) + '), '
            count += 1
            if count > 200:
                print to_print, 
                to_print = ''
                count = 1
        if count > 0: to_print = to_print[:-2]
        to_print = to_print + ']'
        print to_print
    
    def print_data(self):
        # prints number of nodes and edges
        print 'graph with', len(self.nodes), 'nodes and', len(self.edges), 'edges\n'

        for node in self.nodes:
            print 'x coordinate of', node.id, 'is', node.x_pos
            print 'y coordinate of', node.id, 'is', node.y_pos
            print 'repulsive force', node.e_force
            print 'attractive force', node.s_force
            print
    
    def set_random_node_position(self, center_distance=1.0):
        # sets random positions for all nodes
        for node in self.nodes:
            if not node.fix:
                node.x_pos = random.random() * center_distance
                node.y_pos = random.random() * center_distance
    
