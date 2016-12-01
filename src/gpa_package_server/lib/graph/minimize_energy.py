import sys, os, string, random, time
from math import sqrt
import numpy as np
import copy

max_velocity = 0.005
friction = 0.000005

force_type = 'h' # c - color_energy,s - saliency ,h - hybrid

import _mypath

import common_gpa.global_variables as gv

_server = gv.__SERVER

min_bound, max_bound = 1./6, 5./6

def edge_dist(x,y):
    return sqrt(x*x + y*y)

def spring_force(node_1, node_2, K2):
    x_dist = node_1.x_pos - node_2.x_pos
    y_dist = node_1.y_pos - node_2.y_pos

    dist = edge_dist(x_dist, y_dist)
    a1 = np.abs(node_1.color_energy - node_2.color_energy)
    a2 = np.abs(node_1.saliency - node_2.saliency)

    a = 1.0

    K = K2

    a = a1 + a2
    if force_type == 'c':
        a = a1
        K = K2/2
    elif force_type == 's':
        a = a2
        K = K2/2
    
    # a = np.exp(a)

    s_force = [0.0, 0.0]
    if dist != 0:
        s_force[0] = 1.0*x_dist*dist*a/K
        s_force[1] = 1.0*y_dist*dist*a/K
    else:
        s_force[0] = random.random()
        s_force[1] = random.random()

    return s_force

# only the magnitude of the force is calculated here
# add the negative sign later
def electrical_force(node_1, node_2, K1):
    x_dist = node_1.x_pos - node_2.x_pos
    y_dist = node_1.y_pos - node_2.y_pos

    squared_dist = x_dist*x_dist + y_dist*y_dist
    e1 = node_1.color_energy*node_2.color_energy
    e2 = node_1.saliency*node_2.saliency

    e = e1+e2
    K = K1
    if force_type == 'c':
        e = e1
        K = 2*K1
    elif force_type == 's':
        e = e2
        K = 2*K1

    # e = np.exp(e)

    e_force = [0.0, 0.0]
    if squared_dist != 0:
        # e_force[0] = 1.0*x_dist*(K1*K1+e1*e2)/squared_dist
        # e_force[1] = 1.0*y_dist*(K1*K1+e1*e2)/squared_dist
        e_force[0] = K*x_dist*e/squared_dist
        e_force[1] = K*y_dist*e/squared_dist
    else:
        e_force[0] = random.random()
        e_force[1] = random.random()

    return e_force


def compute_energy(graph, C):
    energy = 0.0

    for node in graph.nodes:
        node.disp[0] = -C*node.e_force[0] + node.s_force[0]
        node.disp[1] = -C*node.e_force[1] + node.s_force[1]

        node.energy = node.disp[0]*node.disp[0] + node.disp[1]*node.disp[1]
        energy += node.energy

    # store the energy gain
    graph.energy_gain = graph.energy - energy
    graph.energy = energy

    # print graph.energy, graph.energy_gain

    return graph

def compute_eforce(graph, K1):

    for node in graph.nodes:
        node.e_force = [0,0]
        for n_node in graph.nodes:
            if (node.id != n_node.id):
                e_force = electrical_force(n_node, node, K1)
                node.e_force[0] += e_force[0]
                node.e_force[1] += e_force[1]

    return graph


def compute_sforce(graph, K2):
    for node in graph.nodes:
        node.s_force = [0,0]
        for n_node in node.neighbours:
            # compute spring force
            s_force = spring_force(n_node, node, K2)

            node.s_force[0] += s_force[0]
            node.s_force[1] += s_force[1]

    return graph


def check_bounds(obj, x_dir, y_dir):
    b_stop = False

    if (obj.x_pos < min_bound and not x_dir) or (obj.x_pos > max_bound and x_dir) or (obj.y_pos < 2*min_bound and not y_dir) or (obj.y_pos > max_bound and y_dir):
        b_stop = True

    return b_stop

def update_positions(graph, temperature, b_stop=False, y_fix=False):

    """
        first iterate through all the psuedo nodes and compute 
        disp vector for each of the segment nodes
        and then update the position of graph nodes
    """
    p_disp = np.zeros(shape=(graph.num_people,2))
    p_move = np.zeros(shape=(graph.num_people,2))
    for i in range(graph.num_pnodes):
        node = graph.pnodes[i]
        gid = node.gid
        p_disp[gid,:] += node.disp

    for i in range(graph.num_pnodes):
        node = graph.pnodes[i]
        gid = node.gid
        node.disp = p_disp[gid,:]
        node.energy = node.disp[0]*node.disp[0] + node.disp[1]*node.disp[1]

    for node in graph.nodes:
        b_original = True
        if not node.fix:
            disp = sqrt(node.energy)
            gid = node.gid

            # print "optimizing"
            # node.x_pos += (node.disp[0]/disp)*np.min([temperature, disp])
            # node.y_pos += (node.disp[1]/disp)*np.min([temperature, disp])

            if b_original:
                node.move[0] = (node.disp[0]/disp)*np.min([temperature, disp])
                node.move[1] = (node.disp[1]/disp)*np.min([temperature, disp])
            else:
                node.move[0] = node.disp[0]*temperature
                node.move[1] = node.disp[1]*temperature

            # ensure maximum move
            if (node.move[0] > max_velocity):
                node.move[0] = max_velocity
            if (node.move[1] > max_velocity):
                node.move[1] = max_velocity
            if (node.move[0] < -1*max_velocity):
                node.move[0] = -1*max_velocity
            if (node.move[1] < -1*max_velocity):
                node.move[1] = -1*max_velocity
            # get friction into play
            if node.move[0] > friction:
                node.move[0] -= friction
            if node.move[0] < -1*friction:
                node.move[0] += friction
            if node.move[1] > friction:
                node.move[1] -= friction
            if node.move[1] < -1*friction:
                node.move[1] += friction

            if y_fix:
                node.x_pos += node.move[0]
                p_move[gid,0] = node.move[0]
                p_move[gid,1] = 0.0
            else:
                node.x_pos += node.move[0]
                node.y_pos += node.move[1]
                p_move[gid,0] = node.move[0]
                p_move[gid,1] = node.move[1]
        
    # update the position of psuedo nodes
    for i in range(graph.num_people):
        obj = graph.psuedo_nodes[i]
        obj.x_pos += p_move[i,0]
        obj.y_pos += p_move[i,1]

        if not b_stop:
            x_dir = p_move[i,0] > 0
            y_dir = p_move[i,1] > 0
            b_stop = check_bounds(obj, x_dir, y_dir)

    return graph, b_stop

# make one step for energy minimization
def optimize(graph, K1, K2, C, temperature=0.06, y_fix=False):

    # calculate the repulsive force for each node
    graph = compute_eforce(graph, K1)

    # calculate the attractive force for each edge
    graph = compute_sforce(graph, K2)

    graph = compute_energy(graph, C)
    
    # update the positions
    graph, b_stop = update_positions(graph, temperature, y_fix=y_fix)
    
    return graph, b_stop

def optimize_positions(graph):

    b_stop = False

    disp = np.array([0.0, 0.0])
    for i in range(graph.num_pnodes):
        node = graph.pnodes[i]
        disp += node.disp

    if edge_dist(disp[0], disp[1]) < 0.01:
        b_stop = True

    # update displacement of people nodes
    for i in range(graph.num_pnodes):
        node = graph.pnodes[i]
        node.disp = disp
        node.energy = node.disp[0]*node.disp[0] + node.disp[1]*node.disp[1]

    return b_stop

def optimize_positions_0(graph):
    b_stop = False

    num_nodes = len(graph.nodes)

    positions = np.zeros(shape=(num_nodes, 2))
    forces = np.zeros(shape=(num_nodes, 2))
    saliency = np.zeros(shape=(num_nodes, 2))
    m_pos = np.array([0.0, 0.0])
    c_pos = np.array([0.5, 0.5])

    for i in range(num_nodes):

        node = graph.nodes[i]
        
        positions[i] = [node.x_pos, node.y_pos]
        forces[i] = [node.disp[0], node.disp[1]]
        node_s = node.saliency
        saliency[i] = [node_s, node_s]
    
    # find the center of energy
    # np.set_printoptions(formatter={'float': lambda x: "{0:0.6f}".format(x)})
    forces = forces/abs(forces).max()
    saliency = saliency/saliency.max()
    # print forces
    # print saliency
    fs = forces*forces
    m_pos[0] = np.dot(fs[:,0].transpose(), positions[:,0])/np.sum(fs[:,0])
    m_pos[1] = np.dot(fs[:,1].transpose(), positions[:,1])/np.sum(fs[:,1])

    disp = c_pos - m_pos

    if edge_dist(disp[0], disp[1]) < 0.001:
        b_stop = True

    # update displacement of people nodes
    for i in range(graph.num_pnodes):
        node = graph.pnodes[i]
        node.disp = disp
        node.energy = node.disp[0]*node.disp[0] + node.disp[1]*node.disp[1]

    return b_stop


def comp_energy(graph, K1, K2, C):

    # calculate the repulsive force for each node
    compute_eforce(graph, K1)

    # calculate the attractive force for each edge
    compute_sforce(graph, K2)

    compute_energy(graph, C)

    return graph.energy_gain

def update_radius(graph):
    for i in range(graph.num_pnodes):
        node = graph.pnodes[i]
        node.radius=np.sqrt(node.size)/4

    
def optimize_size(graph, K1, K2, C):

    # if size is too big reduce it
    t_size = 0.0
    
    for i in range(graph.num_pnodes):
        node = graph.pnodes[i]
        t_size += 5.0*node.size

    s_step = 0.0002
    c_step = 0.005

    if t_size > 0.30 or graph.pnodes[0].size > 0.125:
        for i in range(graph.num_pnodes):
            node = graph.pnodes[i]
            node.size -= s_step
            node.color_energy -= c_step
            node.radius=np.sqrt(node.size)/4

    elif t_size < 0.05 or graph.pnodes[0].size < 0.0078:
        for i in range(graph.num_pnodes):
            node = graph.pnodes[i]
            node.size += s_step
            node.color_energy += c_step
            node.radius=np.sqrt(node.size)/4

    return graph

def optimize_pos(graph, K1, K2, C, temperature=0.06):

    # calculate the repulsive force for each node
    compute_eforce(graph, K1)

    # calculate the attractive force for each edge
    compute_sforce(graph, K2)

    compute_energy(graph, C)
    
    # update the force on movable nodes
    b_stop = optimize_positions(graph)

    # increase or decrease size based on energy gain
    # graph = optimize_size(graph, K1, K2, C)
    
    # update the positions
    graph, b_stop = update_positions(graph, temperature, b_stop=b_stop)
    

    return b_stop, graph

