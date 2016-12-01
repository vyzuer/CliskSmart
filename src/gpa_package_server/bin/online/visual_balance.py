import sys, os, string, random, time
from math import sqrt
import Tkinter as tk 
from PIL import ImageTk, Image, ImageDraw
from skimage import color

import numpy as np

import _mypath

from graph.graph import Node, Edge, Graph
import graph.minimize_energy as min_energy

input_image = None
img_height = 240
img_width = 320

center_distance = 10.0    
scaling_factor = 2.0      
circle_radius = 20

#screen properties
c_width = 1000
c_height = 600
border = 20

b_stop = False
PI = 22.0/7
size_scale = 1.0

K1 = 0.01
K2 = 1.0
C = 1.0

# the coolest hue is considered at 225 degree which is equivalent to 0.625 when scaled between 0-1
_COOLEST_HUE = 0.5
# _COOLEST_HUE = 0.625

_CE_WEIGHTS = [3.0,5.0,3.0,2.0,2.0]  # hue, saturation, value, size, contrast


def draw_graph(graph, canvas_view, root, dump_path=None):
    # clear the screen
    for c_item in canvas_view.find_all():
        canvas_view.delete(c_item)
    
    for edge in graph.get_edges():
        # calculate position of this node
        x0 = edge.src.x_pos*c_width
        y0 = edge.src.y_pos*c_height
        x1 = edge.dst.x_pos*c_width
        y1 = edge.dst.y_pos*c_height
        canvas_view.create_line(x0, y0, x1, y1)

    for node in graph.get_nodes():
        # calculate position of this node
        x0 = node.x_pos*c_width
        y0 = node.y_pos*c_height
        print node.x_pos, node.y_pos
        print x0, y0
        # draw this node
        rgb_color = 255*node.color

        circle_radius = node.radius*size_scale
        fill_color = '#%02x%02x%02x' % (rgb_color[0], rgb_color[1], rgb_color[2])

        stipple=''
        # if not node.fix:
        #     stipple='gray75'
        #     canvas_view.create_rectangle(x0-1.5*circle_radius, y0-3.5*circle_radius, x0+1.5*circle_radius, y0+5*circle_radius, outline=fill_color, width=3)

        canvas_view.create_oval(x0-circle_radius, y0-circle_radius, x0 + circle_radius, y0 + circle_radius, fill=fill_color, stipple=stipple)

        # write the id under the node
        # canvas_view.create_text(x0-circle_radius/2, y0+2*circle_radius, anchor=tk.SW, text=str(node.id))
      
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.update()

    if dump_path:
        canvas_view.postscript(file=dump_path, colormode='color')
 
def create_graph(s_objects, p_objects):

    print 'Creating graph ...'

    graph = Graph()

    node_id = 0 # use this as node id as some segments might have similar id

    # add the imp object nodes
    for s_obj in s_objects:
        node = Node(node_id, \
                    seg_id=s_obj.id, \
                    x_pos=s_obj.pos[1], \
                    y_pos=s_obj.pos[0], \
                    b_fix=True, \
                    size=s_obj.size, \
                    color_energy=s_obj.color_energy, \
                    color=s_obj.color, \
                    saliency=s_obj.saliency, \
                    radius=np.sqrt(s_obj.size)/4)

        graph.add_node(node)
        node_id += 1

    max_snode_id = node_id

    # add the people node
    for p_obj in p_objects:
        node = Node(node_id,\
                    seg_id=p_obj.id, \
                    x_pos=p_obj.pos[1], \
                    y_pos=p_obj.pos[0], \
                    b_fix=False, \
                    size=p_obj.size, \
                    color_energy=1.2*p_obj.color_energy, \
                    color=p_obj.color, \
                    saliency=1.2*p_obj.saliency, \
                    radius=np.sqrt(p_obj.size)/4)

        graph.add_node(node)
        # add this node to the list of people nodes
        graph.add_pnode(node)
        graph.num_people += 1

        # create edges from the people node to rest of nodes
        for dst_id in range(max_snode_id):
            src_id = node_id
            src_node = node
            dst_node = graph.get_node(dst_id)
    
            assert src_node is not None
            assert dst_node is not None

            graph.add_edge(src_node, dst_node, spring_length=0.0)
    
        node_id += 1
    
    return graph


def compute_hue_energy(hue):
    warmness = abs(hue - _COOLEST_HUE)

    # to find mimimum circular distance on the hue wheel
    # if warmness > 0.5:
    #     warmness = 1.0 - warmness

    return 2*warmness

def get_color_energy(rgb_color, size):

    hsv_color = color.rgb2hsv(np.reshape(rgb_color, (1,1,3)))[0,0]

    # compute hue energy
    hue_energy = compute_hue_energy(hsv_color[0])

    saturation_energy = hsv_color[1]
    brightness_energy = hsv_color[2]

    # compute relative size of the object
    size_energy = size

    # compute contrast of the salient object
    contrast_energy = 0.5

    energy_ = [hue_energy, saturation_energy, brightness_energy, size_energy, contrast_energy]
    color_energy = np.dot(energy_, _CE_WEIGHTS)/np.sum(_CE_WEIGHTS)

    return color_energy

# create graph from dump
def create_graph_0(dump_path, nodes_file, edges_file):

    print 'Creating graph ...'

    graph = Graph()

    # add the nodes
    nodes_list = dump_path + nodes_file
    nodes_data = np.loadtxt(nodes_list, skiprows=1)
    n_nodes, n_dim = nodes_data.shape
    for i in range(n_nodes):
        node_id = int(nodes_data[i,0])
        if node_id < 0:
            continue
        x_pos = nodes_data[i,1]
        y_pos = nodes_data[i,2]
        size = nodes_data[i,3]
        b_fix = int(nodes_data[i,4])

        rgb_color = nodes_data[i,5:8]
        color_energy = get_color_energy(rgb_color, size)
        print color_energy

        node = Node(node_id,\
                    seg_id=node_id, \
                    x_pos=x_pos, \
                    y_pos=y_pos, \
                    b_fix=b_fix, \
                    size=size, \
                    color_energy=color_energy, \
                    color=rgb_color, \
                    saliency=1.0, \
                    radius=size/2)

        graph.add_node(node)
        # add this node to the list of people nodes
        # graph.add_pnode(node)
        # graph.num_people += 1

    edges_list = dump_path + edges_file

    edges_data = np.loadtxt(edges_list, skiprows=1)
    n_edges, n_dim = edges_data.shape

    for i in range(n_edges):
        src_id = edges_data[i,0]
        if src_id < 0:
            continue
        dst_id = edges_data[i,1]
        length = edges_data[i,2]
        
        src_node = graph.get_node(src_id)
        dst_node = graph.get_node(dst_id)

        assert src_node is not None
        assert dst_node is not None

        graph.add_edge(src_node, dst_node, spring_length=length)

    return graph
    
def stop(root):
    global b_stop
    b_stop = True

def quit(root):
    global b_stop
    b_stop = True
    root.destroy()
    
def prepare_canvas(img_src=None):
    global b_stop, size_scale
    global c_width, c_height
    global input_image

    b_stop = False
    
    root = tk.Tk()

    title = tk.Button(root, bg='sky blue', activebackground='sky blue', activeforeground='navy', fg='navy', font=("TkMenuFont", "14"), width=10, height=1, text="SEG Model for Visual Balance")
    title.pack(side='top', fill='both', padx=4, pady=4)
    
    # create the main sections of the layout, 
    # and lay them out
    top = tk.Frame(root)
    bottom = tk.Frame(root)
    top.pack(side='bottom')
    bottom.pack(side='bottom', fill='both', expand=True)
    
    # create the widgets for the top part of the GUI,
    # and lay them out
    b = tk.Button(root, text="Quit", bg='azure4', width=10, height=2, command=lambda root=root:quit(root))
    c = tk.Button(root, text="Stop", bg='honeydew4', width=10, height=2, command=lambda root=root:stop(root))
    b.pack(in_=top, side='left')
    c.pack(in_=top, side='left')

    w, h = root.winfo_screenwidth(), root.winfo_screenheight()
    root.attributes('-fullscreen', True)

    root.geometry("%dx%d+%d+%d" % (w/2, h, 0, 0))
    
    c_width = w/2 - border*2 - img_width
    c_height = h - border*2 
    # c_height = c_width

    size_scale = np.sqrt(c_width*c_height)
    
    root.title("Group Formation")
    canvas_view = tk.Canvas(root, width=c_width, height=c_height, bg='mint cream')

    if img_src is not None:
        input_image = ImageTk.PhotoImage(Image.open(img_src).resize((img_width, img_height), Image.ANTIALIAS))
        label = tk.Label(image=input_image, height=img_height, width=img_width)
        label.image = input_image # keep a reference!
        label.pack()
    
    canvas_view.pack()
    canvas_view.focus_set()

    return canvas_view, root

def find_mean_pos(graph):

    num_people = len(graph.pnodes)

    positions = np.zeros(shape=(num_people, 2))
    m_pos = np.array([0.0, 0.0])

    for i in range(num_people):
        node = graph.pnodes[i]
        positions[i] = [node.x_pos, node.y_pos]

    m_pos = np.mean(positions, axis=0)

    return m_pos


def find_mpos(graph):

    num_people = len(graph.pnodes)

    positions = np.zeros(shape=(num_people, 2))
    m_pos = np.array([0.0, 0.0])

    for i in range(num_people):
        node = graph.pnodes[i]
        positions[i] = [node.x_pos, node.y_pos]

    m_pos = np.mean(positions, axis=0)

    # move all the nodes to the same y axis level
    positions[:,1] = m_pos[1]

    # sort the nodes according to their x positions
    sn_id = np.argsort(positions[:,0])

    # arrange the nodes next to each other
    for i in range(1,num_people):
        r0 = graph.pnodes[sn_id[i-1]].radius
        r1 = graph.pnodes[sn_id[i]].radius

        # find the x position
        positions[sn_id[i], 0] = positions[sn_id[i-1], 0] + 1.0*r0 + 1.0*r1 

    # move the nodes to the center
    x_dist = m_pos[0] - np.mean(positions[:,0])
    positions[:,0] += x_dist

    return positions


def move_nodes(graph, steps, x_fix=False):

    for i in range(graph.num_people):
        node = graph.pnodes[i]

        if x_fix:
            node.y_pos += steps[i]
        else:
            node.x_pos += steps[i,0]
            node.y_pos += steps[i,1]

def opt_arrangement(graph, m_pos, canvas_view, root, n_iter=100, visualize=False):

    steps = np.zeros(graph.num_people)

    # find x and y steps for each node
    for i in range(graph.num_people):
        node = graph.pnodes[i]
        
        pos0 = node.y_pos
        pos1 = m_pos[1]

        steps[i] = (pos1 - pos0)/n_iter

    temperature = 0.01
    cooling_factor = 0.00005

    for i in range(n_iter):

        move_nodes(graph, steps, x_fix=True)

        min_energy.optimize(graph, K1, K2, C, temperature, y_fix=True)

        temperature = np.abs(temperature - cooling_factor)

        if visualize:
            draw_graph(graph, canvas_view, root)

    return graph

def arrange_people(graph, canvas_view, root, n_iter=100, visualize=False):

    steps = np.zeros(shape=(graph.num_people, 2))

    # find x and y steps for each node
    for i in range(graph.num_people):
        node = graph.pnodes[i]
        
        pos0 = np.asarray([node.x_pos, node.y_pos])
        pos1 = graph.p_positions[i]

        steps[i] = (pos1 - pos0)/n_iter

    for i in range(n_iter):
        move_nodes(graph, steps)

        if visualize:
            draw_graph(graph, canvas_view, root)

    return graph

def find_pos_and_move(graph, canvas_view, root, visualize=False):

    # find the mean position
    m_position = find_mean_pos(graph)

    # animate the movement to the mean position
    graph = opt_arrangement(graph, m_position, canvas_view, root, visualize=visualize)

    graph.p_positions = find_mpos(graph)

    # animate the movement to the mean position
    graph = arrange_people(graph, canvas_view, root, visualize=visualize)

    return graph

def find_pos_and_move_0(graph, canvas_view, root, visualize=False):

    # find the mean position
    graph.p_positions = find_mpos(graph)

    # animate the movement to the mean position
    graph = arrange_people(graph, canvas_view, root, visualize=visualize)

    return graph

def find_ordering(graph, canvas_view, root, visualize=False, max_iter=5000):
    global b_stop

    timestep = 0
    time_spent = 0.0
    temperature = 0.5
    cooling_factor = 0.0005

    while (not b_stop and timestep < max_iter):
        timer = time.time()

        # min_energy.optimize(graph, 1.0, 1.0, 1.0, np.exp(temperature))
        min_energy.optimize(graph, K1, K2, C, temperature)

        # if abs(graph.energy_gain) < 0.0:
        #     b_stop = True

        temperature = np.abs(temperature - cooling_factor)

        time_spent += time.time() - timer
        timestep += 1

        if visualize:
            draw_graph(graph, canvas_view, root)
    
    print 'Iterations:', timestep
    print 'Time:', time_spent

    return graph


def optimize_position(graph, canvas_view, root, visualize=False, max_iter=100):
    global b_stop
    b_stop = False

    # find the centre of energy and update it towards centre of image

    timestep = 0
    time_spent = 0.0
    temperature = 100
    cooling_factor = 0.5

    while (not b_stop and timestep < max_iter):
        timer = time.time()

        b_stop, graph = min_energy.optimize_pos(graph, K1, K2, C, temperature)

        # temperature = np.abs(temperature - cooling_factor)

        time_spent += time.time() - timer
        timestep += 1

        if visualize:
            draw_graph(graph, canvas_view, root)
    
    print 'Iterations:', timestep
    print 'Time:', time_spent

    return graph


def minimize_energy(graph, data_path, visualize=False, graph_dumps=None):

    if not os.path.exists(graph_dumps):
        os.makedirs(graph_dumps)

    img_name = os.path.split(os.path.split(data_path)[0])[1]
    img_ps = graph_dumps + img_name + '.ps'

    # prepare the canvas
    canvas_view = None
    root = None

    if visualize:
        canvas_view, root = prepare_canvas()
        draw_graph(graph, canvas_view, root)

    # optimize energy for ordering
    graph = find_ordering(graph, canvas_view, root, visualize=visualize)

    # Club the people nodes together in a formation
    # later when we start predicted formations we can use that as well
    # graph = find_pos_and_move(graph, canvas_view, root, visualize=visualize)

    # move the formation for balance
    # graph = optimize_position(graph, canvas_view, root, visualize=visualize)

    draw_graph(graph, canvas_view, root, img_ps)

    # time.sleep(1)
    # quit(root)
    if visualize:
        canvas_view.mainloop()

    return graph


if __name__ == "__main__":
    
    if (len(sys.argv) != 5):
        print 'Usage : data_path nodes_file edges_file dump_path'
        sys.exit(0)

    data_path = sys.argv[1]
    nodes_file = sys.argv[2]
    edges_file = sys.argv[3]
    dump_path = sys.argv[4]
    
    graph = create_graph_0(data_path, nodes_file, edges_file)
    
    # set the position of all nodes in the graph randomly to 
    graph.set_random_node_position()
    
    # visualize the enrgy minimization
    minimize_energy(graph, data_path=data_path, visualize=True, graph_dumps=dump_path)

