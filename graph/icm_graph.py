import networkx as nx
import matplotlib.pyplot as plt
from qutip.qip.icm import Icm

def get_icm_attrs(gate_count=None, target=None, cleft=None, cright=None,
                 rough=None, smooth=None, color=None, pos=None, bit=None,
                 X=None, Z=None, injector=None, injector_type=None, pin=None, contracted_pins=None):

        icm_dict = {}        
        icm_dict["bit"]=bit
        icm_dict["gate_count"]=gate_count
        icm_dict["smooth"]=smooth
        icm_dict["rough"]=rough
        icm_dict["color"]=color
        icm_dict["pos"]=pos
        icm_dict["X"]=X
        icm_dict["Z"]=Z
        icm_dict["injector"]=injector
        icm_dict["injector_type"]=injector_type
        icm_dict["pin"]=pin
        icm_dict["contracted_pins"]=contracted_pins
        return icm_dict

def _get_bit_blocks(icm_rep):    
    cmat = [[] for i in range(icm_rep.N)]
    gate_count = 0
    for gate in icm_rep.gates:
        if gate.name != "CNOT":
            target_bit = gate.targets[0]
            cmat[target_bit] += [gate.arg_label]
        if gate.name == "CNOT":
            cmat[gate.controls[0]] += [(gate_count, gate.controls[0] ,"cleft")]
            cmat[gate.controls[0]] += [(gate_count, gate.controls[0] ,"cright")]
            cmat[gate.targets[0]] += [(gate_count, target_bit , "target")]
            gate_count +=1
    return cmat

def construct_icm_graph(qcircuit):
    graphs = []
    gate_count = 0
    bit_blocks = _get_bit_blocks(qcircuit)
    for gate in qcircuit.gates:
        if gate.name == "CNOT":
            g = nx.Graph()

            control_bit = gate.controls[0]
            target_bit = gate.targets[0]
            
            rough = (gate_count, str(control_bit)+str(target_bit), "rough")
            cleft = (gate_count, control_bit, "cleft")
            cright = (gate_count, control_bit, "cright")
            target = (gate_count, target_bit, "target")
            
            pin_cleft = None
            pin_cright = None
            pin_target = None
            
            try:
                if bit_blocks[control_bit][bit_blocks[control_bit].index(cleft) - 1]== "input":
                    pin_cleft = ["cap"]
                if bit_blocks[control_bit][bit_blocks[control_bit].index(cleft) - 1] == "ancilla":
                    pin_cleft = ["injector"]

                if bit_blocks[control_bit][bit_blocks[control_bit].index(cright) + 1 ]== "output":
                    pin_cright = ["cap"]
                if bit_blocks[control_bit][bit_blocks[control_bit].index(cright) + 1]== "measurement":
                    pin_cright = ["cap"]
                if bit_blocks[control_bit][bit_blocks[control_bit].index(cright) + 1] == "correction":
                    pin_cright = ["cap"]

                if bit_blocks[target_bit][bit_blocks[target_bit].index(target) - 1]== "input":
                    pin_target = ["cap"]
                if bit_blocks[target_bit][bit_blocks[target_bit].index(target) - 1] == "ancilla":
                    pin_target = ["injector"]

                if bit_blocks[target_bit][bit_blocks[target_bit].index(target) + 1 ]== "output":
                    pin_target = ["cap"]
                if bit_blocks[target_bit][bit_blocks[target_bit].index(target) + 1]== "measurement":
                    pin_target = ["cap"]
                if bit_blocks[target_bit][bit_blocks[target_bit].index(target) + 1] == "correction":
                    pin_target = ["cap"]

            except ValueError:
                print(gate.name, gate_count)

            g.add_node(cleft, get_icm_attrs(gate_count = gate_count, color="r", pos = (gate_count - 0.50, - control_bit), pin = pin_cleft))
            g.add_node(cright, get_icm_attrs(gate_count = gate_count, color="r", pos = (gate_count + 0.50, - control_bit), pin = pin_cright))
            g.add_node(target, get_icm_attrs(gate_count = gate_count, color="r", pos = (gate_count, - target_bit), pin=pin_target))
            
            g.add_node(rough, get_icm_attrs(color="b", pos = (gate_count, - (target_bit + control_bit)/2.)))

            g.add_edge(cleft, rough)
            g.add_edge(cright, rough)
            g.add_edge(target, rough)

            graphs += [g]
            
            gate_count += 1
    g = nx.compose_all(graphs)
    return g

def draw_graph(g):
    plt.figure(1,figsize=(12,6)) 
    node_color=[g.node[key]["color"] for key in g.nodes()]
    node_labels = {}
    for key in g.nodes():        
        node_labels[key] = (g.node[key]["pin"], key)
    pos=nx.get_node_attributes(g,'pos')
    nx.draw_networkx(g, node_color=node_color, pos = pos, labels=node_labels)
    plt.show()

def _get_mergers(circuit):
    cmat = [[] for i in range(circuit.N)]
    idx = 0
    for gate in circuit.gates:
        if gate.name == "CNOT":
            target_bit = gate.targets[0]
            control_bit = gate.controls[0]

            cmat[control_bit] += [(idx, control_bit, "cleft")]
            cmat[control_bit] += [(idx, control_bit, "cright")]
            cmat[target_bit] += [(idx, target_bit, "target")]
            idx += 1
            
    t_mergers = {}
    c_mergers = {}
    
    for row_id, row in enumerate(cmat):
        for column_id, column in enumerate(row):
            if column[2] == "target":
                i = column_id
                temp_ = []
                while i+1 < len(row) and row[i+1][2] != "cright":
                    temp_ += [row[i+1]]
                    i +=1
                if temp_:
                    t_mergers[column] = temp_
            elif column[2] == "cright" and (column_id + 1) < len(row):
                c_mergers[column] = row[column_id + 1]
    return t_mergers, c_mergers

def combine_graph(g, circuit):
    t_mergers, c_mergers = _get_mergers(circuit)
    for key in t_mergers:
        for item in t_mergers[key]:
            temp_pins = []
            if g.node[key]["pin"] is not None:
                [temp_pins.append(x) for x in g.node[key]["pin"]]
            if g.node[item]["pin"] is not None:
                [temp_pins.append(x) for x in g.node[item]["pin"]]
                
            if len(temp_pins) == 0:
                temp_pins = None
            g = nx.contracted_nodes(g, key, item, self_loops=False)
            nx.set_node_attributes(g, "pin", {key: temp_pins})
            
    for key in c_mergers:
        temp_pins = []
        if g.node[key]["pin"] is not None:
            [temp_pins.append(x) for x in g.node[key]["pin"]]

        if g.node[c_mergers[key]]["pin"] is not None:
            [temp_pins.append(x) for x in g.node[c_mergers[key]]["pin"]]
        
        if len(temp_pins) == 0:
            temp_pins = None
            
        g = nx.contracted_nodes(g, key, c_mergers[key], self_loops=False)
        nx.set_node_attributes(g, "pin", {key: temp_pins})

    return g

def teleport(g, count=1):
    teleport_list = []
    pin_list = {}
    color_list = {}
    tcount = 0
    for node in g:
        neighbors = g.neighbors(node)
        if len(neighbors) == 1:
            temp_pin = []
            temp_pin1 = g.node[node]["pin"]
            temp_pin2 = g.node[neighbors[0]]["pin"]
            if temp_pin1 is not None:
                [temp_pin.append(x) for x in temp_pin1]
            if temp_pin2 is not None:
                [temp_pin.append(x) for x in temp_pin2]
                
            if len(temp_pin) == 0:
                temp_pins = None

            c = g.node[node]["color"]
            if tcount < count:
                teleport_list += [(neighbors[0], node)]
                g = nx.contracted_nodes(g, neighbors[0], node, self_loops=False)

        
                nx.set_node_attributes(g, "pin", {neighbors[0]: temp_pin})
                # nx.set_node_attributes(g, "color", {neighbors[0]: c})
                tcount += 1

    return g


def three_loop_reduction(g, opt1=0, opt2=0):
    removeables = []
    optional = {}
    for node in g.node:
        p = g.node[node]["pin"]
        if p is None:
            neighbors = g.neighbors(node)
            if len(neighbors) == 3:
                removeables += [node]
                temp_optional = []
                for temp_node in neighbors:
                    if g.node[temp_node]["pin"] is None:
                        temp_optional += [temp_node]
                if len(temp_optional) > 0:
                    optional[node] = temp_optional

    if len(removeables) > 0:
        print("removeables", removeables)
        g.remove_node(removeables[opt1])
        print("optional", optional)
        g.remove_node(optional[removeables[opt1]][opt2])
    return g
        
def two_loop_reduction(g, opt1=0, opt2=0):
    removeables = []
    optional = []
    for node in g.node:
        p = g.node[node]["pin"]
        if p is None:
            neighbors = g.neighbors(node)
            if len(neighbors) == 2:
                removeables += [node]
                for temp_node in neighbors:
                    if g.node[temp_node]["pin"] is None:
                        optional += [temp_node]
                        print("optional", optional)
    if len(removeables) > 0:
        g.remove_node(removeables[opt1])
        if len(optional) != 0:
            try:
                g.remove_node(optional[opt2])
            except:
                pass
    return g