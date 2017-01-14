import networkx as nx
import matplotlib.pyplot as plt

def construct_icm_graph(qcircuit):
    graphs = []
    gate_count = 0
    for gate in qcircuit.gates:
        if gate.name == "CNOT":
            g = nx.Graph()

            control_bit = gate.controls[0]
            target_bit = gate.targets[0]
            
            rough = (gate_count, str(control_bit)+str(target_bit), "rough")
            cleft = (gate_count, control_bit, "cleft")
            cright = (gate_count, control_bit, "cright")
            target = (gate_count, target_bit, "target")

            g.add_node(cleft, color="r", pos = (gate_count - 0.25, - control_bit))
            g.add_node(cright, color="r", pos = (gate_count + 0.25, - control_bit))

            g.add_node(target, color="r", pos = (gate_count, - target_bit))
            g.add_node(rough, color="b", pos = (gate_count, - (target_bit + control_bit)/2.))

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
    pos=nx.get_node_attributes(g,'pos')
    nx.draw_networkx(g, node_color=node_color, pos = pos)
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
            try:
                g = nx.contracted_nodes(g, key, item) 
            except:
                print(item)
    for key in c_mergers:
        g = nx.contracted_nodes(g, key, c_mergers[key])
        
    return g