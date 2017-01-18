"""
A representation of a quantum circuit consiting of qubit initialisation (I),
controlled NOT gates (C) and measurements (M) with respect to different bases.
"""
import numpy as np
import json
from qutip import QubitCircuit, Qobj
from qutip.qip.circuit import Gate
from qutip.qip.gates import rx, ry, rz, snot, phasegate, cnot, toffoli

_icm_gate_dict = {("RZ", r"\pi/2"): "P",
                  ("RZ", r"\pi/4"): "T",
                  ("RX", r"\pi/2"): "V",
                  ("RZ", r"-\pi/2"): "P_dagger",
                  ("RZ", r"-\pi/4"): "T_dagger",
                  ("RX", r"-\pi/2"): "V_dagger"}


def pgate(targets=None, dagger=False):
    """
    The P gate. It is a rotation about RZ by pi/2.

    Parameters
    ----------
    targets: list
        A list of targets. Although this will be just one value but
        to maintain consistency with the `Gate` class, we pass a list.
    dagger : bool
        Return the P or P_dagger gate
        default: False

    Returns
    -------
    P: Gate
        The P gate
    """
    arg_value = np.pi / 2
    arg_label = r"\pi/2"

    if dagger:
        arg_value = - np.pi / 2
        arg_label = r"-\pi/2"

    return(Gate("RZ", targets=targets, arg_value=arg_value, arg_label=arg_label))


def tgate(targets=None, dagger=False):
    """
    The T gate. It is a rotation about RZ by pi/4.

    Parameters
    ----------
    targets: list
        A list of targets. Although this will be just one value but
        to maintain consistency with the `Gate` class, we pass a list.
    dagger : bool
        Return the T or T_dagger gate
        default: False

    Returns
    -------
    T: Gate
        The T or T_dagger gate.
    """
    arg_value = np.pi / 4
    arg_label = r"\pi/4"

    if dagger:
        arg_value = - np.pi / 4
        arg_label = r"-\pi/4"
    return(Gate("RZ", targets=targets, arg_value=arg_value, arg_label=arg_label))


def vgate(targets=None, dagger=False):
    """
    The V gate. It is a rotation about RX by pi/2.

    Parameters
    ----------
    targets: list
        A list of targets. Although this will be just one value but
        to maintain consistency with the `Gate` class, we pass a list.
    dagger : bool
        Return the V or V_dagger gate.
        default: False

    Returns
    -------
    V: Gate
        The V or V_dagger gate.
    """
    arg_value = np.pi / 2
    arg_label = r"\pi/2"

    if dagger:
        arg_value = - np.pi / 2
        arg_label = r"-\pi/2"
    return(Gate("RX", targets=targets, arg_value=arg_value, arg_label=arg_label))


def decompose_toffoli(qcircuit):
    """
    Decompose the TOFFOLI gates in qcircuit to CNOT, H, P and T gates.

    Parameters
    ----------
    qcircuit: QubitCircuit
        The circuit containing `TOFFOLI` gates.

    Returns
    -------
    decomposed_circuit: QubitCircuit
        The circuit with TOFFOLI gates decomposed.
    """
    decomposed_circuit = QubitCircuit(qcircuit.N, reverse_states=False)

    for idx, gate in enumerate(qcircuit.gates):
        if gate.name == "TOFFOLI":
            c1 = gate.controls[0]
            c2 = gate.controls[1]
            t = gate.targets[0]

            decomposed_circuit.add_gate(Gate("SNOT", targets=[t]))
            decomposed_circuit.add_gate(
                Gate("CNOT", targets=[t], controls=[c2]))
            decomposed_circuit.add_gate(tgate(targets=[t], dagger=True))
            decomposed_circuit.add_gate(
                Gate("CNOT", targets=[t], controls=[c1]))
            decomposed_circuit.add_gate(tgate(targets=[t], dagger=False))
            decomposed_circuit.add_gate(
                Gate("CNOT", targets=[t], controls=[c2]))
            decomposed_circuit.add_gate(tgate(targets=[t], dagger=True))
            decomposed_circuit.add_gate(
                Gate("CNOT", targets=[t], controls=[c1]))
            decomposed_circuit.add_gate(tgate(targets=[c2], dagger=True))
            decomposed_circuit.add_gate(tgate(targets=[t], dagger=False))
            decomposed_circuit.add_gate(Gate("SNOT", targets=[t]))
            decomposed_circuit.add_gate(
                Gate("CNOT", targets=[c2], controls=[c1]))
            decomposed_circuit.add_gate(tgate(targets=[c2], dagger=True))
            decomposed_circuit.add_gate(
                Gate("CNOT", targets=[c2], controls=[c1]))
            decomposed_circuit.add_gate(tgate(targets=[c1], dagger=False))
            decomposed_circuit.add_gate(pgate(targets=[c2], dagger=False))

        else:
            decomposed_circuit.add_gate(gate)

    return decomposed_circuit


def decompose_SNOT(qcircuit):
    """
    Decompose the SNOT gates in qcircuit to PVP.

    Parameters
    ----------
    qcircuit: QubitCircuit
        The circuit containing `SNOT` gates.

    Returns
    -------
    decomposed_circuit: QubitCircuit
        The circuit with SNOT gates decomposed as PVP
    """
    decomposed_circuit = QubitCircuit(qcircuit.N, reverse_states=False)

    for idx, gate in enumerate(qcircuit.gates):
        if gate.name == "SNOT":
            target = gate.targets
            decomposed_circuit.add_gate(pgate(targets=target))
            decomposed_circuit.add_gate(vgate(targets=target))
            decomposed_circuit.add_gate(pgate(targets=target))
        else:
            decomposed_circuit.add_gate(gate)

    return decomposed_circuit


class Icm(QubitCircuit):
    """
    A representation of a quantum circuit consisting entirely of qubit
    initialisations (I), a network of controlled (C) NOT gates and measurement
    (M) in different basis. According to [1], the transformation to
    the ICM representation provides a cannonical form for an exact and
    fault-tolerant, error corrected circuit needed for optimization prior to
    final implementation on a hardware model.

    References
    ----------
    .. [1] arXiv:1509.03962v1 [quant-ph]
    """

    def __init__(self, qcircuit):
        """
        Parameters
        ----------
        qcircuit: QubitCircuit
            A quantum circuit which is an instance of a QubitCircuit class.
        """
        self.qcircuit = qcircuit
        self.gates = qcircuit.gates
        self.icm = None
        self.inputs = []
        self.outputs = []

    def decompose_gates(self):
        """
        A function to decompose `qcircuit` in terms of the P, V, T, SNOT, CNOT
        and TOFFOLI gates which forms the starting point of ICM conversion.
        First we will use the `resolve_gate` function to get all the gates in
        terms of CNOT, SNOT, TOFFOLI, RX and RZ. Then we will re-name the
        rotation gates as (P (P_dagger), T (T_dagger), V (V_dagger)).

        Returns
        -------
        decomposed_circuit: QubitCircuit
        """
        resolved_circuit = self.qcircuit.resolve_gates(["RX", "RZ", "CNOT",
                                                        "SNOT", "TOFFOLI"])
        decomposed_circuit = QubitCircuit(resolved_circuit.N)

        # Check all gates and set argument label with necessary ICM gates

        for gate in resolved_circuit.gates:

            if gate.arg_label == r"\pi":
                decomposed_circuit.add_gate(gate.name,
                                            targets=gate.targets,
                                            controls=gate.controls,
                                            arg_value=np.pi / 2,
                                            arg_label=r'\pi/2')

                decomposed_circuit.add_gate(gate.name,
                                            targets=gate.targets,
                                            controls=gate.controls,
                                            arg_value=np.pi / 2,
                                            arg_label=r'\pi/2')

            elif gate.name == "GLOBALPHASE":
                decomposed_circuit.add_gate(gate.name,
                                            targets=gate.targets,
                                            controls=gate.controls,
                                            arg_value=gate.arg_value,
                                            arg_label=gate.arg_label)

            elif gate.name == "TOFFOLI":
                decomposed_circuit.add_gate(gate.name,
                                            targets=gate.targets,
                                            controls=gate.controls,
                                            arg_value=gate.arg_value,
                                            arg_label=gate.arg_label)
            elif gate.name == "CNOT":
                decomposed_circuit.add_gate(gate.name,
                                            targets=gate.targets,
                                            controls=gate.controls,
                                            arg_value=gate.arg_value,
                                            arg_label=gate.arg_label)

            elif gate.name == "SNOT":
                decomposed_circuit.add_gate(gate.name,
                                            targets=gate.targets,
                                            controls=gate.controls,
                                            arg_value=gate.arg_value,
                                            arg_label=gate.arg_label)

            elif (gate.name, gate.arg_label) not in _icm_gate_dict.keys():
                raise(ValueError)

            else:
                decomposed_circuit.add_gate(gate.name,
                                            targets=gate.targets,
                                            controls=gate.controls,
                                            arg_value=gate.arg_value,
                                            arg_label=gate.arg_label)
        return (decomposed_circuit)

    def ancilla_cost(self):
        """
        for ICM representation of a given quantum circuit decomposed into P, T, V
        SNOT and Toffoli gates. The P, T, V gates are implemented using ancilla
        qubits and gate teleportation requiring CNOT and measurement. Each T gate
        requires 5 ancillae and 6 CNOT gates. The P and V gates each require
        1 ancilla and 1 CNOT gate. Each Hadamard gate is implemented
        using a sequence of P and V gates requiring 3 extra ancillae and gates. The
        Toffoli gate requires 55 extra gates and 42 ancillae.

        Returns
        -------
        ancilla_cost: dict
            A dictionary which gives the ancilla count for each type of gate from
            the set (P (P_dagger), T (T_dagger), V (V_dagger), SNOT, TOFFOLI)

        References
        ----------
        .. [1] arXiv:1509.03962v1 [quant-ph]
        """
        decomposed_circuit = self.decompose_gates()
        cost = dict({"P": 0, "T": 0, "V": 0,
                             "SNOT": 0, "TOFFOLI": 0})

        for gate in decomposed_circuit.gates:
            if gate.name == "CNOT":
                continue

            elif gate.name == "SNOT":
                cost["SNOT"] += 3

            elif gate.name == "TOFFOLI":
                cost["TOFFOLI"] += 42

            else:
                try:
                    icm_gate = _icm_gate_dict[(gate.name, gate.arg_label)]
                except:
                    raise ValueError("Gate decomposition is not in correct ICM basis")

                if icm_gate == "P" or icm_gate == "P_dagger":
                    cost["P"] += 1

                if icm_gate == "T" or icm_gate == "T_dagger":
                    cost["T"] += 5

                if icm_gate == "V" or icm_gate == "V_dagger":
                    cost["V"] += 1

        return (cost)

    def to_icm(self):
        """
        A function to convert the initially decomposed circuit to the ICM model.
        We first resolve all the gates into TOFFOLI, SNOT, RX, RZ and CNOT gates.
        This is done by using `decompose_gates`. Using `_icm_gate_dict` for
        labeling of rotation gates as P, T, or V based on the angles, we get the
        initial ICM circuit. Then we use the algorithm outlined in [1]
        to implement each gate using ancilla qubits and gate teleportation. Ancillae
        cost can be calculated by the function `ancilla_cost`.

        Returns
        -------
        icm_circuit: QubitCircuit
            The ICM representation of the given quantum circuit.
            Converts `self.qcircuit` into a ICM representation.
        """
        decomposed_circuit = self.decompose_gates()
        ancilla_qubits = self.ancilla_cost()
        total_qubits = decomposed_circuit.N + sum(ancilla_qubits.values())
        icm_circuit = QubitCircuit(total_qubits, reverse_states=False)
        # Replace "TOFFOLI" and "SNOT" with their equivalent ICM representation
        decomposed_circuit = decompose_toffoli(decomposed_circuit)
        decomposed_circuit = decompose_SNOT(decomposed_circuit)

        for i in range(decomposed_circuit.N):
            icm_circuit.add_gate("IN", arg_label="input", targets=[i])

        for gate in decomposed_circuit.gates:
            icm_circuit.add_gate(gate)

        for i in range(decomposed_circuit.N):
            icm_circuit.add_gate("OUT", arg_label="output", targets=[i])


        # Implement the loop for all P and P_dagger gates
        # We consider the dagger operation to be the same here for now.
        # This needs to be looked at

        for idx, gate in enumerate(icm_circuit.gates):
            if gate.name == "CNOT":
                continue

            if gate.arg_label in ["ancilla", 'measurement', 'correction', 'input', 'output']:
                continue

            if _icm_gate_dict[(gate.name, gate.arg_label)] == "P" or _icm_gate_dict[(gate.name, gate.arg_label)] == "P_dagger":
                t = gate.targets[0]

                # Move targets and controls of all gates before the P gate
                for gate in icm_circuit.gates[:idx]:
                    if gate.arg_label in ["ancilla",
                                          "measurement",
                                          "correction"]:
                        continue

                    if gate.targets[0] > t:
                        gate.targets[0] += 1

                    if gate.controls and gate.controls[0] > t:
                        gate.controls[0] += 1

                icm_circuit.gates[idx] = Gate("Y", targets=[t + 1],
                               arg_label=r"ancilla")

                CNOT = Gate("CNOT", targets=[t], controls=[t + 1])
                measurement = Gate("z", targets=[t], arg_label=r"measurement")
                correction = Gate("xz", controls=[t], targets=[t + 1],
                                  arg_label=r"correction")

                icm_circuit.gates.insert(idx + 1, CNOT)
                icm_circuit.gates.insert(idx + 2, measurement)
                icm_circuit.gates.insert(idx + 3, correction)

                # Move targets and controls of all gates after the P gate

                for gate in icm_circuit.gates[idx+4:]:
                    if gate.targets[0] >= t:
                        gate.targets[0] += 1

                    if gate.controls and gate.controls[0] >= t:
                        gate.controls[0] += 1

        for idx, gate in enumerate(icm_circuit.gates):
            if gate.name == "CNOT":
                continue

            if gate.arg_label in ["ancilla", 'measurement', 'correction', 'input', 'output']:
                continue

            if _icm_gate_dict[(gate.name, gate.arg_label)] == "V" or _icm_gate_dict[(gate.name, gate.arg_label)] == "V_dagger":
                t = gate.targets[0]

                # Move targets and controls of all gates before the P gate
                icm_circuit.gates[idx] = Gate("y", targets=[t + 1],
                               arg_label=r"ancilla")

                for gate in icm_circuit.gates[:idx]:
                    if gate.arg_label in ["ancilla",
                                          "measurement",
                                          "correction"]:
                        continue


                    if gate.targets[0] > t:
                        gate.targets[0] += 1

                    if gate.controls and gate.controls[0] > t:
                        gate.controls[0] += 1


                CNOT = Gate("CNOT", targets=[t + 1], controls=[t])
                measurement = Gate("x", targets=[t], arg_label=r"measurement")
                correction = Gate("x/z", controls=[t], targets=[t + 1],
                                  arg_label=r"correction")

                icm_circuit.gates.insert(idx + 1, CNOT)
                icm_circuit.gates.insert(idx + 2, measurement)
                icm_circuit.gates.insert(idx + 3, correction)

                # Move targets and controls of all gates after the P gate

                for gate in icm_circuit.gates[idx+4:]:
                    if gate.targets[0] >= t:
                        gate.targets[0] += 1

                    if gate.controls and gate.controls[0] >= t:
                        gate.controls[0] += 1



        for idx, gate in enumerate(icm_circuit.gates):
            if gate.name == "CNOT":
                continue

            if gate.arg_label in ["ancilla", 'measurement', 'correction', 'input', 'output']:
                continue
        
            if _icm_gate_dict[(gate.name, gate.arg_label)] == "T" or _icm_gate_dict[(gate.name, gate.arg_label)] == "T_dagger":
                t = gate.targets[0]
                # Move targets and controls of all gates before the P gate
                for gate in icm_circuit.gates[:idx]:
                    if gate.arg_label in ["ancilla",
                                          "measurement",
                                          "correction"]:
                        continue

                    if gate.targets[0] > t:
                        gate.targets[0] += 5

                    if gate.controls and gate.controls[0] > t:
                        gate.controls[0] += 5

                icm_circuit.gates[idx] = Gate("a", targets=[t + 1],
                               arg_label=r"ancilla")
                icm_circuit.gates.insert(idx + 1, Gate("0", targets=[t + 2],
                               arg_label=r"ancilla"))
                icm_circuit.gates.insert(idx + 2, Gate("y", targets=[t + 3],
                               arg_label=r"ancilla"))
                icm_circuit.gates.insert(idx + 3, Gate("+", targets=[t + 4],
                               arg_label=r"ancilla"))
                icm_circuit.gates.insert(idx + 4, Gate("0", targets=[t + 5],
                               arg_label=r"ancilla"))

                icm_circuit.gates.insert(idx + 5, Gate("CNOT", targets=[t], controls=[t+1]))
                icm_circuit.gates.insert(idx + 6, Gate("CNOT", targets=[t + 2], controls=[t + 1]))
                icm_circuit.gates.insert(idx + 7, Gate("CNOT", targets=[t + 1], controls=[t + 3]))
                icm_circuit.gates.insert(idx + 8, Gate("CNOT", targets=[t + 2], controls=[t + 4]))
                icm_circuit.gates.insert(idx + 9, Gate("CNOT", targets=[t + 5], controls=[t + 3]))
                icm_circuit.gates.insert(idx + 10, Gate("CNOT", targets=[t+ 5], controls=[t + 4]))

                icm_circuit.gates.insert(idx + 11, Gate("z", targets=[t], arg_label="measurement"))
                icm_circuit.gates.insert(idx + 12, Gate("z/x", targets=[t + 1], controls=[t], arg_label="correction"))
                icm_circuit.gates.insert(idx + 13, Gate("x/z", targets=[t + 2], controls=[t+1], arg_label="correction"))
                icm_circuit.gates.insert(idx + 14, Gate("x/z", targets=[t + 3], controls=[t+2], arg_label="correction"))
                icm_circuit.gates.insert(idx + 15, Gate("z/x", targets=[t + 4], controls=[t+3], arg_label="correction"))

                # Move targets and controls of all gates after the T gate

                for gate in icm_circuit.gates[idx+16:]:
                    if gate.targets[0] >= t:
                        gate.targets[0] += 5

                    if gate.controls and gate.controls[0] >= t:
                        gate.controls[0] += 5
        return(icm_circuit)

def visualise(icm_circuit):
    bits = [i for i in range(icm_circuit.N)]
    inputs = []
    outputs = []

    initializations = []
    measurements = []

    cnots = []

    corrections = []

    pauli_tracked = QubitCircuit(icm_circuit.N, reverse_states=False)

    for gate in icm_circuit.gates:
        if gate.name == "CNOT":
            cnots += [(gate.controls[0], gate.targets[0])]
        if gate.arg_label == "ancilla":
            initializations += [(gate.targets[0], gate.name)]
        if gate.arg_label == "measurement":
            measurements += [(gate.targets[0], gate.name)]
        if gate.arg_label == "correction":
            corrections += [(gate.targets[0], gate.name, gate.controls[0])]
        if gate.arg_label == "input":
            inputs += [gate.targets[0]]
        if gate.arg_label == "output":
            outputs += [gate.targets[0]]

    for item in inputs:
        pauli_tracked.add_gate("IN", arg_label="input", targets=[item])

    for item in initializations:
        pauli_tracked.add_gate(item[1], arg_label="ancilla", targets=item[0])

    for item in cnots:
        pauli_tracked.add_gate("CNOT", controls=[item[0]], targets=[item[1]])

    for item in measurements:
        pauli_tracked.add_gate(item[1], arg_label="measurement", targets=[item[0]])

    for item in corrections:
        pauli_tracked.add_gate(item[1], arg_label="correction", controls=[item[0]-1], targets=[item[0]])

    for item in outputs:
        pauli_tracked.add_gate("OUT", arg_label="output", targets=[item])

    json_dict = {"bits": bits, "inputs": inputs, "outputs": outputs,
                 "initializations": [{"bit": item[0], "type": item[1]} for item in initializations],
                 "measurements": [{"bit": item[0], "type": item[1]} for item in measurements],
                "cnots": [{"controls": [item[0]], "targets": [item[1]]} for item in cnots]}

    return (pauli_tracked, json_dict)

def make_json_file(json_dict, fname="tqc"):
    """
    Write a json file
    """
    jsonData = {"format": "icm", "circuit": json_dict}
    with open(fname+".json", 'w') as outfile:
        json.dump(jsonData, outfile, sort_keys = False)

