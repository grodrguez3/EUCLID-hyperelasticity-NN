import pandas as pd
import re
import torch
import xml.etree.ElementTree as ET



def read_multi_ste_output2(file, p=False):
    """
    Reads a STE output CSV where the first column is unused, and the remaining
    columns are grouped per-state either as:
      • 3‐field: [x, y, z]
      • 6‐field: [xx, yy, zz, xy, xz, yz]
      • 9‐field: [F11, F12, F13, F21, F22, F23, F31, F32, F33]
    Returns a dict:
      {
        1: {'x': […], 'y': […], 'z': […]},
        2: {...},
        …
      }
    or for 6‐field:
      {
        1: {'xx': […], …, 'yz': […]},
        …
      }
    or for 9‐field:
      {
        1: {'F11': […], …, 'F33': […]},
        …
      }
    """
    # 1) Load & drop the first (unused) column
    df = pd.read_csv(file, header=None)
    df = df.drop(columns=0)

    # 2) Determine whether it's a 3-, 6- or 9-field file
    ncols = df.shape[1]
    if   ncols % 9 == 0:
        fields = ['F11','F12','F13',
                  'F21','F22','F23',
                  'F31','F32','F33']
    elif ncols % 6 == 0:
        fields = ['xx','yy','zz','xy','xz','yz']
    elif ncols % 3 == 0:
        fields = ['x','y','z']
    else:
        fields = ['element']
        #raise ValueError(f"Unexpected number of data columns: {ncols}")

    n_fields = len(fields)
    n_states = ncols // n_fields

    # 3) Build the per‐state dict
    state_dict = {}
    for i in range(n_states):
        block = df.iloc[:, i*n_fields : (i+1)*n_fields]
        state_dict[i+1] = {
            fields[j]: block.iloc[:, j].tolist()
            for j in range(n_fields)
        }

    # 4) Optional sanity print
    if p:
        print(f"Detected {n_fields}-field blocks → {fields}")
        print(f"Number of states: {n_states}")
        for s, data in state_dict.items():
            counts = ", ".join(f"{f}={len(data[f])}" for f in fields)
            print(f" State {s}: {counts}")

    return state_dict


def read_multi_stepped_output_txt(file, p=False):
    """
    Reads a file containing repeated blocks of the form:
      *Step  = <step_idx>
      *Time  = <time>
      *Data  = <field1>;<field2>;...;<fieldN>
      <id> val1 val2 ... valN
      ...
    and returns a dict:
      {
        <step_idx>: {
           '<field1>': [val1, val1, ...],
           '<field2>': [val2, val2, ...],
            …
        },
        …
      }
    """
    result = {}
    with open(file, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        raw = lines[i].strip()
        # strip leading '*' so we catch '*Step' or 'Step'
        line = raw.lstrip('*').strip()
        if line.startswith('Step'):
            # parse step index
            step = int(line.split('=', 1)[1])
            # skip "*Time" line
            # parse "*Data" line
            data_line = lines[i+2].strip().lstrip('*')
            _, fields_str = data_line.split('=', 1)
            fields = [fld.strip() for fld in fields_str.split(';') if fld.strip()]
            n_fields = len(fields)

            # collect data rows
            state_data = {fld: [] for fld in fields}
            j = i + 3
            while j < len(lines):
                ln = lines[j].strip()
                if not ln or ln.lstrip('*').strip().startswith('Step'):
                    break
                parts = ln.split()
                vals = parts[1:1+n_fields]
                for fld, v in zip(fields, vals):
                    state_data[fld].append(float(v))
                j += 1

            result[step] = state_data
            i = j
        else:
            i += 1

    if p:
        for step, data in result.items():
            counts = ", ".join(f"{fld}={len(vals)}" for fld, vals in data.items())
            print(f"Step {step}: {counts}")

    return result


def read_multi_ste_output_VOIGT(file, p=False):
    """
    Reads a STE output CSV where the first column is unused, and the remaining
    columns are grouped per-state either as:
      • 3‐field: [x, y, z]
      • 6‐field: [xx, yy, zz, xy, yz, xz]
      • 9‐field: [F11, F12, F13, F21, F22, F23, F31, F32, F33]
    Returns a dict:
      {
        1: {'x': […], 'y': […], 'z': […]},
        2: {...},
        …
      }
    or for 6‐field:
      {
        1: {'xx': […], …, 'yz': […]},
        …
      }
    or for 9‐field:
      {
        1: {'F11': […], …, 'F33': […]},
        …
      }
    """
    # 1) Load & drop the first (unused) column
    df = pd.read_csv(file, header=None)
    df = df.drop(columns=0)

    # 2) Determine whether it's a 3-, 6- or 9-field file
    ncols = df.shape[1]
    if   ncols % 9 == 0:
        fields = ['F11','F12','F13',
                  'F21','F22','F23',
                  'F31','F32','F33']
    elif ncols % 6 == 0:
        fields = ['xx','yy','zz','xy','yz','xz'] #https://help.febio.org/FEBioTheory/FEBio_tm_3-4-Section-2.1.html
    elif ncols % 3 == 0:
        fields = ['x','y','z']
    else:
        fields = ['element']
        #raise ValueError(f"Unexpected number of data columns: {ncols}")

    n_fields = len(fields)
    n_states = ncols // n_fields

    # 3) Build the per‐state dict
    state_dict = {}
    for i in range(n_states):
        block = df.iloc[:, i*n_fields : (i+1)*n_fields]
        state_dict[i+1] = {
            fields[j]: block.iloc[:, j].tolist()
            for j in range(n_fields)
        }

    # 4) Optional sanity print
    if p:
        print(f"Detected {n_fields}-field blocks → {fields}")
        print(f"Number of states: {n_states}")
        for s, data in state_dict.items():
            counts = ", ".join(f"{f}={len(data[f])}" for f in fields)
            print(f" State {s}: {counts}")

    return state_dict









def parse_states(filename):
    """
    Parses a file with repeating blocks like:

      Data Record #1
      ===========================================================================
      Step = 0
      Time = 0
      Data = x
      1   0.000
      2   0.000
      …
      Data Record #2
      ===========================================================================
      Step = 0
      Time = 0
      Data = y
      1   0.010
      …
      Data Record #3
      …
      Data = z
      1   0.020
      …

      Data Record #1       ← next step block → Step = 1, Data = x, …
      …

    Returns a dict
       { 0: {'x':[...], 'y':[...], 'z':[...]},
         1: {'x':[...], 'y':[...], 'z':[...]},
         … }
    """
    # regexes
    re_record = re.compile(r'^Data\s+Record\s*#\s*(\d+)', re.IGNORECASE)
    re_step   = re.compile(r'^\s*Step\s*[=:]\s*(\d+)', re.IGNORECASE)
    re_value  = re.compile(r'^\s*\d+\s+([+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)')

    # map record number → variable name
    record_to_var = {1: 'x', 2: 'y', 3: 'z'}

    states = {}
    current_state = None
    current_var   = None
    seen_first_record = False

    with open(filename, 'r') as f:
        for raw in f:
            line = raw.strip()

            # 1) Skip everything until the first Data Record #
            if not seen_first_record:
                m = re_record.match(line)
                if m:
                    seen_first_record = True
                    # fall through so we also handle this record line
                else:
                    continue

            # 2) Detect “Data Record #N” → choose var x/y/z
            m = re_record.match(line)
            if m:
                rec_num = int(m.group(1))
                current_var = record_to_var.get(rec_num, None)
                continue

            # 3) Detect “Step = M” → set current state, but only init once
            m = re_step.match(line)
            if m:
                idx = int(m.group(1))
                current_state = idx
                if idx not in states:
                    states[idx] = {'x': [], 'y': [], 'z': []}
                # don’t reset current_var here
                continue

            # 4) If we have a state *and* a var, try to read a data line
            if current_state is not None and current_var is not None:
                m = re_value.match(raw)
                if m:
                    val = float(m.group(1))
                    states[current_state][current_var].append(val)
                # else: skip separators or “Time = …” lines

    return states

def states_to_tensor(state_dict, var_order):
    """
    state_dict: dict[state_idx → dict[var_name→list_of_vals]]
       e.g. {0: {'x':[...], 'y':[...], 'z':[...]}, 1: {...}, …}
    var_order: list of the keys in the order you want them as channels
       e.g. ['x','y','z']  or ['sxx','syy','szz','sxy','sxz','syz'] or ['x']

    Returns:
      tensor of shape (Nelements, len(var_order), nstates)
    """
    # sort states by index
    states = sorted(state_dict.keys())
    nstates   = len(states)
    # assume every state has the same number of elements
    first     = state_dict[states[0]][var_order[0]]
    Nelements = len(first)

    tensor = torch.zeros(Nelements, len(var_order), nstates, dtype=torch.float32)
    for t, step in enumerate(states):
        for c, var in enumerate(var_order):
            vals = state_dict[step][var]
            if len(vals) != Nelements:
                raise ValueError(f"Step {step} var {var} has {len(vals)} elems, expected {Nelements}")
            tensor[:, c, t] = torch.tensor(vals, dtype=torch.float32)
    return tensor


def parse_quad4_from_feb(feb_file, surface_name=None):
    """
    Parse a .feb (XML) file and extract all <quad4> elements.

    Parameters
    ----------
    feb_file : str
        Path to your .feb file.
    surface_name : str or None
        If given, only <Surface name="..."> matching this will be parsed.
        If None, all <quad4> in any <Surface> are returned.

    Returns
    -------
    quad_dict : dict[int, list[int]]
        Keys are the quad4 `id` attributes (as ints), values are lists of node IDs.
    """
    tree = ET.parse(feb_file)
    root = tree.getroot()

    quad_dict = {}
    # find every <Surface>
    for surface in root.findall(".//Surface"):
        if surface_name and surface.get("name") != surface_name:
            continue

        # for each quad4 child
        for quad in surface.findall("quad4"):
            # read the id attribute
            qid = int(quad.get("id"))
            # split the text "5,35,522,108" → ["5","35",…] → [5,35,…]
            nodes = [int(n) for n in quad.text.strip().split(",")]
            quad_dict[qid] = nodes

    return quad_dict


def map_pressure_to_elements(connectivity, pressure_nodes):
    """
    Parameters
    ----------
    connectivity : list of tuples
        Each tuple is (elem_id, node1, node2, node3, ..., nodeK).
    pressure_nodes : set or list of ints
        Node IDs that carry a pressure boundary condition.

    Returns
    -------
    mapping : dict
        { elem_id: {
            'all_nodes':    [n1, n2, …, nK],
            'pressure_nodes': [ni, …]   # only those nodes in this element with pressure
          }
        }
        Only elements with at least one pressure node are included.
    """
    # convert to set for O(1) membership tests
    P = set(pressure_nodes)

    mapping = {}
    for entry in connectivity:
        elem_id, *nodes = entry
        applied = [n for n in nodes if n in P]
        if applied:
            mapping[elem_id] = {
                'all_nodes':     nodes,
                'pressure_nodes': applied
            }
    return mapping


def map_facets_to_elements(connectivity, facets):
    """
    Parameters
    ----------
    connectivity : list of tuples
        Each tuple is (elem_id, node1, node2, ..., nodeK).
    facets : list of tuples
        Each tuple is (facet_id, n1, n2, n3, n4).

    Returns
    -------
    mapping : dict
        {
          facet_id: {
            'facet_nodes':    [n1, n2, n3, n4],
            'element_ids':    [e1, e2, …]    # all elements that contain these 4 nodes
          },
          …
        }
        Only facets that are found in at least one element are included.
    """
    mapping = {}
    for facet_entry in facets:
        facet_id, *f_nodes = facet_entry
        fset = set(f_nodes)

        # find all elems whose node-set contains these 4 nodes
        matches = []
        for elem_entry in connectivity:
            elem_id, *e_nodes = elem_entry
            if fset.issubset(e_nodes):
                matches.append(elem_id)

        if matches:
            mapping[facet_id] = {
                'facet_nodes':   f_nodes,
                'element_ids':   matches
            }

    return mapping