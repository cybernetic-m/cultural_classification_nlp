import networkx as nx
#import pandas as pd
import numpy as np

def make_graph(G, df, add_label):
  """
  Build or update a graph from a DataFrame.

  Inputs:
  - G: a networkx graph object where nodes and edges will be added
  - df: a pandas DataFrame containing entity information and properties
  - add_label: boolean, whether to add the label attribute to entity nodes, it must be false when we pass the validation and test data
  """
  #G = nx.Graph()

  # cols we chose to use
  cols_to_save = ["P17",  "P21", "P569", "P27", "P19", "P214", "P511", "P856", "P213", "P106", "P269", "P735", "P10832", "P1412", "P734", "P345", "P131", "P625", "P570", "P6366", "P8814", "P8408", "P2347", "P2002", "P159", "P1014"]
  # most important cols
  cols = ['en', 'P17' , 'P279', 'subcategory_encoded', 'category_encoded', 'total_views'] + cols_to_save


  # cols we want to split into multiple nodes based on the numeric value
  make_nodes_by_val = ['en', 'total_views']
  # value to split the make_nodes_by_val, 2000 = if en has value 2300 it will be connected to the node en_2000, if it has value 500 it will be connected to the node en_0
  # en_0 will be a node where the elements with en values from ' to 1999 will be connected
  split_val = 1000
  #binary_props = set(binary_props)

  # check for missing columns
  if not set(cols).issubset(set(df.columns)):
    print(f"Missing required columns: {set(cols) - set(df.columns)}")

    for col in cols:
        if col not in df.columns:
            print(f"Adding '{col}' with default 0")
            df[col] = 0


  for _, row in df.iterrows():
      entity = row["qid"]

      if add_label:
          # label added only if add_labels = True, which means we are passing training data
          G.add_node(entity, label=row["label"])
      else:
          G.add_node(entity, label=-1)

      for prop in cols:
          val = row[prop]

          if prop in make_nodes_by_val:
            not_done = True
            start = 0
            i = 1
            # this code will split the cols in make_nodes_by_val into many nodes according to the value
            while not_done:
              if start <= val < split_val * i:
                val = split_val*i
                not_done = False

                #print('added ', f"{prop}_{val}")
              else:
                i += 1
                start += split_val

          # nodes and edge for the cols, the code will automatically make a new node or connect to an existing node if the name exist
          prop_node = f"{prop}_{val}"
          G.add_node(prop_node)
          G.add_edge(entity, prop_node)

def prepare_data(G):
    """
    Prepare adjacency matrix and label vector from a graph.

    Inputs:
    - G: a networkx graph object containing entity and property nodes

    Outputs:
    - A: scipy sparse adjacency matrix (CSR format)
    - y: numpy array of labels (-1 if unknown)
    - node_idx: dictionary mapping node identifiers to matrix indices
    """
    all_nodes = list(G.nodes())

    # Create a mapping from node name to index
    node_idx = {node: i for i, node in enumerate(all_nodes)}
    # Generate the adjacency matrix
    A = nx.to_scipy_sparse_array(G, format='csr')

    # Create label array initialized to -1 (unknown)
    y = np.full(len(all_nodes), -1)
    for node, i in node_idx.items():
        if "label" in G.nodes[node]:
            y[i] = G.nodes[node]["label"]  # Assign known labels
            
    return A, y, node_idx