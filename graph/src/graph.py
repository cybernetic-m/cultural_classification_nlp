import networkx as nx
#import pandas as pd
import numpy as np

def make_graph(G, df, add_label):
  #G = nx.Graph()
  cols_to_save = ["P17",  "P21", "P569", "P27", "P19", "P214", "P511", "P856", "P213", "P106", "P269", "P735", "P10832", "P1412", "P734", "P345", "P131", "P625", "P570", "P6366", "P8814", "P8408", "P2347", "P2002", "P159", "P1014"]

  cols = ['en', 'P17' , 'P279', 'subcategory_encoded', 'category_encoded', 'total_views'] + cols_to_save

  #binary_props = ['P17', 'P279']  + cols_to_save2  # solo queste vanno con valore 1
  make_nodes_by_val = ['en', 'total_views']
  split_val = 2000
  #binary_props = set(binary_props)

  if not set(cols).issubset(set(df.columns)):
    print(f"Missing required columns: {set(cols) - set(df.columns)}")

    for col in cols:
        if col not in df.columns:
            print(f"Adding '{col}' with default 0")
            df[col] = 0


  for _, row in df.iterrows():
      entity = row["qid"]
      if add_label:
          G.add_node(entity, label=row["label"])
      else:
          G.add_node(entity, label=-1)

      for prop in cols:
          val = row[prop]

          if prop in make_nodes_by_val: # spezzetto la linga inglese in vari nodi in base a quante visualizzazioni ha la lingua
            not_done = True
            start = 0
            i = 1
            while not_done:
              if start <= val < split_val * i:
                val = split_val*i
                not_done = False

                #print('added ', f"{prop}_{val}")
              else:
                i += 1
                start += split_val

          prop_node = f"{prop}_{val}"
          G.add_node(prop_node)
          G.add_edge(entity, prop_node)

def prepare_data(G):
    all_nodes = list(G.nodes())
    node_idx = {node: i for i, node in enumerate(all_nodes)}
    A = nx.to_scipy_sparse_array(G, format='csr')

    y = np.full(len(all_nodes), -1)  # Initialize with -1 for unlabeled nodes
    for node, i in node_idx.items():
        if "label" in G.nodes[node]:
            y[i] = G.nodes[node]["label"]  # Assign known labels
            
    return A, y, node_idx