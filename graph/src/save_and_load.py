import pickle
import networkx as nx
import json 

#-------------------------------------#
# Functions to save and load the Label Encoder
#-------------------------------------#
def save_encoder(encoder, name):
  # Save category_encoder
  with open(f'../models/{name}.pkl', 'wb') as f:
      pickle.dump(encoder, f)

def load_encoder(name):
  with open(f'../models/{name}.pkl', 'rb') as f:
      return pickle.load(f)
  
#-------------------------------------#
# Functions to save and load the graph
#-------------------------------------#
def save_graph(G, filename):
    # Convert the graph to a node-link format
    data = nx.node_link_data(G, edges="links")  # Preserve current behavior
    # Save the graph as a JSON file
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)  # Save as JSON
 
def load_graph(filename):
    with open(f'../models/{filename}', 'r') as f:
        data = json.load(f)  # Load JSON data
    G = nx.node_link_graph(data, edges="links")  # Reconstruct the graph
    return G