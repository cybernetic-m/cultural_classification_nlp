import pickle
import networkx as nx
import json 
import os
#-------------------------------------#
# Functions to save and load the Label Encoder
#-------------------------------------#
def save_encoder(encoder, name):
  # Save category_encoder
  with open(f'../models/{name}.pkl', 'wb') as f:
      pickle.dump(encoder, f)

def load_encoder(name):
    script_dir = os.path.dirname(os.path.abspath(__file__))  # cartella dove si trova il file .py
    model_path = os.path.join(script_dir, '..', 'models', name)
    model_path = os.path.normpath(model_path)  # normalizza il percorso

    print("Carico da:", model_path)  # utile per debug
    with open(f'{model_path}.pkl', 'rb') as f:
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
    script_dir = os.path.dirname(os.path.abspath(__file__))  # cartella dove si trova il file .py
    model_path = os.path.join(script_dir, '..', 'models', filename)
    model_path = os.path.normpath(model_path)  # normalizza il percorso

    print("Carico da:", model_path)  # utile per debug
    with open(f'{model_path}.json', 'r') as f:
        data = json.load(f)  # Load JSON data
    G = nx.node_link_graph(data, edges="links")  # Reconstruct the graph
    return G