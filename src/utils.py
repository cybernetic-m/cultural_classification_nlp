import pandas as pd
import json 


def dataset_parser(dataset, client):

# Function to read the tsv dataset file and returning it as a pandas dataframe (and save it into a json file)
# Args:
#       - dataset: it is the tsv file containing the dataset
#       - client: it is the wikidata client
# Output:
#       - original_df: it is the it is the original dataframe
#       - my_df: return the dataframe modified with properties in one hot encoding mode

    # Read the tsv file using a pandas dataframe
    original_df = pd.read_csv(dataset, sep='\t')
    list_dict = [] # Initialization of the empty dictionary

    # Dictionary used to map the labels into numbers
    label_map = {
    "cultural agnostic": 0,
    "cultural ag": 0,
    "cultural agn": 0,
    "cultural represent": 1,
    "cultural representative": 1,
    "cultural ex": 2,
    "cultural exclusive": 2
    }

    urls_col = original_df['item'] # select the column of each url
    label_col = original_df['label'] # select the column of the label
    name_col = original_df['name'] # select the column of the label

    # Taking the url of each data and append to list_id the ID of the wikidata item (Ex. "Q307")
    for url, label_ in zip(urls_col, label_col):
        single_item_dict = {} # Initialization of a local dict void
        single_id = url.split("/")[-1] # "Q207"
        name_id = id2string(single_id, client) # "George W. Bush"
        entity, prop_list, prop_names_list = getEP(single_id, client)
        # Creation of the dict referred to the single_id item to append to the list of dictionaries
        single_item_dict = {
            single_id:{
                'name':name_id,
                'properties':{
                'id': prop_list,
                'name': prop_names_list,
                },
                'label': label_map.get(label_, -1),
                       }
            }
        list_dict.append(single_item_dict)
    
    dict_save_and_load(list_dict, './entity_properties.json', todo='save')
    return original_df

def id2string(id_, client):

    # This function transform an ID (both for an entity of wikidata or a pid of a property) to the string
    # Args:
    #       - id: the identifier
    #       - client: it is the wikidata client that get entity and properties
    # Output:
    #       - str that identify the identity or properties

    element_ = client.get(id_, load=True)
    return element_.label.get('en')

def getEP(id_, client):

  # This function return the Entity and Property given one item id
  # Args:
  #       - id_: it is the id_ of the item (Ex. "Q207")
  #       - client: it is the wikidata client
  # Output:
  #       - entity: it is the entity object
  #       - prop_list: it is a list of all its properties id ["P207", ...]
  #       - prop_names_list: it is a list of all its property names ["subclass of", ...]

    prop_list = [] # Initialization of the list of properties id
    prop_names_list = [] # Initialization of the list of properties names
    entity = client.get(id_, load=True) # get the entity using wikidata

    # Loop for selection of all properties
    for prop in entity.data['claims']:
        prop_list.append(prop)
        prop_name = id2string(prop, client)
        prop_names_list.append(prop_name)

    return entity, prop_list, prop_names_list

def dict_save_and_load(mydict, path, todo):

  # This function can save or load dictionaries in json format
  # Args:
  #       - mydict: it is the python dictionary that you want to save (only needed if todo=='save')
  #       - path: it is the wikidata client that get entity and properties
  #       - todo: it can be 'load' in case of loading or 'save' in the case of saving
  # Output:
  #       - mydict: in case of todo=='load', return the loaded dictionary

  if todo=="save":
    # Save it to a JSON file
    with open(path, "w", encoding="utf-8") as f:
      json.dump(mydict, f, ensure_ascii=False, indent=4)

  elif todo=="load":
    # Load the JSON file
    with open(path, "r", encoding="utf-8") as f:
      mydict = json.load(f)
    return mydict