import pandas as pd


def dataset_parser(dataset, client, list_dict):
# Function to read the tsv dataset file and returning it as a pandas dataframe
# Args: 
#       - dataset: is the tsv file containing the dataset

    # Read the tsv file using a pandas dataframe
    original_df = pd.read_csv(dataset, sep='\t')

    # Dictionary used to map the labels into numbers
    label_map = {
    "cultural agnostic": 0,
    "cultural representative": 1,
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
                'properties': prop_list,
                'properties_names': prop_names_list,
                'label': label_map.get(label_, -1)
                       }
            }
        list_dict.append(single_item_dict)
        print(type(single_item_dict[single_id]['name']))

    return original_df, list_dict

def id2string(id_, client):
    # This function transform an ID (both for an entity of wikidata or a pid of a property) to the string
    # Args: 
    #       - id: the identifier
    #       - client: it is the wikidata client that get entity and properties
    # Output: 
    #       - str that identify the identity or properties 

    element_ = client.get(id_, load=True)
    return element_.label

def getEP(id_, client):
    prop_list = []
    prop_names_list = []
    entity = client.get(id_, load=True)
    for prop in entity.data['claims']:
        prop_list.append(prop)
        prop_name = id2string(prop, client)
        prop_names_list.append(prop_name)

    return entity, prop_list, prop_names_list


        






    


    

