import pandas as pd


def dataset_parser(dataset):
# Function to read the tsv dataset file and returning it as a pandas dataframe
# Args: 
#       - dataset: is the tsv file containing the dataset

    # Read the tsv file using a pandas dataframe
    df = pd.read_csv(dataset, sep='\t')

    list_id = [] # A list of all wikidata id of the entities
    urls_col = df['item'] # select the column of each url

    # Taking the url of each data and append to list_id the ID of the wikidata item (Ex. "Q307")
    for url in urls_col:
        single_id = url.split("/")[-1] 
        list_id.append(single_id) 

    return df, list_id

